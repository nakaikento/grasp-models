#!/usr/bin/env python3
"""
MarianMT 学習スクリプト v2

昨日の反省を踏まえた改善版：
- クリーニング済みデータ使用（splits_v2）
- L4最適化（バッチサイズ、勾配累積）
- より頻繁な評価・保存
- chrF++も評価に追加

Usage（RunPod L4想定）:
    python scripts/train_v2.py

Output:
    models/ko-ja-v2/
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Optional
import shutil

from datasets import Dataset, DatasetDict
from transformers import (
    MarianConfig,
    MarianMTModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
import sentencepiece as spm
import evaluate

# ========== 設定 ==========

# データパス
DATA_DIR = Path("data/splits_qwen72b_20260215")
TOKENIZER_PATH = "data/tokenized/spm.model"
OUTPUT_DIR = Path("models/marian-ko-ja-qwen72b-20260215")

# モデル設定（前回と同じ）
MODEL_CONFIG = {
    "encoder_layers": 6,
    "decoder_layers": 6,
    "d_model": 512,
    "encoder_ffn_dim": 2048,
    "decoder_ffn_dim": 2048,
    "encoder_attention_heads": 8,
    "decoder_attention_heads": 8,
    "max_position_embeddings": 512,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "activation_dropout": 0.1,
    "activation_function": "gelu",
    "pad_token_id": 0,
    "eos_token_id": 3,
    "decoder_start_token_id": 0,  # 重要: config.json準拠
    "static_position_embeddings": False,
}

# 学習設定（L4最適化）
TRAIN_CONFIG = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 64,  # L4 24GB VRAM
    "per_device_eval_batch_size": 128,
    "gradient_accumulation_steps": 2,   # 有効バッチ128
    "learning_rate": 3e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "fp16": True,
    
    # 評価・保存（より頻繁に）
    "eval_strategy": "steps",
    "eval_steps": 2000,
    "save_strategy": "steps",
    "save_steps": 2000,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_bleu",
    "greater_is_better": True,
    
    # 生成設定
    "generation_max_length": 128,
    "generation_num_beams": 4,
    
    # Early stopping
    "early_stopping_patience": 5,
    
    # その他
    "logging_steps": 100,
    "dataloader_num_workers": 4,
    "max_length": 128,
}


# ========== トークナイザー ==========

class SPMTokenizer:
    """SentencePieceベースのトークナイザー"""
    
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.model_path = model_path
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        
        self.vocab_size = self.sp.get_piece_size()
        self.padding_side = "right"
    
    def save_pretrained(self, save_directory):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        dest_path = save_dir / "spm.model"
        with open(dest_path, "wb") as f:
            f.write(self.sp.serialized_model_proto())
    
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        
        max_length = kwargs.get('max_length', 512)
        padding = kwargs.get('padding', False)
        truncation = kwargs.get('truncation', False)
        return_tensors = kwargs.get('return_tensors', None)
        
        input_ids = []
        attention_mask = []
        
        for text in texts:
            ids = self.sp.encode_as_ids(text)
            ids = ids + [self.eos_token_id]
            
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            
            mask = [1] * len(ids)
            input_ids.append(ids)
            attention_mask.append(mask)
        
        if padding:
            max_len = max(len(ids) for ids in input_ids)
            for i in range(len(input_ids)):
                pad_len = max_len - len(input_ids[i])
                input_ids[i] = input_ids[i] + [self.pad_token_id] * pad_len
                attention_mask[i] = attention_mask[i] + [0] * pad_len
        
        result = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        if return_tensors == 'pt':
            result = {k: torch.tensor(v) for k, v in result.items()}
        
        return result
    
    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
        
        valid_ids = []
        for i in ids:
            i = int(i)
            if 0 <= i < self.vocab_size:
                if skip_special_tokens and i in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                    continue
                valid_ids.append(i)
        
        if not valid_ids:
            return ""
        
        return self.sp.decode_ids(valid_ids)
    
    def batch_decode(self, batch_ids, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]
    
    def pad(self, features, padding=True, max_length=None, return_tensors=None, **kwargs):
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f.get('attention_mask') for f in features]
        labels = [f.get('labels') for f in features]
        
        max_len = max(len(ids) for ids in input_ids)
        if labels and labels[0] is not None:
            max_label_len = max(len(l) for l in labels)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i, ids in enumerate(input_ids):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
            if attention_mask and attention_mask[i] is not None:
                padded_attention_mask.append(attention_mask[i] + [0] * pad_len)
            if labels and labels[i] is not None:
                label_pad_len = max_label_len - len(labels[i])
                padded_labels.append(labels[i] + [-100] * label_pad_len)
        
        result = {'input_ids': padded_input_ids}
        if padded_attention_mask:
            result['attention_mask'] = padded_attention_mask
        if padded_labels:
            result['labels'] = padded_labels
        
        if return_tensors == 'pt':
            result = {k: torch.tensor(v) for k, v in result.items()}
        
        return result


# ========== データ ==========

def load_data(data_dir: Path) -> DatasetDict:
    """データをロード"""
    dataset_dict = {}
    
    for split in ['train', 'val', 'test']:
        ko_path = data_dir / f"{split}.ko"
        ja_path = data_dir / f"{split}.ja"
        
        with open(ko_path, 'r', encoding='utf-8') as f:
            ko_lines = [line.strip() for line in f]
        with open(ja_path, 'r', encoding='utf-8') as f:
            ja_lines = [line.strip() for line in f]
        
        dataset_dict[split if split != 'val' else 'validation'] = Dataset.from_dict({
            'ko': ko_lines,
            'ja': ja_lines
        })
    
    return DatasetDict(dataset_dict)


def preprocess_function(examples, tokenizer, max_length=128):
    """前処理"""
    inputs = tokenizer(
        examples['ko'],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    
    targets = tokenizer(
        examples['ja'],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    
    inputs['labels'] = targets['input_ids']
    return inputs


# ========== 評価 ==========

def compute_metrics(eval_preds, tokenizer):
    """評価メトリクス計算（BLEU + chrF++）"""
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")
    
    preds, labels = eval_preds
    
    # 範囲外の値をpad_token_idに置換
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    preds = np.where(preds >= tokenizer.vocab_size, tokenizer.pad_token_id, preds)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # デコード
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 空文字列を除外
    filtered = [(p, l) for p, l in zip(decoded_preds, decoded_labels) if p.strip() and l.strip()]
    if not filtered:
        return {"bleu": 0.0, "chrf": 0.0}
    
    decoded_preds, decoded_labels = zip(*filtered)
    
    # BLEU
    bleu_result = bleu_metric.compute(
        predictions=list(decoded_preds),
        references=[[label] for label in decoded_labels]
    )
    
    # chrF++
    chrf_result = chrf_metric.compute(
        predictions=list(decoded_preds),
        references=[[label] for label in decoded_labels],
        word_order=2
    )
    
    return {
        "bleu": bleu_result["score"],
        "chrf": chrf_result["score"],
    }


# ========== メイン ==========

def main():
    print("=" * 60)
    print("MarianMT Training v2 - 韓国語 → 日本語")
    print("=" * 60)
    
    # デバイス確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # トークナイザー
    print(f"\nLoading tokenizer: {TOKENIZER_PATH}")
    tokenizer = SPMTokenizer(TOKENIZER_PATH)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # データ
    print(f"\nLoading data: {DATA_DIR}")
    if not DATA_DIR.exists():
        print(f"❌ Error: {DATA_DIR} not found!")
        print("Run 'python scripts/clean_and_prepare.py' first.")
        sys.exit(1)
    
    dataset = load_data(DATA_DIR)
    print(f"  Train: {len(dataset['train']):,}")
    print(f"  Val:   {len(dataset['validation']):,}")
    print(f"  Test:  {len(dataset['test']):,}")
    
    # 前処理
    print("\nTokenizing...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, TRAIN_CONFIG["max_length"]),
        batched=True,
        remove_columns=['ko', 'ja'],
        desc="Tokenizing"
    )
    
    # モデル作成
    print("\nCreating model...")
    config = MarianConfig(
        vocab_size=tokenizer.vocab_size,
        **MODEL_CONFIG
    )
    model = MarianMTModel(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )
    
    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        
        # バッチ
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
        
        # 学習率
        learning_rate=TRAIN_CONFIG["learning_rate"],
        lr_scheduler_type=TRAIN_CONFIG["lr_scheduler_type"],
        warmup_ratio=TRAIN_CONFIG["warmup_ratio"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        
        # エポック
        num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
        
        # 評価・保存
        eval_strategy=TRAIN_CONFIG["eval_strategy"],
        eval_steps=TRAIN_CONFIG["eval_steps"],
        save_strategy=TRAIN_CONFIG["save_strategy"],
        save_steps=TRAIN_CONFIG["save_steps"],
        save_total_limit=TRAIN_CONFIG["save_total_limit"],
        load_best_model_at_end=TRAIN_CONFIG["load_best_model_at_end"],
        metric_for_best_model=TRAIN_CONFIG["metric_for_best_model"],
        greater_is_better=TRAIN_CONFIG["greater_is_better"],
        
        # 生成（評価用）
        predict_with_generate=True,
        generation_max_length=TRAIN_CONFIG["generation_max_length"],
        generation_num_beams=TRAIN_CONFIG["generation_num_beams"],
        
        # その他
        fp16=TRAIN_CONFIG["fp16"] and device == "cuda",
        dataloader_num_workers=TRAIN_CONFIG["dataloader_num_workers"] if device == "cuda" else 0,
        logging_steps=TRAIN_CONFIG["logging_steps"],
        report_to=["none"],
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=TRAIN_CONFIG["early_stopping_patience"]
            ),
        ],
    )
    
    # 学習開始
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Epochs: {TRAIN_CONFIG['num_train_epochs']}")
    print(f"  Batch size: {TRAIN_CONFIG['per_device_train_batch_size']} x {TRAIN_CONFIG['gradient_accumulation_steps']} = {TRAIN_CONFIG['per_device_train_batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps']}")
    print(f"  Learning rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        print(f"Checkpoints saved to: {OUTPUT_DIR}")
        return
    
    # 保存
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 最終評価
    print("\nEvaluating on test set...")
    results = trainer.evaluate(tokenized_dataset['test'])
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Test BLEU:  {results['eval_bleu']:.2f}")
    print(f"Test chrF++: {results['eval_chrf']:.2f}")
    print(f"Output: {OUTPUT_DIR}")
    
    print("\nNext steps:")
    print(f"  1. ONNX: python training/convert_to_onnx.py --model-dir {OUTPUT_DIR}")
    print(f"  2. INT8: python training/quantize_onnx.py --model-dir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
