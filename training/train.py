#!/usr/bin/env python3
"""
MarianMT 学習スクリプト

Knowledge Distillation: 日本語 → 教師韓国語 で学習

Usage:
    python training/train.py
    python training/train.py --use-opus-target  # OPUS韓国語をターゲットに（比較用）

Input:
    data/splits/train.ja
    data/teacher/train.ko (教師翻訳) or data/splits/train.ko (OPUS)
Output:
    models/ja-ko/
"""

import os
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import argparse

from datasets import Dataset, DatasetDict
from transformers import (
    MarianConfig,
    MarianMTModel,
    MarianTokenizer,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
import sentencepiece as spm
import evaluate

import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.config import ModelConfig, TrainingConfig


class SPMTokenizer:
    """SentencePieceベースのトークナイザー（MarianMT互換）"""
    
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
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
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        if return_tensors == 'pt':
            result = {k: torch.tensor(v) for k, v in result.items()}
        
        return result
    
    def decode(self, ids, skip_special_tokens=True):
        """デコード（範囲外ID対応）"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
        
        # 範囲内の有効なIDのみ抽出
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
        input_ids = [f['input_ids'] for f in features] if isinstance(features, list) else features['input_ids']
        attention_mask = [f.get('attention_mask') for f in features] if isinstance(features, list) else features.get('attention_mask')
        labels = [f.get('labels') for f in features] if isinstance(features, list) else features.get('labels')
        
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


def load_data(data_dir: Path, teacher_dir: Optional[Path], use_opus_target: bool = False) -> DatasetDict:
    with open(data_dir / "train.ja", 'r', encoding='utf-8') as f:
        train_ja = [line.strip() for line in f]
    with open(data_dir / "val.ja", 'r', encoding='utf-8') as f:
        val_ja = [line.strip() for line in f]
    with open(data_dir / "test.ja", 'r', encoding='utf-8') as f:
        test_ja = [line.strip() for line in f]
    
    if use_opus_target:
        print("ターゲット: OPUS韓国語")
        ko_train_path = data_dir / "train.ko"
    else:
        print("ターゲット: 教師翻訳（NLLB/M2M100）")
        ko_train_path = teacher_dir / "train.ko"
    
    with open(ko_train_path, 'r', encoding='utf-8') as f:
        train_ko = [line.strip() for line in f]
    with open(data_dir / "val.ko", 'r', encoding='utf-8') as f:
        val_ko = [line.strip() for line in f]
    with open(data_dir / "test.ko", 'r', encoding='utf-8') as f:
        test_ko = [line.strip() for line in f]
    
    dataset = DatasetDict({
        'train': Dataset.from_dict({'ja': train_ja, 'ko': train_ko}),
        'validation': Dataset.from_dict({'ja': val_ja, 'ko': val_ko}),
        'test': Dataset.from_dict({'ja': test_ja, 'ko': test_ko}),
    })
    
    print(f"Train: {len(dataset['train']):,}")
    print(f"Val:   {len(dataset['validation']):,}")
    print(f"Test:  {len(dataset['test']):,}")
    
    return dataset


def create_model(config: ModelConfig, vocab_size: int) -> MarianMTModel:
    marian_config = MarianConfig(
        vocab_size=vocab_size,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        encoder_attention_heads=config.encoder_attention_heads,
        decoder_attention_heads=config.decoder_attention_heads,
        d_model=config.d_model,
        encoder_ffn_dim=config.encoder_ffn_dim,
        decoder_ffn_dim=config.decoder_ffn_dim,
        max_position_embeddings=config.max_position_embeddings,
        dropout=config.dropout,
        attention_dropout=config.attention_dropout,
        activation_dropout=config.activation_dropout,
        activation_function=config.activation_function,
        pad_token_id=config.pad_token_id,
        eos_token_id=config.eos_token_id,
        decoder_start_token_id=config.decoder_start_token_id,
        static_position_embeddings=config.static_position_embeddings,
    )
    
    model = MarianMTModel(marian_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"モデルパラメータ数: {num_params:,} ({num_params/1e6:.1f}M)")
    
    return model


def preprocess_function(examples, tokenizer, max_length=128):
    inputs = tokenizer(
        examples['ja'],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    
    targets = tokenizer(
        examples['ko'],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    
    inputs['labels'] = targets['input_ids']
    return inputs


def compute_metrics(eval_preds, tokenizer):
    metric = evaluate.load("sacrebleu")
    
    preds, labels = eval_preds
    
    # 範囲外の値をpad_token_idに置換
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    preds = np.where(preds >= tokenizer.vocab_size, tokenizer.pad_token_id, preds)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 空文字列を除外
    filtered = [(p, l) for p, l in zip(decoded_preds, decoded_labels) if p.strip() and l.strip()]
    if not filtered:
        return {"bleu": 0.0}
    
    decoded_preds, decoded_labels = zip(*filtered)
    
    result = metric.compute(
        predictions=list(decoded_preds),
        references=[[label] for label in decoded_labels]
    )
    
    return {"bleu": result["score"]}


def main():
    parser = argparse.ArgumentParser(description="MarianMT学習")
    parser.add_argument("--use-opus-target", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="models/ja-ko")
    parser.add_argument("--data-dir", type=str, default="data/splits")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    
    print("=" * 50)
    print("MarianMT 学習")
    print("=" * 50)
    
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    train_config.output_dir = Path(args.output_dir)
    train_config.num_train_epochs = args.epochs
    train_config.per_device_train_batch_size = args.batch_size
    train_config.learning_rate = args.learning_rate
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    tokenizer_path = args.tokenizer if args.tokenizer else str(train_config.tokenizer_path)
    print(f"\nトークナイザーをロード: {tokenizer_path}")
    tokenizer = SPMTokenizer(tokenizer_path)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    print(f"\nデータをロード...")
    data_dir = Path(args.data_dir)
    teacher_dir = None
    dataset = load_data(data_dir, teacher_dir, use_opus_target=True)
    
    print(f"\n前処理中...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, model_config.max_length),
        batched=True,
        remove_columns=['ja', 'ko'],
        desc="Tokenizing"
    )
    
    print(f"\nモデルを作成...")
    model = create_model(model_config, tokenizer.vocab_size)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(train_config.output_dir),
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        lr_scheduler_type=train_config.lr_scheduler_type,
        warmup_steps=train_config.warmup_steps,
        weight_decay=train_config.weight_decay,
        num_train_epochs=train_config.num_train_epochs,
        eval_strategy=train_config.eval_strategy,
        eval_steps=train_config.eval_steps,
        save_strategy=train_config.save_strategy,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        load_best_model_at_end=train_config.load_best_model_at_end,
        metric_for_best_model=train_config.metric_for_best_model,
        greater_is_better=train_config.greater_is_better,
        predict_with_generate=True,
        generation_max_length=train_config.generation_max_length,
        generation_num_beams=train_config.generation_num_beams,
        fp16=train_config.fp16 and device == "cuda",
        dataloader_num_workers=args.num_workers if device == "cuda" else 0,
        logging_steps=train_config.logging_steps,
        report_to=train_config.report_to,
        resume_from_checkpoint=args.resume,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_config.early_stopping_patience)],
    )
    
    print(f"\n学習開始...")
    print(f"  エポック: {train_config.num_train_epochs}")
    print(f"  バッチサイズ: {train_config.per_device_train_batch_size} x {train_config.gradient_accumulation_steps} = {train_config.per_device_train_batch_size * train_config.gradient_accumulation_steps}")
    print(f"  学習率: {train_config.learning_rate}")
    
    trainer.train(resume_from_checkpoint=args.resume)
    
    print(f"\nモデルを保存: {train_config.output_dir}")
    trainer.save_model()
    
    import shutil
    shutil.copy(tokenizer_path, train_config.output_dir / "spm.model")
    
    print(f"\nテストセットで評価...")
    results = trainer.evaluate(tokenized_dataset['test'])
    print(f"Test BLEU: {results['eval_bleu']:.2f}")
    
    print("\n完了！")


if __name__ == "__main__":
    main()
