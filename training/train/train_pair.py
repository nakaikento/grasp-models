#!/usr/bin/env python3
"""
MarianMT 双方向学習スクリプト

Knowledge Distillation対応の翻訳モデル学習

Usage:
    # 日本語 → 韓国語
    python training/train_pair.py --src-lang ja --tgt-lang ko
    
    # 韓国語 → 日本語
    python training/train_pair.py --src-lang ko --tgt-lang ja

Arguments:
    --src-lang: ソース言語（ja/ko）
    --tgt-lang: ターゲット言語（ja/ko）
    --use-teacher: 教師データを使用（デフォルト: True）
    --generate-teacher: 教師データを自動生成
    --epochs: エポック数（デフォルト: 10）
    --batch-size: バッチサイズ（デフォルト: 64）
    --learning-rate: 学習率（デフォルト: 3e-4）
    --resume: チェックポイントから再開

Input:
    data/splits/{train,val,test}.{src_lang}
    data/splits/{train,val,test}.{tgt_lang} (OPUS, 評価用)
    data/teacher/train_{src_lang}_{tgt_lang}.{tgt_lang} (教師翻訳)

Output:
    models/{src_lang}-{tgt_lang}/
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import argparse
import subprocess
from tqdm import tqdm

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
    TrainerCallback,
)
import sentencepiece as spm
import evaluate

# 設定をインポート
sys.path.append(str(Path(__file__).parent.parent))
from training.train.config import ModelConfig, TrainingConfig


class SPMTokenizer:
    """SentencePieceベースのトークナイザー（MarianMT互換）"""
    
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
        
        # HuggingFace互換属性
        self.padding_side = "right"
    
    def save_pretrained(self, save_directory):
        """HuggingFace互換の保存メソッド"""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        dest_path = save_dir / "spm.model"
        with open(dest_path, "wb") as f:
            f.write(self.sp.serialized_model_proto())
    
    def __call__(self, texts, **kwargs):
        """HuggingFace互換のエンコード"""
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
            
            # EOS追加
            ids = ids + [self.eos_token_id]
            
            # Truncation
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            
            mask = [1] * len(ids)
            input_ids.append(ids)
            attention_mask.append(mask)
        
        # Padding
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
        """バッチデコード"""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]
    
    def pad(self, features, padding=True, max_length=None, return_tensors=None, **kwargs):
        """HuggingFace DataCollator互換のpadメソッド"""
        # input_ids, attention_mask, labelsを取得
        input_ids = [f['input_ids'] for f in features] if isinstance(features, list) else features['input_ids']
        attention_mask = [f.get('attention_mask') for f in features] if isinstance(features, list) else features.get('attention_mask')
        labels = [f.get('labels') for f in features] if isinstance(features, list) else features.get('labels')
        
        # 最大長を計算
        max_len = max(len(ids) for ids in input_ids)
        if labels and labels[0] is not None:
            max_label_len = max(len(l) for l in labels)
        
        # パディング
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


class ProgressCallback(TrainerCallback):
    """学習進捗を表示するコールバック"""
    
    def __init__(self):
        self.pbar = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        """学習開始"""
        total_steps = state.max_steps
        self.pbar = tqdm(total=total_steps, desc="Training", unit="step")
    
    def on_step_end(self, args, state, control, **kwargs):
        """ステップ終了"""
        if self.pbar:
            self.pbar.update(1)
            # 最新のログを表示
            if state.log_history:
                latest_log = state.log_history[-1]
                postfix = {}
                if 'loss' in latest_log:
                    postfix['loss'] = f"{latest_log['loss']:.4f}"
                if 'eval_bleu' in latest_log:
                    postfix['BLEU'] = f"{latest_log['eval_bleu']:.2f}"
                self.pbar.set_postfix(postfix)
    
    def on_train_end(self, args, state, control, **kwargs):
        """学習終了"""
        if self.pbar:
            self.pbar.close()


def generate_teacher_data(src_lang: str, tgt_lang: str, data_dir: Path, teacher_dir: Path):
    """教師データを生成"""
    print(f"\n{'='*50}")
    print(f"教師データ生成: {src_lang} → {tgt_lang}")
    print(f"{'='*50}")
    
    teacher_dir.mkdir(parents=True, exist_ok=True)
    
    src_file = data_dir / f"train.{src_lang}"
    output_file = teacher_dir / f"train_{src_lang}_{tgt_lang}.{tgt_lang}"
    
    cmd = [
        "python3", "training/generate_teacher_data.py",
        "--src-lang", src_lang,
        "--tgt-lang", tgt_lang,
        "--src-file", str(src_file),
        "--output-file", str(output_file)
    ]
    
    print(f"実行: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode == 0:
        print(f"✅ 教師データ生成完了: {output_file}")
    else:
        raise RuntimeError(f"教師データ生成に失敗しました（exit code: {result.returncode}）")


def load_data(
    src_lang: str,
    tgt_lang: str,
    data_dir: Path,
    teacher_dir: Optional[Path],
    use_teacher: bool = True
) -> DatasetDict:
    """データをロード"""
    
    # ソース言語
    with open(data_dir / f"train.{src_lang}", 'r', encoding='utf-8') as f:
        train_src = [line.strip() for line in f]
    with open(data_dir / f"val.{src_lang}", 'r', encoding='utf-8') as f:
        val_src = [line.strip() for line in f]
    with open(data_dir / f"test.{src_lang}", 'r', encoding='utf-8') as f:
        test_src = [line.strip() for line in f]
    
    # ターゲット言語
    if use_teacher and teacher_dir:
        print(f"ターゲット: 教師翻訳（{teacher_dir}/train_{src_lang}_{tgt_lang}.{tgt_lang}）")
        tgt_train_path = teacher_dir / f"train_{src_lang}_{tgt_lang}.{tgt_lang}"
        if not tgt_train_path.exists():
            raise FileNotFoundError(
                f"教師データが見つかりません: {tgt_train_path}\n"
                f"--generate-teacher を使って生成してください"
            )
    else:
        print(f"ターゲット: OPUS（{data_dir}/train.{tgt_lang}）")
        tgt_train_path = data_dir / f"train.{tgt_lang}"
    
    with open(tgt_train_path, 'r', encoding='utf-8') as f:
        train_tgt = [line.strip() for line in f]
    
    # val/testは常にOPUS（評価用）
    with open(data_dir / f"val.{tgt_lang}", 'r', encoding='utf-8') as f:
        val_tgt = [line.strip() for line in f]
    with open(data_dir / f"test.{tgt_lang}", 'r', encoding='utf-8') as f:
        test_tgt = [line.strip() for line in f]
    
    # Dataset作成
    dataset = DatasetDict({
        'train': Dataset.from_dict({src_lang: train_src, tgt_lang: train_tgt}),
        'validation': Dataset.from_dict({src_lang: val_src, tgt_lang: val_tgt}),
        'test': Dataset.from_dict({src_lang: test_src, tgt_lang: test_tgt}),
    })
    
    print(f"Train: {len(dataset['train']):,}")
    print(f"Val:   {len(dataset['validation']):,}")
    print(f"Test:  {len(dataset['test']):,}")
    
    return dataset


def create_model(config: ModelConfig, vocab_size: int) -> MarianMTModel:
    """MarianMTモデルを作成"""
    
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
    
    # パラメータ数を表示
    num_params = sum(p.numel() for p in model.parameters())
    print(f"モデルパラメータ数: {num_params:,} ({num_params/1e6:.1f}M)")
    
    return model


def preprocess_function(examples, src_lang, tgt_lang, tokenizer, max_length=128):
    """データを前処理"""
    inputs = tokenizer(
        examples[src_lang],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    
    targets = tokenizer(
        examples[tgt_lang],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    
    inputs['labels'] = targets['input_ids']
    return inputs


def compute_metrics(eval_preds, tokenizer):
    """評価メトリクス計算"""
    metric = evaluate.load("sacrebleu")
    
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
        return {"bleu": 0.0}
    
    decoded_preds, decoded_labels = zip(*filtered)
    
    # BLEU計算
    result = metric.compute(
        predictions=list(decoded_preds),
        references=[[label] for label in decoded_labels]
    )
    
    return {"bleu": result["score"]}


def main():
    parser = argparse.ArgumentParser(description="MarianMT双方向学習")
    parser.add_argument("--src-lang", type=str, required=True, choices=["ja", "ko"],
                        help="ソース言語")
    parser.add_argument("--tgt-lang", type=str, required=True, choices=["ja", "ko"],
                        help="ターゲット言語")
    parser.add_argument("--use-teacher", action="store_true", default=True,
                        help="教師データを使用（デフォルト: True）")
    parser.add_argument("--no-teacher", dest="use_teacher", action="store_false",
                        help="OPUS生データで学習（教師データなし）")
    parser.add_argument("--generate-teacher", action="store_true",
                        help="教師データを自動生成してから学習")
    parser.add_argument("--resume", type=str, default=None,
                        help="チェックポイントから再開")
    parser.add_argument("--data-dir", type=str, default="data/splits",
                        help="データディレクトリ")
    parser.add_argument("--teacher-dir", type=str, default="data/teacher",
                        help="教師データディレクトリ")
    parser.add_argument("--tokenizer", type=str, default="data/tokenized/spm.model",
                        help="トークナイザーパス（spm.model）")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoaderのワーカー数")
    args = parser.parse_args()
    
    # 言語ペアチェック
    if args.src_lang == args.tgt_lang:
        raise ValueError("ソース言語とターゲット言語は異なる必要があります")
    
    print("=" * 50)
    print(f"MarianMT 学習: {args.src_lang} → {args.tgt_lang}")
    print("=" * 50)
    
    # パス設定
    data_dir = Path(args.data_dir)
    teacher_dir = Path(args.teacher_dir)
    output_dir = Path(f"models/{args.src_lang}-{args.tgt_lang}")
    
    # 教師データ生成
    if args.generate_teacher:
        generate_teacher_data(args.src_lang, args.tgt_lang, data_dir, teacher_dir)
    
    # 設定
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # 引数で上書き
    train_config.output_dir = output_dir
    train_config.num_train_epochs = args.epochs
    train_config.per_device_train_batch_size = args.batch_size
    train_config.learning_rate = args.learning_rate
    
    # デバイス確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nデバイス: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # トークナイザー
    print(f"\nトークナイザーをロード: {args.tokenizer}")
    tokenizer = SPMTokenizer(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # データ
    print(f"\nデータをロード...")
    dataset = load_data(
        args.src_lang,
        args.tgt_lang,
        data_dir,
        teacher_dir if args.use_teacher else None,
        args.use_teacher
    )
    
    # 前処理
    print(f"\n前処理中...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, args.src_lang, args.tgt_lang, tokenizer, model_config.max_length),
        batched=True,
        remove_columns=[args.src_lang, args.tgt_lang],
        desc="Tokenizing"
    )
    
    # モデル
    print(f"\nモデルを作成...")
    model = create_model(model_config, tokenizer.vocab_size)
    
    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )
    
    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(train_config.output_dir),
        
        # バッチ
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        
        # 学習率
        learning_rate=train_config.learning_rate,
        lr_scheduler_type=train_config.lr_scheduler_type,
        warmup_steps=train_config.warmup_steps,
        weight_decay=train_config.weight_decay,
        
        # エポック
        num_train_epochs=train_config.num_train_epochs,
        
        # 評価・保存
        eval_strategy=train_config.eval_strategy,
        eval_steps=train_config.eval_steps,
        save_strategy=train_config.save_strategy,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        load_best_model_at_end=train_config.load_best_model_at_end,
        metric_for_best_model=train_config.metric_for_best_model,
        greater_is_better=train_config.greater_is_better,
        
        # 生成（評価用）
        predict_with_generate=True,
        generation_max_length=train_config.generation_max_length,
        generation_num_beams=train_config.generation_num_beams,
        
        # その他
        fp16=train_config.fp16 and device == "cuda",
        dataloader_num_workers=args.num_workers if device == "cuda" else 0,
        logging_steps=train_config.logging_steps,
        report_to=["none"],  # wandb無効化
        
        # 再開
        resume_from_checkpoint=args.resume,
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
            EarlyStoppingCallback(early_stopping_patience=train_config.early_stopping_patience),
            ProgressCallback(),
        ],
    )
    
    # 学習
    print(f"\n学習開始...")
    print(f"  言語ペア: {args.src_lang} → {args.tgt_lang}")
    print(f"  エポック: {train_config.num_train_epochs}")
    print(f"  バッチサイズ: {train_config.per_device_train_batch_size} x {train_config.gradient_accumulation_steps} = {train_config.per_device_train_batch_size * train_config.gradient_accumulation_steps}")
    print(f"  学習率: {train_config.learning_rate}")
    print(f"  出力: {train_config.output_dir}")
    print()
    
    try:
        trainer.train(resume_from_checkpoint=args.resume)
    except KeyboardInterrupt:
        print("\n⚠️ 学習が中断されました")
        print(f"チェックポイント: {train_config.output_dir}")
        print(f"再開するには: --resume {train_config.output_dir}/checkpoint-XXXX")
        return
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 保存
    print(f"\nモデルを保存: {train_config.output_dir}")
    trainer.save_model()
    
    # トークナイザーも保存（推論用）
    import shutil
    shutil.copy(args.tokenizer, train_config.output_dir / "spm.model")
    print(f"トークナイザーをコピー: {args.tokenizer} → {train_config.output_dir}/spm.model")
    
    # 最終評価
    print(f"\nテストセットで評価...")
    results = trainer.evaluate(tokenized_dataset['test'])
    print(f"\n{'='*50}")
    print(f"✅ 学習完了！")
    print(f"{'='*50}")
    print(f"Test BLEU: {results['eval_bleu']:.2f}")
    print(f"出力: {train_config.output_dir}")
    
    # ONNX変換のヒント
    print(f"\n次のステップ:")
    print(f"  1. ONNX変換: python training/convert_to_onnx.py --model-dir {train_config.output_dir}")
    print(f"  2. 量子化: python training/quantize_onnx.py --model-dir {train_config.output_dir}")


if __name__ == "__main__":
    main()
