#!/usr/bin/env python3
"""
MarianMT KO→JA ファインチューニングスクリプト

Qwen教師データを使用してMarianMTをファインチューニング
- 英語混入・アラビア文字などの低品質データをフィルタリング
- Knowledge Distillation方式

使用方法:
    python train_marian_ko_ja.py \
        --src_file ../data/raw/OpenSubtitles.ja-ko.ko \
        --tgt_file ../data/teacher/qwen_train.ja \
        --output_dir ../models/marian-ko-ja-finetuned \
        --epochs 3
"""

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from datasets import Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """フィルタリング統計"""
    total: int = 0
    pure_english: int = 0
    english_mixed: int = 0
    arabic_chars: int = 0
    empty_lines: int = 0
    too_long: int = 0
    passed: int = 0


class DataFilter:
    """低品質データをフィルタリング"""
    
    # 純英語行のパターン
    PURE_ENGLISH_PATTERN = re.compile(r'^[A-Za-z0-9\s.,!?\-\'"()]+$')
    
    # 5文字以上の連続英語（固有名詞以外）
    LONG_ENGLISH_PATTERN = re.compile(r'[A-Za-z]{5,}')
    
    # アラビア文字
    ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF؟]')
    
    # 許可する英語（固有名詞、一般的な略語）
    ALLOWED_ENGLISH = {
        'OK', 'TV', 'CD', 'DVD', 'PC', 'FBI', 'CIA', 'DNA', 'GPS', 'VIP',
        'iPhone', 'iPad', 'Google', 'Facebook', 'Twitter', 'YouTube',
        'Mr', 'Mrs', 'Dr', 'Jr', 'Sr', 'vs', 'etc', 'No', 'OK',
        'LOVE', 'HAPPY', 'NEW', 'GOOD', 'BAD', 'THE', 'AND', 'FOR',
    }
    
    def __init__(self, max_length: int = 256):
        self.max_length = max_length
        self.stats = FilterStats()
    
    def is_valid(self, src: str, tgt: str) -> bool:
        """ペアが有効かどうか判定"""
        self.stats.total += 1
        
        # 空行チェック
        if not src.strip() or not tgt.strip():
            self.stats.empty_lines += 1
            return False
        
        # ターゲット（日本語）のチェック
        tgt = tgt.strip()
        
        # 純英語行
        if self.PURE_ENGLISH_PATTERN.match(tgt):
            self.stats.pure_english += 1
            return False
        
        # アラビア文字
        if self.ARABIC_PATTERN.search(tgt):
            self.stats.arabic_chars += 1
            return False
        
        # 長い英語が含まれている（許可リスト以外）
        english_matches = self.LONG_ENGLISH_PATTERN.findall(tgt)
        if english_matches:
            # 許可リストにない英語があればフィルタ
            non_allowed = [m for m in english_matches if m.upper() not in self.ALLOWED_ENGLISH]
            if non_allowed:
                self.stats.english_mixed += 1
                return False
        
        # 長すぎる文
        if len(src) > self.max_length or len(tgt) > self.max_length:
            self.stats.too_long += 1
            return False
        
        self.stats.passed += 1
        return True
    
    def report(self) -> str:
        """統計レポート"""
        s = self.stats
        pass_rate = (s.passed / s.total * 100) if s.total > 0 else 0
        return f"""
========================================
📊 フィルタリング結果
========================================
総行数:         {s.total:,}
----------------------------------------
純英語行:       {s.pure_english:,} ({s.pure_english/s.total*100:.2f}%)
英語混入:       {s.english_mixed:,} ({s.english_mixed/s.total*100:.2f}%)
アラビア文字:   {s.arabic_chars:,} ({s.arabic_chars/s.total*100:.2f}%)
空行:           {s.empty_lines:,} ({s.empty_lines/s.total*100:.2f}%)
長すぎる文:     {s.too_long:,} ({s.too_long/s.total*100:.2f}%)
----------------------------------------
✅ 通過:        {s.passed:,} ({pass_rate:.2f}%)
========================================
"""


def load_and_filter_data(
    src_file: str,
    tgt_file: str,
    max_length: int = 256,
    val_ratio: float = 0.01,
) -> Tuple[Dataset, Dataset, FilterStats]:
    """データを読み込み、フィルタリングして分割"""
    
    logger.info(f"📂 ソースファイル: {src_file}")
    logger.info(f"📂 ターゲットファイル: {tgt_file}")
    
    # ファイル読み込み
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = f.readlines()
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = f.readlines()
    
    assert len(src_lines) == len(tgt_lines), \
        f"行数が一致しません: src={len(src_lines)}, tgt={len(tgt_lines)}"
    
    logger.info(f"📊 総行数: {len(src_lines):,}")
    
    # フィルタリング
    data_filter = DataFilter(max_length=max_length)
    filtered_pairs = []
    
    for src, tgt in zip(src_lines, tgt_lines):
        src = src.strip()
        tgt = tgt.strip()
        if data_filter.is_valid(src, tgt):
            filtered_pairs.append({'source': src, 'target': tgt})
    
    logger.info(data_filter.report())
    
    # シャッフルして分割
    import random
    random.seed(42)
    random.shuffle(filtered_pairs)
    
    val_size = int(len(filtered_pairs) * val_ratio)
    train_data = filtered_pairs[val_size:]
    val_data = filtered_pairs[:val_size]
    
    logger.info(f"📚 学習データ: {len(train_data):,}")
    logger.info(f"📚 検証データ: {len(val_data):,}")
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset, data_filter.stats


def preprocess_function(examples, tokenizer, max_length=128):
    """トークナイズ処理"""
    inputs = examples['source']
    targets = examples['target']
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding='max_length',
    )
    
    labels = tokenizer(
        text_target=targets,
        max_length=max_length,
        truncation=True,
        padding='max_length',
    )
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def compute_metrics(eval_preds, tokenizer, metric_bleu, metric_chrf):
    """評価メトリクス計算"""
    preds, labels = eval_preds
    
    # -100をpad_token_idに置換
    labels = [[l if l != -100 else tokenizer.pad_token_id for l in label] for label in labels]
    
    # デコード
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # BLEU
    bleu_result = metric_bleu.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels]
    )
    
    # chrF++
    chrf_result = metric_chrf.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels],
        word_order=2  # chrF++
    )
    
    return {
        'bleu': bleu_result['bleu'] * 100,
        'chrf': chrf_result['score'],
    }


def main():
    parser = argparse.ArgumentParser(description='MarianMT KO→JA ファインチューニング')
    parser.add_argument('--src_file', type=str, required=True,
                        help='ソース（韓国語）ファイル')
    parser.add_argument('--tgt_file', type=str, required=True,
                        help='ターゲット（日本語）ファイル')
    parser.add_argument('--output_dir', type=str, default='../models/marian-ko-ja-finetuned',
                        help='出力ディレクトリ')
    parser.add_argument('--base_model', type=str, default='Helsinki-NLP/opus-mt-ko-ja',
                        help='ベースモデル')
    parser.add_argument('--epochs', type=int, default=3,
                        help='エポック数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='バッチサイズ')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学習率')
    parser.add_argument('--max_length', type=int, default=128,
                        help='最大トークン長')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='ウォームアップ比率')
    parser.add_argument('--val_ratio', type=float, default=0.01,
                        help='検証データ比率')
    parser.add_argument('--fp16', action='store_true',
                        help='混合精度学習')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='勾配蓄積ステップ')
    
    args = parser.parse_args()
    
    # GPU確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"🖥️ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # データ読み込み・フィルタリング
    train_dataset, val_dataset, filter_stats = load_and_filter_data(
        args.src_file,
        args.tgt_file,
        max_length=args.max_length * 4,  # 文字数ベースでざっくり
        val_ratio=args.val_ratio,
    )
    
    # モデル・トークナイザー読み込み
    logger.info(f"📦 ベースモデル: {args.base_model}")
    tokenizer = MarianTokenizer.from_pretrained(args.base_model)
    model = MarianMTModel.from_pretrained(args.base_model)
    
    # トークナイズ
    logger.info("🔄 トークナイズ中...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=['source', 'target'],
        desc="Tokenizing train",
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=['source', 'target'],
        desc="Tokenizing val",
    )
    
    # 評価メトリクス
    metric_bleu = evaluate.load('bleu')
    metric_chrf = evaluate.load('chrf')
    
    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # 学習設定
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=1000,
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='chrf',
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        report_to='none',  # wandb無効
        dataloader_num_workers=4,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, metric_bleu, metric_chrf),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # 学習開始
    logger.info("🚀 学習開始!")
    logger.info(f"   エポック数: {args.epochs}")
    logger.info(f"   バッチサイズ: {args.batch_size}")
    logger.info(f"   学習率: {args.learning_rate}")
    logger.info(f"   総ステップ数: {len(train_dataset) // args.batch_size * args.epochs:,}")
    
    trainer.train()
    
    # 保存
    logger.info(f"💾 モデル保存: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 最終評価
    logger.info("📊 最終評価...")
    eval_results = trainer.evaluate()
    logger.info(f"   BLEU: {eval_results.get('eval_bleu', 0):.2f}")
    logger.info(f"   chrF++: {eval_results.get('eval_chrf', 0):.2f}")
    
    logger.info("✅ 学習完了!")


if __name__ == '__main__':
    main()
