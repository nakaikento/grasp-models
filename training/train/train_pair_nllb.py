#!/usr/bin/env python3
"""
NLLB (No Language Left Behind) 学習スクリプト

Meta の NLLB-200 モデルを韓日翻訳用にファインチューニング

Usage:
    # 韓国語 → 日本語（推奨）
    python training/train/train_pair_nllb.py --src-lang ko --tgt-lang ja
    
    # 日本語 → 韓国語
    python training/train/train_pair_nllb.py --src-lang ja --tgt-lang ko

    # 小規模テスト
    python training/train/train_pair_nllb.py --src-lang ko --tgt-lang ja --limit 1000 --epochs 1

Models:
    - facebook/nllb-200-distilled-600M  (推奨、モバイル向け)
    - facebook/nllb-200-distilled-1.3B  (高品質)

Input:
    data/raw/OpenSubtitles.ja-ko.ko (ソース)
    data/teacher/OpenSubtitles-Qwen72B.ja-ko.ja (教師翻訳)

Output:
    models/nllb-{src_lang}-{tgt_lang}/
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List
import argparse
from tqdm import tqdm

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
import evaluate

# NLLB言語コード
LANG_CODES = {
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
}


class ProgressCallback(TrainerCallback):
    """学習進捗を表示するコールバック"""
    
    def __init__(self):
        self.pbar = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = state.max_steps
        self.pbar = tqdm(total=total_steps, desc="Training", unit="step")
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.update(1)
            if state.log_history:
                latest_log = state.log_history[-1]
                postfix = {}
                if 'loss' in latest_log:
                    postfix['loss'] = f"{latest_log['loss']:.4f}"
                if 'eval_bleu' in latest_log:
                    postfix['BLEU'] = f"{latest_log['eval_bleu']:.2f}"
                if 'eval_chrf' in latest_log:
                    postfix['chrF'] = f"{latest_log['eval_chrf']:.2f}"
                self.pbar.set_postfix(postfix)
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()


def load_parallel_data(
    src_file: Path,
    tgt_file: Path,
    limit: Optional[int] = None,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
) -> DatasetDict:
    """並列データを読み込んでtrain/val/testに分割"""
    
    print(f"ソース: {src_file}")
    print(f"ターゲット: {tgt_file}")
    
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f if line.strip()]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f if line.strip()]
    
    # 行数を揃える
    min_len = min(len(src_lines), len(tgt_lines))
    src_lines = src_lines[:min_len]
    tgt_lines = tgt_lines[:min_len]
    
    if limit:
        src_lines = src_lines[:limit]
        tgt_lines = tgt_lines[:limit]
    
    total = len(src_lines)
    print(f"総データ数: {total:,}")
    
    # 分割
    val_size = int(total * val_ratio)
    test_size = int(total * test_ratio)
    train_size = total - val_size - test_size
    
    # シャッフル（再現性のためseed固定）
    np.random.seed(42)
    indices = np.random.permutation(total)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    def create_split(idxs):
        return Dataset.from_dict({
            'src': [src_lines[i] for i in idxs],
            'tgt': [tgt_lines[i] for i in idxs],
        })
    
    dataset = DatasetDict({
        'train': create_split(train_indices),
        'validation': create_split(val_indices),
        'test': create_split(test_indices),
    })
    
    print(f"Train: {len(dataset['train']):,}")
    print(f"Val:   {len(dataset['validation']):,}")
    print(f"Test:  {len(dataset['test']):,}")
    
    return dataset


def preprocess_function(
    examples,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    max_length: int = 128,
):
    """データを前処理"""
    # NLLBは言語コードをsrc_langとして設定
    tokenizer.src_lang = LANG_CODES[src_lang]
    
    inputs = tokenizer(
        examples['src'],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    
    # ターゲット言語を設定してラベル作成
    tokenizer.tgt_lang = LANG_CODES[tgt_lang]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['tgt'],
            max_length=max_length,
            truncation=True,
            padding=False,
        )
    
    inputs['labels'] = labels['input_ids']
    return inputs


def compute_metrics(eval_preds, tokenizer):
    """評価メトリクス計算（BLEU + chrF++）"""
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")
    
    preds, labels = eval_preds
    
    # -100をpad_token_idに置換
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # デコード
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 空文字列を除外
    filtered = [(p.strip(), l.strip()) for p, l in zip(decoded_preds, decoded_labels) 
                if p.strip() and l.strip()]
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
        word_order=2,  # chrF++
    )
    
    return {
        "bleu": bleu_result["score"],
        "chrf": chrf_result["score"],
    }


def main():
    parser = argparse.ArgumentParser(description="NLLB ファインチューニング")
    parser.add_argument("--src-lang", type=str, required=True, choices=["ja", "ko"],
                        help="ソース言語")
    parser.add_argument("--tgt-lang", type=str, required=True, choices=["ja", "ko"],
                        help="ターゲット言語")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-distilled-600M",
                        help="ベースモデル")
    parser.add_argument("--src-file", type=str, default=None,
                        help="ソースファイル（デフォルト: data/raw/OpenSubtitles.ja-ko.{src_lang}）")
    parser.add_argument("--tgt-file", type=str, default=None,
                        help="ターゲットファイル（デフォルト: data/teacher/OpenSubtitles-Qwen72B.ja-ko.{tgt_lang}）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="出力ディレクトリ")
    parser.add_argument("--limit", type=int, default=None,
                        help="データ数制限（テスト用）")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="チェックポイントから再開")
    args = parser.parse_args()
    
    # 言語ペアチェック
    if args.src_lang == args.tgt_lang:
        raise ValueError("ソース言語とターゲット言語は異なる必要があります")
    
    print("=" * 60)
    print(f"NLLB ファインチューニング: {args.src_lang} → {args.tgt_lang}")
    print("=" * 60)
    
    # パス設定
    if args.src_file:
        src_file = Path(args.src_file)
    else:
        src_file = Path(f"data/raw/OpenSubtitles.ja-ko.{args.src_lang}")
    
    if args.tgt_file:
        tgt_file = Path(args.tgt_file)
    else:
        tgt_file = Path(f"data/teacher/OpenSubtitles-Qwen72B.ja-ko.{args.tgt_lang}")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"models/nllb-{args.src_lang}-{args.tgt_lang}")
    
    print(f"\nモデル: {args.model}")
    print(f"ソース: {src_file}")
    print(f"ターゲット: {tgt_file}")
    print(f"出力: {output_dir}")
    
    # デバイス確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nデバイス: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # トークナイザー
    print(f"\nトークナイザーをロード: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # モデル
    print(f"モデルをロード: {args.model}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # データ
    print(f"\nデータをロード...")
    dataset = load_parallel_data(
        src_file,
        tgt_file,
        limit=args.limit,
    )
    
    # 前処理
    print(f"\n前処理中...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(
            x, tokenizer, args.src_lang, args.tgt_lang, args.max_length
        ),
        batched=True,
        remove_columns=['src', 'tgt'],
        desc="Tokenizing",
    )
    
    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )
    
    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        
        # バッチ
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        
        # 学習率
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        
        # エポック
        num_train_epochs=args.epochs,
        
        # 評価・保存
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        
        # 生成（評価用）
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=4,
        
        # その他
        fp16=device == "cuda",
        dataloader_num_workers=args.num_workers if device == "cuda" else 0,
        logging_steps=100,
        report_to=["none"],
        
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
            EarlyStoppingCallback(early_stopping_patience=3),
            ProgressCallback(),
        ],
    )
    
    # 学習
    print(f"\n学習開始...")
    print(f"  言語ペア: {args.src_lang} ({LANG_CODES[args.src_lang]}) → {args.tgt_lang} ({LANG_CODES[args.tgt_lang]})")
    print(f"  エポック: {args.epochs}")
    print(f"  バッチサイズ: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"  学習率: {args.learning_rate}")
    print()
    
    try:
        trainer.train(resume_from_checkpoint=args.resume)
    except KeyboardInterrupt:
        print("\n⚠️ 学習が中断されました")
        print(f"チェックポイント: {output_dir}")
        return
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 保存
    print(f"\nモデルを保存: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # 最終評価
    print(f"\nテストセットで評価...")
    results = trainer.evaluate(tokenized_dataset['test'])
    
    print(f"\n{'='*60}")
    print(f"✅ 学習完了！")
    print(f"{'='*60}")
    print(f"Test BLEU:  {results['eval_bleu']:.2f}")
    print(f"Test chrF++: {results['eval_chrf']:.2f}")
    print(f"出力: {output_dir}")
    
    # 次のステップ
    print(f"\n次のステップ:")
    print(f"  1. ONNX変換: python scripts/convert_to_onnx.py --model-dir {output_dir}")
    print(f"  2. 量子化:   python scripts/quantize_onnx.py --model-dir {output_dir}")


if __name__ == "__main__":
    main()
