import os
import torch
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import evaluate
import numpy as np

# --- 設定項目 ---
MODEL_NAME = "Helsinki-NLP/opus-ja-ko"  # ベースモデル
DATA_JA = "data/clean/train.ja"
DATA_KO = "data/clean/train.ko"
OUTPUT_DIR = "models/marian_ja_ko_v1"

def load_and_split_data():
    """データをロードして学習用と検証用に分割"""
    print(f"📂 データをロード中: {DATA_JA} / {DATA_KO}")
    with open(DATA_JA, 'r', encoding='utf-8') as f:
        ja_lines = [line.strip() for line in f]
    with open(DATA_KO, 'r', encoding='utf-8') as f:
        ko_lines = [line.strip() for line in f]
    
    # Dataset作成
    full_dataset = Dataset.from_dict({
        "ja": ja_lines,
        "ko": ko_lines
    })
    
    # 5%を検証用に分割 (約3万行)
    return full_dataset.train_test_split(test_size=0.05, seed=42)

def compute_metrics(eval_preds, tokenizer, metric):
    """BLEUスコアの計算"""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # -100 (ラベル無視用) を pad に戻す
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # SacreBLEU用に整形
    decoded_labels = [[line] for line in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128) # 4090なら128で安定
    parser.add_argument("--epochs", type=int, default=5)       # 60万行なら3~5エポック
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    # 1. トークナイザー & メトリクス
    print(f"📝 トークナイザーをロード中: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    metric = evaluate.load("sacrebleu")

    # 2. データの準備
    dataset = load_and_split_data()
    
    def preprocess_function(examples):
        inputs = examples["ja"]
        targets = examples["ko"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        
        # ターゲットのトークナイズ
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("⚡ トークナイズ実行中 (マルチプロセス)...")
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True, 
        num_proc=8, # CPUコア数に合わせて調整
        remove_columns=dataset["train"].column_names
    )

    # 3. モデルのロード
    print(f"🤖 モデルを初期化中: {MODEL_NAME}")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, config=config)

    # 4. 学習引数の設定
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=2000,               # 2000ステップごとに評価
        save_strategy="steps",
        save_steps=2000,
        logging_steps=500,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        # 4090 最適化設定
        bf16=True,                     # Ampere/Ada GPUなら必須
        fp16=False,
        gradient_checkpointing=False,  # Marianは軽いのでFalseでOK
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        warmup_steps=1000,
    )

    # 5. トレーナーの構築
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, metric),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # BLEUが改善しなくなったら停止
    )

    # 6. 学習実行
    print("🚀 学習開始！")
    trainer.train()

    # 7. 保存
    print(f"💾 最終モデルを保存中: {OUTPUT_DIR}/final")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print("✅ 全ての工程が完了しました。")

if __name__ == "__main__":
    main()