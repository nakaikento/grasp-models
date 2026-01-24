#!/usr/bin/env python3
import torch
import re
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

# --- 定数設定 ---
MODEL_NAME = "facebook/nllb-200-3.3b"
SOURCE_FILE = "data/raw/OpenSubtitles.ja-ko.ja"
OUTPUT_FILE = "data/teacher/train.ko"
SAMPLE_INTERVAL = 10000  # 1万行ごとにサンプルを表示

# 日本語検知用
JP_PATTERN = re.compile(r'[ぁ-んァ-ヶ一-龠]')
def contains_japanese(text):
    return bool(JP_PATTERN.search(text))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    print(f"🚀 モデルをロード中: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 4-bit量子化設定（VRAM節約と高速化）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    tgt_lang_id = tokenizer.lang_code_to_id["kor_Hang"]

    # 原文データの読み込み
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ エラー: {SOURCE_FILE} が見つかりません。")
        return

    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        ja_lines = [line.strip() for line in f]
    
    total_lines = len(ja_lines)
    print(f"📖 総行数: {total_lines}")

    # 再開ポイントの確認（既存ファイルの行数をカウント）
    start_idx = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            start_idx = sum(1 for _ in f)
        print(f"🔄 {start_idx}行目から再開します（既存データと同期）")
    else:
        # 新規作成時にディレクトリがない場合は作成
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 翻訳メインループ
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        # tqdmで進捗を表示
        for i in tqdm(range(start_idx, total_lines, args.batch_size), initial=start_idx//args.batch_size):
            batch = ja_lines[i : i + args.batch_size]
            
            # 1. 翻訳実行
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    forced_bos_token_id=tgt_lang_id, 
                    max_length=128
                )
            results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # 2. 日本語が混じった場合の簡易リトライ（オプション）
            # ※今回は「行の同期」を最優先するため、失敗しても必ず1行書き出します
            final_results = []
            for idx, res in enumerate(results):
                clean_res = res.replace("\n", " ").strip()
                if contains_japanese(clean_res) or not clean_res:
                    final_results.append("FAILED_TRANSLATION_CLEANED")
                else:
                    final_results.append(clean_res)

            # 3. 1万行ごとのサンプル表示（安心機能）
            if i % SAMPLE_INTERVAL < args.batch_size:
                print(f"\n\n--- [進捗チェック: {i}行目] ---")
                print(f"日: {batch[0]}")
                print(f"韓: {final_results[0]}")
                print("-" * 40)

            # 4. ファイルへ書き出し
            for res in final_results:
                f.write(res + "\n")
            
            # バッファを強制フラッシュ（スリープ対策）
            f.flush()

    print(f"✨ 完了しました！出力先: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()