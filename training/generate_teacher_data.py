#!/usr/bin/env python3
"""
教師データ生成スクリプト（3.3Bモデル + リトライロジック搭載版）

NLLB-200-3.3Bを使用して高品質な翻訳を生成。
4-bit量子化によりL4/T4 GPUに対応し、日本語混入時に設定を変えて再試行します。
"""

import torch
import re
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

# 日本語検知用正規表現（ひらがな・カタカナ・漢字）
JP_PATTERN = re.compile(r'[ぁ-んァ-ヶ一-龠]')

def contains_japanese(text):
    """テキストに日本語が含まれているか判定"""
    return bool(JP_PATTERN.search(text))

def load_model_optimized(model_name: str):
    """3.3Bモデルを4-bit量子化でロード"""
    print(f"🚀 モデルをロード中: {model_name} (4-bit quantization)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    return model, tokenizer

def count_existing_lines(file_path: Path) -> int:
    if not file_path.exists():
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="途中から再開")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-3.3B")
    parser.add_argument("--batch_size", type=int, default=32) # L4向けに32を設定
    parser.add_argument("--num_beams", type=int, default=1)  # 速度重視
    parser.add_argument("--input", type=str, default="data/splits/train.ja")
    parser.add_argument("--output", type=str, default="data/teacher/train.ko")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # データの読み込み
    if not input_path.exists():
        print(f"❌ 入力ファイルが見つかりません: {input_path}")
        return
    
    with open(input_path, "r", encoding="utf-8") as f:
        ja_texts = [line.strip() for line in f if line.strip()]

    # 再開処理
    start_idx = 0
    if args.resume:
        start_idx = count_existing_lines(output_path)
        if start_idx > 0:
            print(f"🔄 再開モード: {start_idx:,}行目から開始します")
            ja_texts = ja_texts[start_idx:]
    else:
        if output_path.exists():
            print(f"⚠️ 警告: {output_path} は既に存在します。上書きします。")
            output_path.unlink()

    if not ja_texts:
        print("✅ 処理するデータがありません。")
        return

    # モデルロード
    model, tokenizer = load_model_optimized(args.model)
    tgt_lang = "kor_Hang"
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    print(f"\n翻訳開始 (Target: {tgt_lang}, Batch Size: {args.batch_size})...")
    
    

    with open(output_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(ja_texts), args.batch_size)):
            batch = ja_texts[i : i + args.batch_size]
            
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
            
            # 1回目の生成 (通常の推論)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=128,
                    num_beams=args.num_beams,
                    do_sample=False
                )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            final_results = []
            for idx, (original_ja, translated_ko) in enumerate(zip(batch, decoded)):
                # 日本語が含まれているかチェック
                if contains_japanese(translated_ko):
                    # --- リトライ試行 (Samplingモードで1回だけ挑戦) ---
                    retry_input = tokenizer([original_ja], return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        retry_output = model.generate(
                            **retry_input,
                            forced_bos_token_id=tgt_lang_id,
                            max_length=128,
                            do_sample=True,      # ランダム性を導入
                            temperature=0.7,     # 柔軟な生成
                            top_p=0.9,
                            num_beams=1
                        )
                    retry_text = tokenizer.decode(retry_output[0], skip_special_tokens=True)
                    
                    if contains_japanese(retry_text):
                        final_results.append("FAILED_TRANSLATION_CLEANED")
                    else:
                        final_results.append(retry_text)
                else:
                    final_results.append(translated_ko)

            # ファイルに書き出し
            for res in final_results:
                f.write(res + "\n")

    print(f"\n✨ 完了! 出力先: {output_path}")

if __name__ == "__main__":
    main()