import torch
import re
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --- 設定 ---
MODEL_NAME = "facebook/nllb-200-3.3b"
SOURCE_FILE = "data/raw/OpenSubtitles.ja-ko.ja"
OUTPUT_FILE = "data/teacher/train.ko"
SAMPLE_INTERVAL = 5000 

JP_PATTERN = re.compile(r'[ぁ-んァ-ヶ一-龠]')
def contains_japanese(text):
    return bool(JP_PATTERN.search(text))

def clean_input(text):
    """モデルが混乱しやすい記号を一時的に除去"""
    t = text.replace('・・', '')
    t = t.replace('・', '')
    t = re.sub(r'^[-ー－]\s*', '', t) # 文頭のハイフン等を除去
    return t.strip()

def main():
    parser = argparse.ArgumentParser()
    # num_beams=3 にするため、安全を見てバッチサイズを少し調整（32-40推奨）
    parser.add_argument("--batch_size", type=int, default=40)
    args = parser.parse_args()

    print(f"🚀 超高品質モード(16-bit + Beam3 + Cleaning)でロード中...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tgt_lang_id = tokenizer.convert_tokens_to_ids("kor_Hang")

    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        ja_lines = [line.strip() for line in f]
    
    start_idx = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            start_idx = sum(1 for _ in f)
        print(f"🔄 {start_idx}行目から再開します...")

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for i in tqdm(range(start_idx, len(ja_lines), args.batch_size), initial=start_idx//args.batch_size):
            batch_raw = ja_lines[i : i + args.batch_size]
            
            # 【修正】入力文をクリーニング
            batch_cleaned = [clean_input(line) if len(line) > 1 else line for line in batch_raw]
            
            # 空行対策（クリーニングで空になった場合用）
            batch_cleaned = [c if c else "。" for c in batch_cleaned]

            inputs = tokenizer(batch_cleaned, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=128,
                    num_beams=3,           # 探索の幅をさらに強化
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            final_results = []
            for res in results:
                clean_res = res.replace("\n", " ").strip()
                # 日本語が残っているか、極端に短い（失敗）場合は弾く
                if contains_japanese(clean_res) or len(clean_res) < 1:
                    final_results.append("FAILED_TRANSLATION_CLEANED")
                else:
                    final_results.append(clean_res)

            if i % SAMPLE_INTERVAL < args.batch_size:
                print(f"\n--- [進捗チェック: {i}行目] ---")
                print(f"原: {batch_raw[0]}")
                print(f"洗: {batch_cleaned[0]}")
                print(f"韓: {final_results[0]}")
                print("-" * 40)

            for res in final_results:
                f.write(res + "\n")
            f.flush()

            del inputs, outputs
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()