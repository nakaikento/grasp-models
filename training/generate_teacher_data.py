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
SAMPLE_INTERVAL = 10000 

JP_PATTERN = re.compile(r'[ぁ-んァ-ヶ一-龠]')
def contains_japanese(text):
    return bool(JP_PATTERN.search(text))

def main():
    parser = argparse.ArgumentParser()
    # 16-bitではメモリ消費が増えるため、batch_sizeは32〜64を推奨
    parser.add_argument("--batch_size", type=int, default=48)
    args = parser.parse_args()

    print(f"🚀 高品質モード(16-bit bfloat16)でロード中: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 【修正の肝】量子化(BitsAndBytesConfig)を外し、bfloat16を指定
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,  # 4090に最適な精度
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
            batch = ja_lines[i : i + args.batch_size]
            
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")

            with torch.no_grad():
                # 【修正の肝】Beam Search (num_beams=2) を有効化
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=128,
                    num_beams=2,           # 候補を2つ探索して質の高い方を選択
                    no_repeat_ngram_size=3, # 繰り返しバグ防止
                    early_stopping=True
                )
            
            results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # 書き出し用のクリーンアップと日本語チェック
            final_results = []
            for res in results:
                clean_res = res.replace("\n", " ").strip()
                if contains_japanese(clean_res) or not clean_res:
                    final_results.append("FAILED_TRANSLATION_CLEANED")
                else:
                    final_results.append(clean_res)

            # サンプル表示
            if i % SAMPLE_INTERVAL < args.batch_size:
                print(f"\n--- [進捗チェック: {i}行目] ---")
                print(f"日: {batch[0]}")
                print(f"韓: {final_results[0]}")
                print("-" * 40)

            for res in final_results:
                f.write(res + "\n")
            f.flush()

            # メモリの明示的解放（OOM対策）
            del inputs, outputs
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()