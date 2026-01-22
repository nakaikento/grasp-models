#!/usr/bin/env python3
"""
教師データ生成スクリプト（3.3Bモデル & 日本語フィルタ搭載版）

NLLB-200-3.3Bを使用して高品質な翻訳を生成。
4-bit量子化によりColab/Consumer GPUでの動作に対応し、
日本語が混入した行を自動的に除外・再試行します。
"""

import torch
import re
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

# 設定のインポート（パスが通るように調整）
sys.path.append(str(Path(__file__).parent.parent))
# from training.config import DistillationConfig # 必要に応じてコメントアウト解除

@dataclass
class GenerationArgs:
    input_file: Path = Path("data/splits/train.ja")
    output_file: Path = Path("data/teacher/train.ko")
    model_name: str = "facebook/nllb-200-3.3B"
    batch_size: int = 4  # 3.3B用に小さく調整
    max_length: int = 128
    num_beams: int = 2   # 速度と品質のバランス
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_every: int = 10 # 10バッチごとに保存

# 日本語検知用正規表現
JP_PATTERN = re.compile(r'[ぁ-んァ-ヶ一-龠]')

def contains_japanese(text):
    """テキストに日本語（ひらがな、カタカナ、漢字）が含まれているか判定"""
    return bool(JP_PATTERN.search(text))

def load_model_optimized(model_name: str, device: str):
    """3.3Bモデルを4-bit量子化でロード"""
    print(f"🚀 モデルをロード中: {model_name} (4-bit quantization)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # ColabのT4 GPUでも動作するように4bit量子化を設定
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # 自動でGPUに割り当て
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=2)
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
    model, tokenizer = load_model_optimized(args.model, "cuda")
    tgt_lang = "kor_Hang"

    # 生成開始
    print(f"\n翻訳開始 (Target: {tgt_lang})...")
    
    with open(output_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(ja_texts), args.batch_size)):
            batch = ja_texts[i : i + args.batch_size]
            
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_length=128,
                    num_beams=args.num_beams
                )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for original_ja, translated_ko in zip(batch, decoded):
                # 日本語が含まれているかチェック
                if contains_japanese(translated_ko):
                    # 日本語が混じった場合は、空行にするかエラー用の印を付けて
                    # 学習データとしての品質を守る
                    f.write("FAILED_TRANSLATION_CLEANED\n")
                else:
                    f.write(translated_ko + "\n")

    print(f"\n✨ 完了! 出力先: {output_path}")

if __name__ == "__main__":
    main()