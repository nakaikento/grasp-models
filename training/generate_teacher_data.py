#!/usr/bin/env python3
"""
教師データ生成スクリプト

M2M100 or NLLB-200を使用して、日本語から高品質な韓国語翻訳を生成

Usage:
    python training/generate_teacher_data.py

Input:  data/splits/train.ja
Output: data/teacher/train.ko (教師翻訳)
"""

import torch
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
import argparse

# 設定をインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.config import DistillationConfig


@dataclass
class GenerationArgs:
    input_file: Path = Path("data/splits/train.ja")
    output_file: Path = Path("data/teacher/train.ko")
    model_name: str = "facebook/nllb-200-1.3B"
    batch_size: int = 16
    max_length: int = 128
    num_beams: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume_from: int = 0  # 途中から再開する場合


def load_model(model_name: str, device: str):
    """教師モデルをロード"""
    print(f"モデルをロード中: {model_name}")
    
    # accelerateを使わずにシンプルにロード
    import os
    os.environ["ACCELERATE_USE_SAFETENSORS"] = "true"
    
    if "nllb" in model_name.lower():
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # シンプルにロード（device_mapなし）
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            use_safetensors=True,
            low_cpu_mem_usage=False,
        )
        if device == "cuda":
            model = model.half()  # GPU時はfp16
        model = model.to(device)
        
        src_lang = "jpn_Jpan"
        tgt_lang = "kor_Hang"
        
    elif "m2m100" in model_name.lower():
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        
        model = M2M100ForConditionalGeneration.from_pretrained(
            model_name,
            use_safetensors=True,
            low_cpu_mem_usage=False,
        )
        if device == "cuda":
            model = model.half()
        model = model.to(device)
        
        tokenizer.src_lang = "ja"
        src_lang = "ja"
        tgt_lang = "ko"
    else:
        raise ValueError(f"未対応のモデル: {model_name}")
    
    model.eval()
    
    return model, tokenizer, src_lang, tgt_lang


def generate_translations(
    model,
    tokenizer,
    texts: list,
    tgt_lang: str,
    batch_size: int,
    max_length: int,
    num_beams: int,
    device: str,
    model_name: str
):
    """バッチ処理で翻訳を生成"""
    translations = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="翻訳生成中"):
        batch = texts[i:i + batch_size]
        
        # トークナイズ
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # 生成
        with torch.no_grad():
            if "nllb" in model_name.lower():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )
            else:  # M2M100
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )
        
        # デコード
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(decoded)
        
        # メモリ解放
        del inputs, generated
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return translations


def main():
    parser = argparse.ArgumentParser(description="教師データ生成")
    parser.add_argument("--input", type=str, default="data/splits/train.ja")
    parser.add_argument("--output", type=str, default="data/teacher/train.ko")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-1.3B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--resume-from", type=int, default=0)
    args = parser.parse_args()
    
    print("=" * 50)
    print("教師データ生成")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")
    
    # モデルロード
    model, tokenizer, src_lang, tgt_lang = load_model(args.model, device)
    print(f"言語ペア: {src_lang} → {tgt_lang}")
    
    # 入力読み込み
    input_path = Path(args.input)
    print(f"\n入力ファイル: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        ja_texts = [line.strip() for line in f]
    
    print(f"入力行数: {len(ja_texts):,}")
    
    # 途中から再開
    if args.resume_from > 0:
        print(f"行 {args.resume_from} から再開")
        ja_texts = ja_texts[args.resume_from:]
    
    # 翻訳生成
    print(f"\n翻訳生成開始...")
    print(f"  バッチサイズ: {args.batch_size}")
    print(f"  ビーム数: {args.num_beams}")
    
    translations = generate_translations(
        model=model,
        tokenizer=tokenizer,
        texts=ja_texts,
        tgt_lang=tgt_lang,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_beams=args.num_beams,
        device=device,
        model_name=args.model
    )
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 再開の場合は追記
    mode = 'a' if args.resume_from > 0 else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        f.write('\n'.join(translations))
        if mode == 'w':
            f.write('\n')
    
    print(f"\n保存完了: {output_path}")
    print(f"生成行数: {len(translations):,}")
    
    # 推定時間表示
    total_lines = len(ja_texts) + args.resume_from
    print(f"\n処理完了: {total_lines:,} / {total_lines:,} 行")


if __name__ == "__main__":
    main()