#!/usr/bin/env python3
"""
MADLAD-400を使ったTeacher Data生成スクリプト
モデル: google/madlad400-3b-mt (Apache 2.0ライセンス)
"""

import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import re

def clean_translation(text: str) -> str:
    """翻訳結果のクリーニング"""
    if not text or not text.strip():
        return "FAILED_TRANSLATION_CLEANED"
    
    # 言語トークンを除去
    text = re.sub(r'<2[a-z]{2}>', '', text)
    text = text.strip()
    
    # 空文字列チェック
    if not text:
        return "FAILED_TRANSLATION_CLEANED"
    
    # 繰り返しパターンの検出（同じ文字が10回以上連続）
    if re.search(r'(.)\1{9,}', text):
        return "FAILED_TRANSLATION_CLEANED"
    
    # 同じ単語が5回以上連続
    words = text.split()
    if len(words) > 5:
        for i in range(len(words) - 4):
            if len(set(words[i:i+5])) == 1:
                return "FAILED_TRANSLATION_CLEANED"
    
    return text

def generate_translations(
    src_file: str,
    output_file: str,
    src_lang: str = "ko",
    tgt_lang: str = "ja",
    model_name: str = "google/madlad400-3b-mt",
    batch_size: int = 16,
    num_beams: int = 4,
    max_length: int = 128,
    device: str = "cuda"
):
    """
    MADLAD-400で翻訳を生成
    
    Args:
        src_file: 入力ファイル（1行1文）
        output_file: 出力ファイル
        src_lang: ソース言語コード（ko, ja, etc.）
        tgt_lang: ターゲット言語コード
        model_name: MADLAD-400モデル名
        batch_size: バッチサイズ
        num_beams: Beam search幅
        max_length: 最大生成長
        device: デバイス（cuda/cpu）
    """
    
    print(f"🔧 モデル読み込み: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    print(f"📖 入力ファイル読み込み: {src_file}")
    with open(src_file, 'r', encoding='utf-8') as f:
        source_texts = [line.strip() for line in f if line.strip()]
    
    total = len(source_texts)
    print(f"📊 総サンプル数: {total}")
    
    # MADLAD-400の言語コードフォーマット: <2ko>, <2ja>, etc.
    lang_prefix = f"<2{tgt_lang}>"
    
    translations = []
    failed_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i in tqdm(range(0, total, batch_size), desc="翻訳中"):
            batch = source_texts[i:i + batch_size]
            
            # MADLAD-400用のフォーマット: "<2ja> {source_text}"
            inputs = [f"{lang_prefix} {text}" for text in batch]
            
            encoded = tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3,  # 繰り返し防止
                    repetition_penalty=1.2   # 繰り返しペナルティ
                )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # クリーニング
            for j, translation in enumerate(decoded):
                cleaned = clean_translation(translation)
                translations.append(cleaned)
                out_f.write(cleaned + '\n')
                
                if cleaned == "FAILED_TRANSLATION_CLEANED":
                    failed_count += 1
    
    print(f"\n✅ 翻訳完了: {output_file}")
    print(f"📊 統計:")
    print(f"  - 総数: {total}")
    print(f"  - 成功: {total - failed_count}")
    print(f"  - 失敗: {failed_count} ({failed_count/total*100:.1f}%)")
    
    # サンプル表示
    print(f"\n--- [サンプル翻訳] ---")
    for i in range(min(5, len(source_texts))):
        print(f"原文 ({src_lang}): {source_texts[i]}")
        print(f"翻訳 ({tgt_lang}): {translations[i]}")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="MADLAD-400でTeacher Data生成")
    parser.add_argument("--src_file", required=True, help="入力ファイルパス")
    parser.add_argument("--output_file", required=True, help="出力ファイルパス")
    parser.add_argument("--src_lang", default="ko", help="ソース言語コード")
    parser.add_argument("--tgt_lang", default="ja", help="ターゲット言語コード")
    parser.add_argument("--model_name", default="google/madlad400-3b-mt", help="モデル名")
    parser.add_argument("--batch_size", type=int, default=16, help="バッチサイズ")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam search幅")
    parser.add_argument("--max_length", type=int, default=128, help="最大生成長")
    parser.add_argument("--device", default="cuda", help="デバイス（cuda/cpu）")
    
    args = parser.parse_args()
    
    generate_translations(
        src_file=args.src_file,
        output_file=args.output_file,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_length=args.max_length,
        device=args.device
    )

if __name__ == "__main__":
    main()
