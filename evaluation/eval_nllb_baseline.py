#!/usr/bin/env python3
"""
NLLB ベースライン評価（学習前）

AI Hub 100サンプルで事前学習済みNLLBの翻訳品質を測定
"""

import argparse
import time
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate

# NLLB言語コード
LANG_CODES = {
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
}


def load_data(ko_file: Path, ja_file: Path, limit: int = 100):
    """AI Hubデータを読み込む"""
    with open(ko_file, 'r', encoding='utf-8') as f:
        ko_lines = [line.strip() for line in f][:limit]
    with open(ja_file, 'r', encoding='utf-8') as f:
        ja_lines = [line.strip() for line in f][:limit]
    return ko_lines, ja_lines


def translate_batch(model, tokenizer, texts, src_lang, tgt_lang, batch_size=8):
    """バッチ翻訳"""
    tokenizer.src_lang = LANG_CODES[src_lang]
    translations = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i + batch_size]
        
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(model.device)
        
        # 強制的にターゲット言語トークンを設定
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(LANG_CODES[tgt_lang])
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=128,
                num_beams=4,
            )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
    
    return translations


def evaluate_translations(hypotheses, references):
    """翻訳品質を評価"""
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    
    bleu_result = bleu.compute(
        predictions=hypotheses,
        references=[[ref] for ref in references]
    )
    
    chrf_result = chrf.compute(
        predictions=hypotheses,
        references=[[ref] for ref in references],
        word_order=2,  # chrF++
    )
    
    return {
        "bleu": bleu_result["score"],
        "chrf": chrf_result["score"],
    }


def main():
    parser = argparse.ArgumentParser(description="NLLB ベースライン評価")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default="evaluation/data/aihub")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"NLLB ベースライン評価: ko → ja")
    print("=" * 60)
    print(f"モデル: {args.model}")
    print(f"サンプル数: {args.limit}")
    print()
    
    # データ読み込み
    data_dir = Path(args.data_dir)
    ko_lines, ja_refs = load_data(
        data_dir / "ko_reference.txt",
        data_dir / "ja_source.txt",
        limit=args.limit,
    )
    print(f"データ読み込み完了: {len(ko_lines)} サンプル")
    
    # モデル読み込み
    print(f"\nモデル読み込み中: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {num_params:,} ({num_params/1e6:.0f}M)")
    
    # 翻訳
    print(f"\n翻訳中...")
    start_time = time.time()
    
    ja_hyps = translate_batch(
        model, tokenizer, ko_lines,
        src_lang="ko", tgt_lang="ja",
        batch_size=args.batch_size,
    )
    
    elapsed = time.time() - start_time
    print(f"翻訳完了: {elapsed:.1f}秒 ({len(ko_lines)/elapsed:.1f} samples/s)")
    
    # 評価
    print(f"\n評価中...")
    results = evaluate_translations(ja_hyps, ja_refs)
    
    print()
    print("=" * 60)
    print("📊 結果")
    print("=" * 60)
    print(f"BLEU:   {results['bleu']:.2f}")
    print(f"chrF++: {results['chrf']:.2f}")
    print()
    
    # サンプル表示
    print("=" * 60)
    print("📝 サンプル翻訳")
    print("=" * 60)
    for i in [0, 1, 2, 49, 99]:
        if i < len(ko_lines):
            print(f"\n[{i+1}]")
            print(f"KO:  {ko_lines[i]}")
            print(f"REF: {ja_refs[i]}")
            print(f"HYP: {ja_hyps[i]}")


if __name__ == "__main__":
    main()
