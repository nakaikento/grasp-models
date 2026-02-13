#!/usr/bin/env python3
"""
M2M100-418M を AI Hub コーパスで評価。
"""

import argparse
import time
from pathlib import Path
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

def load_lines(path: Path) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/aihub/ko_reference.txt"))
    parser.add_argument("--reference", type=Path, default=Path("data/aihub/ja_source.txt"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/m2m100_eval.txt"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    sources = load_lines(args.source)[:args.limit]
    references = load_lines(args.reference)[:args.limit]
    
    print(f"📥 Source (Korean): {len(sources)} lines")
    print(f"📥 Reference (Japanese): {len(references)} lines")
    
    print("\n🔄 Loading M2M100-418M...")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    
    # GPU使用可能なら使う
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"   Device: {device}")
    
    # 翻訳
    tokenizer.src_lang = "ko"
    hypotheses = []
    
    print(f"\n🔄 Translating...")
    start = time.time()
    
    for i in range(0, len(sources), args.batch_size):
        batch = sources[i:i+args.batch_size]
        
        if (i + args.batch_size) % 100 == 0 or i == 0:
            elapsed = time.time() - start
            if elapsed > 0:
                print(f"  Progress: {i}/{len(sources)} ({i/elapsed:.1f} samples/s)")
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id("ja"),
                max_length=128,
                num_beams=1,  # greedy
            )
        
        translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
        hypotheses.extend(translations)
    
    elapsed = time.time() - start
    print(f"   ⏱️ Total: {elapsed:.1f}s ({len(sources)/elapsed:.2f} samples/s)")
    
    # 評価
    print(f"\n📊 Evaluating...")
    from sacrebleu import corpus_chrf, corpus_bleu
    
    chrf = corpus_chrf(hypotheses, [references], word_order=2)
    bleu = corpus_bleu(hypotheses, [references], tokenize='char')
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS: M2M100-418M (Ko→Ja)")
    print(f"{'='*60}")
    print(f"   Samples: {len(sources)}")
    print(f"   chrF++:  {chrf.score:.2f}")
    print(f"   BLEU:    {bleu.score:.2f}")
    print(f"{'='*60}")
    
    # サンプル
    print(f"\n📝 Sample translations:")
    for i in range(min(5, len(sources))):
        print(f"\n[{i}]")
        print(f"  KO:  {sources[i]}")
        print(f"  REF: {references[i]}")
        print(f"  HYP: {hypotheses[i]}")
    
    # 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for h in hypotheses:
            f.write(h + '\n')
    print(f"\n✅ Saved to {args.output}")

if __name__ == "__main__":
    main()
