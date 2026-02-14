#!/usr/bin/env python3
"""
Evaluate ko-ja model on AI Hub dataset.
"""

import argparse
from pathlib import Path
import sys
from tqdm import tqdm

# Metrics
from sacrebleu.metrics import BLEU, CHRF

# ONNX inference
import onnxruntime as ort
import sentencepiece as spm
import numpy as np


def load_model(model_dir: Path):
    """Load ONNX model and tokenizer."""
    encoder_path = model_dir / "encoder_model_quantized.onnx"
    decoder_path = model_dir / "decoder_model_quantized.onnx"
    spm_path = model_dir / "spm.model"
    
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")
    
    # Load sessions
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    encoder = ort.InferenceSession(str(encoder_path), sess_opts)
    decoder = ort.InferenceSession(str(decoder_path), sess_opts)
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(str(spm_path))
    
    return encoder, decoder, sp


def translate(text: str, encoder, decoder, sp, max_length=128, 
               repetition_penalty=1.5, no_repeat_ngram=2):
    """Translate Korean to Japanese with repetition prevention."""
    # Tokenize
    tokens = sp.EncodeAsIds(text)
    input_ids = np.array([tokens], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    
    # Encode
    encoder_out = encoder.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    encoder_hidden = encoder_out[0]
    
    # Decode with repetition prevention
    # Start with PAD token (decoder_start_token_id=0 in config)
    generated = [0]
    
    for _ in range(max_length):
        decoder_out = decoder.run(None, {
            "input_ids": np.array([generated], dtype=np.int64),
            "encoder_hidden_states": encoder_hidden,
            "encoder_attention_mask": attention_mask
        })
        logits = decoder_out[0][0, -1, :].copy()
        
        # Apply repetition penalty (skip first token)
        if repetition_penalty != 1.0 and len(generated) > 1:
            for token_id in set(generated[1:]):
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty
        
        # Block repeated n-grams
        tokens_for_ngram = generated[1:]  # Exclude start token
        if no_repeat_ngram > 0 and len(tokens_for_ngram) >= no_repeat_ngram - 1:
            ngram_prefix = tuple(tokens_for_ngram[-(no_repeat_ngram - 1):])
            for i in range(len(tokens_for_ngram) - no_repeat_ngram + 1):
                if tuple(tokens_for_ngram[i:i + no_repeat_ngram - 1]) == ngram_prefix:
                    blocked_token = tokens_for_ngram[i + no_repeat_ngram - 1]
                    logits[blocked_token] = float('-inf')
        
        next_token = int(np.argmax(logits))
        
        # Stop conditions
        if next_token == sp.eos_id() or next_token == 3:  # EOS
            break
        
        # Detect repetition (same token 3+ times in a row)
        if len(generated) >= 3 and generated[-1] == generated[-2] == next_token:
            break
        
        generated.append(next_token)
    
    # Decode, skipping the initial PAD token
    return sp.DecodeIds(generated[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, 
                        default=Path("../models/ko-ja-onnx-int8"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/aihub"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples for quick test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    
    # Load data (ko→ja)
    # Try OpenSubtitles format first, then AI Hub format
    ko_file = args.data_dir / "test.ko"
    ja_file = args.data_dir / "test.ja"
    if not ko_file.exists():
        ko_file = args.data_dir / "ko_reference.txt"
        ja_file = args.data_dir / "ja_source.txt"
    
    with open(ko_file, encoding="utf-8") as f:
        sources = [line.strip() for line in f]
    with open(ja_file, encoding="utf-8") as f:
        references = [line.strip() for line in f]
    
    if args.limit:
        sources = sources[:args.limit]
        references = references[:args.limit]
    
    print(f"Loaded {len(sources)} samples")
    
    # Load model
    print(f"Loading model from {args.model_dir}")
    encoder, decoder, sp = load_model(args.model_dir)
    
    # Translate
    hypotheses = []
    print("Translating...")
    for src in tqdm(sources):
        hyp = translate(src, encoder, decoder, sp)
        hypotheses.append(hyp)
    
    # Compute metrics
    bleu = BLEU()
    chrf = CHRF(word_order=2)  # chrF++
    
    bleu_score = bleu.corpus_score(hypotheses, [references])
    chrf_score = chrf.corpus_score(hypotheses, [references])
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS (AI Hub ko→ja)")
    print("="*60)
    print(f"Samples: {len(sources)}")
    print(f"BLEU:    {bleu_score.score:.2f}")
    print(f"chrF++:  {chrf_score.score:.2f}")
    print("="*60)
    
    # Show examples
    print("\n📝 Sample translations:")
    for i in range(min(5, len(sources))):
        print(f"\n[{i+1}]")
        print(f"  KO: {sources[i]}")
        print(f"  JA (ref): {references[i]}")
        print(f"  JA (hyp): {hypotheses[i]}")
    
    # Save results
    if args.output:
        import json
        results = {
            "model": str(args.model_dir),
            "dataset": "aihub",
            "samples": len(sources),
            "bleu": bleu_score.score,
            "chrf": chrf_score.score,
            "examples": [
                {"ko": s, "ref": r, "hyp": h}
                for s, r, h in zip(sources[:10], references[:10], hypotheses[:10])
            ]
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
