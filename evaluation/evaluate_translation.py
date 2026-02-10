#!/usr/bin/env python3
"""
Evaluate Korean→Japanese translation accuracy.

Compares:
- Reference: Japanese subtitles from TED talk
- Hypothesis: Translation output from Grasp ko-ja model
"""

import argparse
import json
from pathlib import Path
import sys
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from vtt_to_text import vtt_to_text


def translate_korean_to_japanese(text: str, model_dir: Path):
    """
    Translate Korean text to Japanese using ONNX model.
    
    Args:
        text: Korean text
        model_dir: Path to ko-ja-onnx-int8 model directory
    
    Returns:
        str: Japanese translation
    """
    encoder_path = model_dir / "encoder_model_quantized.onnx"
    decoder_path = model_dir / "decoder_model_quantized.onnx"
    
    if not encoder_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(f"Model not found in: {model_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-ja")
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="np", padding=True)
    
    # Encoder inference
    encoder_session = ort.InferenceSession(str(encoder_path))
    encoder_outputs = encoder_session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
    )
    
    # Decoder inference (Greedy decoding)
    decoder_session = ort.InferenceSession(str(decoder_path))
    
    decoder_input_ids = np.array([[tokenizer.bos_token_id]], dtype=np.int64)
    encoder_hidden_states = encoder_outputs[0]
    
    generated_tokens = []
    max_length = 512
    
    for _ in range(max_length):
        decoder_outputs = decoder_session.run(
            None,
            {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": inputs["attention_mask"].astype(np.int64)
            }
        )
        
        # Get next token (greedy)
        next_token_logits = decoder_outputs[0][0, -1, :]
        next_token = np.argmax(next_token_logits)
        
        if next_token == tokenizer.eos_token_id:
            break
        
        generated_tokens.append(int(next_token))
        decoder_input_ids = np.array([[next_token]], dtype=np.int64)
    
    # Decode
    translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return translation


def calculate_metrics(reference: str, hypothesis: str):
    """
    Calculate translation quality metrics.
    
    Metrics (WMT2023 recommended):
    - chrF++: Character + word n-gram F-score (PRIMARY)
    - BLEU: Bilingual Evaluation Understudy (BASELINE)
    - BERTScore: Semantic similarity (SEMANTIC)
    """
    metrics = {}
    
    # 1. chrF++ (PRIMARY) - Character + word n-grams
    try:
        from sacrebleu import corpus_chrf
        
        # chrF++ with word order (word_order=2)
        chrf_pp = corpus_chrf([hypothesis], [[reference]], word_order=2)
        metrics['chrf++'] = round(chrf_pp.score, 2)
        
    except ImportError:
        print("⚠️  sacrebleu not installed. Install with: pip install sacrebleu")
        metrics['chrf++'] = None
    
    # 2. BLEU (BASELINE) - Industry standard
    try:
        from sacrebleu import corpus_bleu
        
        # Use character-level tokenization for Japanese
        # (ja-mecab requires MeCab installation, so we use char for simplicity)
        bleu = corpus_bleu([hypothesis], [[reference]], tokenize='char')
        metrics['bleu'] = round(bleu.score, 2)
        
    except ImportError:
        print("⚠️  sacrebleu not installed for BLEU")
        metrics['bleu'] = None
    except Exception as e:
        print(f"⚠️  BLEU calculation failed: {e}")
        metrics['bleu'] = None
    
    # 3. BERTScore (SEMANTIC) - Meaning similarity
    try:
        import bert_score
        
        # Use multilingual BERT (supports Japanese)
        P, R, F1 = bert_score.score(
            [hypothesis], 
            [reference],
            lang="ja",
            verbose=False
        )
        
        metrics['bertscore_f1'] = round(F1.item(), 4)
        metrics['bertscore_precision'] = round(P.item(), 4)
        metrics['bertscore_recall'] = round(R.item(), 4)
        
    except ImportError as e:
        # BERTScore is optional
        metrics['bertscore_f1'] = None
    except Exception as e:
        # Log error but continue (BERTScore is optional)
        import traceback
        print(f"⚠️  BERTScore calculation skipped: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        metrics['bertscore_f1'] = None
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Korean→Japanese translation")
    parser.add_argument("source", type=Path, help="Source Korean text (.vtt or .txt)")
    parser.add_argument("reference", type=Path, help="Reference Japanese translation (.vtt or .txt)")
    parser.add_argument("--model-dir", type=Path, 
                       default=Path(__file__).parent.parent / "models" / "ko-ja-onnx-int8",
                       help="Translation model directory")
    parser.add_argument("-o", "--output", type=Path, help="Output results file")
    
    args = parser.parse_args()
    
    # Load source text
    if args.source.suffix == '.vtt':
        source_text = vtt_to_text(args.source)
    else:
        source_text = args.source.read_text(encoding='utf-8')
    
    # Load reference translation
    if args.reference.suffix == '.vtt':
        reference_text = vtt_to_text(args.reference)
    else:
        reference_text = args.reference.read_text(encoding='utf-8')
    
    print("="*60)
    print("🌐 Korean→Japanese Translation Evaluation")
    print("="*60)
    print(f"Source: {args.source}")
    print(f"Reference: {args.reference}")
    print(f"Model: {args.model_dir}")
    print()
    
    # Translate
    print("🔄 Translating...")
    hypothesis_text = translate_korean_to_japanese(source_text, args.model_dir)
    
    print(f"\n📝 Source (Korean):")
    print(f"   {source_text[:200]}...")
    print(f"\n📚 Reference (Japanese):")
    print(f"   {reference_text[:200]}...")
    print(f"\n🌐 Translation (Japanese):")
    print(f"   {hypothesis_text[:200]}...")
    
    # Calculate metrics
    print("\n🔍 Calculating metrics...")
    metrics = calculate_metrics(reference_text, hypothesis_text)
    
    if metrics:
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS (WMT2023 Recommended Metrics)")
        print("="*60)
        
        # Primary metric: chrF++
        chrf_pp = metrics.get('chrf++')
        if chrf_pp is not None:
            chrf_emoji = "🏆" if chrf_pp >= 60 else "✅" if chrf_pp >= 50 else "⚠️" if chrf_pp >= 40 else "❌"
            print(f"\n🎯 PRIMARY: chrF++ (character + word n-gram)")
            print(f"   Score: {chrf_pp:.2f} {chrf_emoji}")
            print(f"   Target: > 50 (Good), > 60 (Excellent)")
        
        # Baseline: BLEU
        bleu = metrics.get('bleu')
        if bleu is not None:
            bleu_emoji = "🏆" if bleu >= 40 else "✅" if bleu >= 30 else "⚠️" if bleu >= 20 else "❌"
            print(f"\n📏 BASELINE: BLEU (industry standard)")
            print(f"   Score: {bleu:.2f} {bleu_emoji}")
            print(f"   Target: > 30 (Good), > 40 (Excellent)")
        
        # Semantic: BERTScore
        bert_f1 = metrics.get('bertscore_f1')
        if bert_f1 is not None:
            bert_emoji = "🏆" if bert_f1 >= 0.90 else "✅" if bert_f1 >= 0.85 else "⚠️" if bert_f1 >= 0.80 else "❌"
            print(f"\n🧠 SEMANTIC: BERTScore (meaning similarity)")
            print(f"   F1:        {bert_f1:.4f} {bert_emoji}")
            print(f"   Precision: {metrics.get('bertscore_precision', 0):.4f}")
            print(f"   Recall:    {metrics.get('bertscore_recall', 0):.4f}")
            print(f"   Target: > 0.85 (Good), > 0.90 (Excellent)")
        
        # Overall assessment
        print("\n" + "="*60)
        print("🎯 OVERALL ASSESSMENT")
        print("="*60)
        
        if chrf_pp and bleu:
            if chrf_pp >= 60 and bleu >= 40:
                assessment = "🏆 EXCELLENT - Production ready"
            elif chrf_pp >= 50 and bleu >= 30:
                assessment = "✅ GOOD - Practical quality"
            elif chrf_pp >= 40 and bleu >= 20:
                assessment = "⚠️ NEEDS IMPROVEMENT - Some issues"
            else:
                assessment = "❌ POOR - Significant problems"
            
            print(f"   {assessment}")
        
        # Save results
        if args.output:
            results = {
                'source': str(args.source),
                'reference': str(args.reference),
                'source_text': source_text,
                'reference_text': reference_text,
                'hypothesis_text': hypothesis_text,
                'metrics': metrics
            }
            
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ Results saved: {args.output}")
    
    print("="*60)


if __name__ == "__main__":
    main()
