#!/usr/bin/env python3
"""
End-to-End evaluation: Audio (Korean) → Japanese translation

Measures the complete pipeline:
1. Audio → Korean ASR
2. Korean → Japanese translation
3. Compare with reference Japanese subtitles
"""

import argparse
import json
from pathlib import Path
from evaluate_asr import run_asr, load_audio
from evaluate_translation import translate_korean_to_japanese, calculate_metrics
from vtt_to_text import vtt_to_text


def main():
    parser = argparse.ArgumentParser(description="End-to-end Korean audio → Japanese translation evaluation")
    parser.add_argument("audio", type=Path, help="Audio file (.wav)")
    parser.add_argument("reference_ja", type=Path, help="Reference Japanese subtitles (.vtt or .txt)")
    parser.add_argument("--asr-model", type=Path, 
                       default=Path(__file__).parent.parent / "models" / "korean-asr",
                       help="Korean ASR model directory")
    parser.add_argument("--translation-model", type=Path, 
                       default=Path(__file__).parent.parent / "models" / "ko-ja-onnx-int8",
                       help="Translation model directory")
    parser.add_argument("--reference-ko", type=Path, help="Optional: Korean reference for ASR evaluation")
    parser.add_argument("-o", "--output", type=Path, help="Output results file")
    
    args = parser.parse_args()
    
    # Load reference Japanese
    if args.reference_ja.suffix == '.vtt':
        reference_ja = vtt_to_text(args.reference_ja)
    else:
        reference_ja = args.reference_ja.read_text(encoding='utf-8')
    
    # Optional: Load reference Korean for ASR eval
    reference_ko = None
    if args.reference_ko:
        if args.reference_ko.suffix == '.vtt':
            reference_ko = vtt_to_text(args.reference_ko)
        else:
            reference_ko = args.reference_ko.read_text(encoding='utf-8')
    
    print("="*60)
    print("🚀 End-to-End Evaluation: Korean Audio → Japanese")
    print("="*60)
    print(f"Audio: {args.audio}")
    print(f"Reference (Japanese): {args.reference_ja}")
    if reference_ko:
        print(f"Reference (Korean): {args.reference_ko}")
    print()
    
    # Step 1: ASR (Audio → Korean text)
    print("="*60)
    print("STEP 1: Korean ASR")
    print("="*60)
    korean_text = run_asr(args.audio, args.asr_model)
    print(f"ASR Output (Korean): {korean_text[:200]}...")
    
    asr_metrics = None
    if reference_ko:
        from evaluate_asr import calculate_wer
        asr_metrics = calculate_wer(reference_ko, korean_text)
        if asr_metrics:
            print(f"ASR WER: {asr_metrics['wer']:.2f}%")
            print(f"ASR CER: {asr_metrics['cer']:.2f}%")
    
    # Step 2: Translation (Korean → Japanese)
    print("\n" + "="*60)
    print("STEP 2: Korean → Japanese Translation")
    print("="*60)
    japanese_text = translate_korean_to_japanese(korean_text, args.translation_model)
    print(f"Translation Output (Japanese): {japanese_text[:200]}...")
    
    # Step 3: Evaluate translation quality
    print("\n" + "="*60)
    print("STEP 3: Translation Quality Evaluation")
    print("="*60)
    print(f"Reference (Japanese): {reference_ja[:200]}...")
    
    translation_metrics = calculate_metrics(reference_ja, japanese_text)
    
    if translation_metrics:
        print(f"\nTranslation BLEU: {translation_metrics['bleu']:.2f}")
        print(f"Translation chrF: {translation_metrics['chrf']:.2f}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    if asr_metrics:
        print(f"ASR Quality:")
        print(f"  - WER: {asr_metrics['wer']:.2f}%")
        print(f"  - CER: {asr_metrics['cer']:.2f}%")
    
    if translation_metrics:
        print(f"Translation Quality:")
        print(f"  - BLEU: {translation_metrics['bleu']:.2f}")
        print(f"  - chrF: {translation_metrics['chrf']:.2f}")
    
    # Save results
    if args.output:
        results = {
            'audio': str(args.audio),
            'reference_japanese': str(args.reference_ja),
            'reference_korean': str(args.reference_ko) if args.reference_ko else None,
            'asr_output': korean_text,
            'translation_output': japanese_text,
            'reference_japanese_text': reference_ja,
            'reference_korean_text': reference_ko,
            'asr_metrics': asr_metrics,
            'translation_metrics': translation_metrics
        }
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Results saved: {args.output}")
    
    print("="*60)


if __name__ == "__main__":
    main()
