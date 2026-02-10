#!/usr/bin/env python3
"""
Evaluate Korean ASR accuracy using Word Error Rate (WER).

Compares:
- Reference: Korean subtitles from TED talk
- Hypothesis: ASR output from Grasp Korean model
"""

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort
from vtt_to_text import vtt_to_text


def load_audio(audio_path: Path, target_sr: int = 16000):
    """Load audio file and resample to target sample rate."""
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed (simple decimation for now)
    if sr != target_sr:
        step = sr // target_sr
        audio = audio[::step]
    
    return audio.astype(np.float32)


def run_asr(audio_path: Path, model_dir: Path):
    """
    Run Korean ASR on audio file.
    
    Args:
        audio_path: Path to audio file (.wav)
        model_dir: Path to Korean ASR model directory
    
    Returns:
        str: Recognized text
    """
    # TODO: Implement sherpa-onnx ASR inference
    # This is a placeholder - we'll need to integrate the actual ASR code
    
    print(f"🎙️ Running ASR on: {audio_path}")
    print(f"📁 Model: {model_dir}")
    
    # For now, return placeholder
    return "ASR 구현 예정"


def calculate_wer(reference: str, hypothesis: str):
    """
    Calculate Word Error Rate (WER).
    
    WER = (S + D + I) / N
    where:
        S = substitutions
        D = deletions
        I = insertions
        N = number of words in reference
    """
    try:
        import jiwer
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)  # Character Error Rate
        
        return {
            'wer': wer * 100,  # Convert to percentage
            'cer': cer * 100,
        }
    except ImportError:
        print("⚠️  jiwer not installed. Install with: pip install jiwer")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Korean ASR accuracy")
    parser.add_argument("audio", type=Path, help="Audio file (.wav)")
    parser.add_argument("reference", type=Path, help="Reference Korean subtitles (.vtt or .txt)")
    parser.add_argument("--model-dir", type=Path, 
                       default=Path(__file__).parent.parent / "models" / "korean-asr",
                       help="Korean ASR model directory")
    parser.add_argument("-o", "--output", type=Path, help="Output results file")
    
    args = parser.parse_args()
    
    # Load reference text
    if args.reference.suffix == '.vtt':
        reference_text = vtt_to_text(args.reference)
    else:
        reference_text = args.reference.read_text(encoding='utf-8')
    
    print("="*60)
    print("📊 Korean ASR Evaluation")
    print("="*60)
    print(f"Audio: {args.audio}")
    print(f"Reference: {args.reference}")
    print(f"Model: {args.model_dir}")
    print()
    
    # Run ASR
    hypothesis_text = run_asr(args.audio, args.model_dir)
    
    print(f"\n📝 Reference (Korean):")
    print(f"   {reference_text[:200]}...")
    print(f"\n🎙️ ASR Output (Korean):")
    print(f"   {hypothesis_text[:200]}...")
    
    # Calculate metrics
    metrics = calculate_wer(reference_text, hypothesis_text)
    
    if metrics:
        print(f"\n📊 Results:")
        print(f"   WER: {metrics['wer']:.2f}%")
        print(f"   CER: {metrics['cer']:.2f}%")
        
        # Save results
        if args.output:
            import json
            results = {
                'audio': str(args.audio),
                'reference': str(args.reference),
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
