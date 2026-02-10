#!/usr/bin/env python3
"""
Quick test script for translation evaluation with recommended metrics.

Usage:
    python quick_test.py "한국어 텍스트" "日本語参照翻訳"
    
Or use built-in test cases:
    python quick_test.py --test
"""

import argparse
import sys
from pathlib import Path

# Import from evaluate_translation.py
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_translation import calculate_metrics, translate_korean_to_japanese


def test_with_samples():
    """Run tests with built-in sample translations."""
    
    test_cases = [
        {
            "name": "Test 1: Good translation",
            "korean": "나는 남자 친구를 아는 줄 알았어",
            "reference": "彼氏を知っていると思っていました",
            "hypothesis": "彼氏を知ってると思ってた"
        },
        {
            "name": "Test 2: Perfect match",
            "korean": "안녕하세요",
            "reference": "こんにちは",
            "hypothesis": "こんにちは"
        },
        {
            "name": "Test 3: Poor translation",
            "korean": "나는 학교에 갑니다",
            "reference": "私は学校に行きます",
            "hypothesis": "私は銀行に行きます"
        }
    ]
    
    print("="*60)
    print("🧪 QUICK TEST: Translation Metrics")
    print("="*60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test['name']}")
        print(f"{'='*60}")
        print(f"Korean:     {test['korean']}")
        print(f"Reference:  {test['reference']}")
        print(f"Hypothesis: {test['hypothesis']}")
        
        metrics = calculate_metrics(test['reference'], test['hypothesis'])
        
        if metrics:
            print(f"\n📊 Results:")
            
            chrf = metrics.get('chrf++')
            if chrf is not None:
                emoji = "🏆" if chrf >= 60 else "✅" if chrf >= 50 else "⚠️" if chrf >= 40 else "❌"
                print(f"   chrF++:    {chrf:6.2f} {emoji}")
            
            bleu = metrics.get('bleu')
            if bleu is not None:
                emoji = "🏆" if bleu >= 40 else "✅" if bleu >= 30 else "⚠️" if bleu >= 20 else "❌"
                print(f"   BLEU:      {bleu:6.2f} {emoji}")
            
            bert_f1 = metrics.get('bertscore_f1')
            if bert_f1 is not None:
                emoji = "🏆" if bert_f1 >= 0.90 else "✅" if bert_f1 >= 0.85 else "⚠️"
                print(f"   BERTScore: {bert_f1:6.4f} {emoji}")
    
    print("\n" + "="*60)
    print("✅ Test completed!")
    print("="*60)


def test_custom(korean: str, reference: str, model_dir: Path = None):
    """Test with custom Korean text and Japanese reference."""
    
    print("="*60)
    print("🧪 CUSTOM TEST: Translation Evaluation")
    print("="*60)
    print(f"Korean:    {korean}")
    print(f"Reference: {reference}")
    
    # Translate if model_dir provided
    if model_dir and model_dir.exists():
        print(f"\n🔄 Translating with model: {model_dir}")
        try:
            hypothesis = translate_korean_to_japanese(korean, model_dir)
            print(f"Translation: {hypothesis}")
        except Exception as e:
            print(f"❌ Translation failed: {e}")
            print("Using reference as hypothesis for metric calculation only")
            hypothesis = reference
    else:
        print("\n⚠️  No model provided. Testing metrics only.")
        print("Using a sample translation...")
        hypothesis = reference  # Use reference as hypothesis for demo
    
    # Calculate metrics
    print("\n🔍 Calculating metrics...")
    metrics = calculate_metrics(reference, hypothesis)
    
    if metrics:
        print("\n📊 Results:")
        
        chrf = metrics.get('chrf++')
        if chrf is not None:
            print(f"\n🎯 chrF++: {chrf:.2f}")
            if chrf >= 60:
                print("   🏆 Excellent!")
            elif chrf >= 50:
                print("   ✅ Good quality")
            elif chrf >= 40:
                print("   ⚠️ Needs improvement")
            else:
                print("   ❌ Poor quality")
        
        bleu = metrics.get('bleu')
        if bleu is not None:
            print(f"\n📏 BLEU: {bleu:.2f}")
        
        bert_f1 = metrics.get('bertscore_f1')
        if bert_f1 is not None:
            print(f"\n🧠 BERTScore F1: {bert_f1:.4f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Quick test for translation evaluation metrics"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run built-in test cases"
    )
    parser.add_argument(
        "korean",
        nargs="?",
        help="Korean source text"
    )
    parser.add_argument(
        "reference",
        nargs="?",
        help="Japanese reference translation"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Translation model directory (for live translation)"
    )
    
    args = parser.parse_args()
    
    if args.test or (not args.korean and not args.reference):
        test_with_samples()
    elif args.korean and args.reference:
        test_custom(args.korean, args.reference, args.model_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
