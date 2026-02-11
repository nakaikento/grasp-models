#!/usr/bin/env python3
"""
翻訳エラー分析スクリプト。

最良条件 vs 最悪条件で50文サンプリングし、エラータイプを分類。
自動指標では測れない「直訳的かどうか」を人手で判断するための材料を提供。

エラータイプ:
- literal: 文法的に正しいが不自然（直訳的）
- mistranslation: 意味が異なる（誤訳）
- unnatural: 表現がおかしい（不自然）
- omission: 情報が落ちている（情報欠落）
- good: 問題なし

使用方法:
  python3 analyze_errors.py \
    --source data/flores/ja_source.txt \
    --reference data/flores/ko_reference.txt \
    --best translations/qwen3-32b-natural.txt \
    --worst translations/qwen3-32b-zero_shot.txt \
    --n-samples 50 \
    --output results/error_samples.json
"""

import argparse
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class ErrorSample:
    index: int
    source_ja: str
    reference_ko: str
    best_ko: str
    worst_ko: str
    # 以下は人手で埋める
    best_error_type: str = ""
    worst_error_type: str = ""
    notes: str = ""


def load_lines(path: Path) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def sample_diverse_errors(sources: list[str], references: list[str],
                          best: list[str], worst: list[str],
                          n_samples: int = 50, seed: int = 42) -> list[ErrorSample]:
    """多様なエラーパターンをサンプリング"""
    random.seed(seed)
    
    n = min(len(sources), len(references), len(best), len(worst))
    
    # 差分が大きいものを優先的にサンプリング
    # (best と worst の長さの差、または表面的な類似度で判断)
    candidates = []
    for i in range(n):
        diff_score = abs(len(best[i]) - len(worst[i])) / max(len(best[i]), len(worst[i]), 1)
        candidates.append((i, diff_score))
    
    # 差分が大きい順にソートし、上位と下位からサンプリング
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 上位25% + ランダム50% + 下位25%
    n_top = n_samples // 4
    n_random = n_samples // 2
    n_bottom = n_samples - n_top - n_random
    
    selected_indices = set()
    
    # 差分が大きいもの
    for idx, _ in candidates[:n_top]:
        selected_indices.add(idx)
    
    # 差分が小さいもの
    for idx, _ in candidates[-n_bottom:]:
        selected_indices.add(idx)
    
    # ランダム
    remaining = [i for i in range(n) if i not in selected_indices]
    random.shuffle(remaining)
    for idx in remaining[:n_random]:
        selected_indices.add(idx)
    
    # ErrorSample作成
    samples = []
    for idx in sorted(selected_indices)[:n_samples]:
        samples.append(ErrorSample(
            index=idx,
            source_ja=sources[idx],
            reference_ko=references[idx],
            best_ko=best[idx],
            worst_ko=worst[idx]
        ))
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="翻訳エラー分析")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--best", type=Path, required=True, help="最良条件の翻訳")
    parser.add_argument("--worst", type=Path, required=True, help="最悪条件の翻訳")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("results/error_samples.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # 読み込み
    sources = load_lines(args.source)
    references = load_lines(args.reference)
    best = load_lines(args.best)
    worst = load_lines(args.worst)
    
    print(f"📥 Loaded {len(sources)} samples")
    print(f"📊 Best condition: {args.best.stem}")
    print(f"📊 Worst condition: {args.worst.stem}")
    
    # サンプリング
    samples = sample_diverse_errors(
        sources, references, best, worst,
        n_samples=args.n_samples,
        seed=args.seed
    )
    
    print(f"\n📝 Sampled {len(samples)} examples for error analysis")
    
    # プレビュー
    print("\n" + "="*80)
    print("PREVIEW (first 5 samples)")
    print("="*80)
    
    for sample in samples[:5]:
        print(f"\n[{sample.index}]")
        print(f"  JA:   {sample.source_ja[:60]}...")
        print(f"  REF:  {sample.reference_ko[:60]}...")
        print(f"  BEST: {sample.best_ko[:60]}...")
        print(f"  WRST: {sample.worst_ko[:60]}...")
    
    # 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metadata': {
            'source_file': str(args.source),
            'reference_file': str(args.reference),
            'best_file': str(args.best),
            'worst_file': str(args.worst),
            'n_samples': len(samples),
            'seed': args.seed
        },
        'error_types': {
            'literal': '文法的に正しいが不自然（直訳的）',
            'mistranslation': '意味が異なる（誤訳）',
            'unnatural': '表現がおかしい（不自然）',
            'omission': '情報が落ちている（情報欠落）',
            'good': '問題なし'
        },
        'samples': [asdict(s) for s in samples]
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved to {args.output}")
    print("\n📋 Next steps:")
    print("   1. Open the JSON file")
    print("   2. For each sample, fill in:")
    print("      - best_error_type: literal/mistranslation/unnatural/omission/good")
    print("      - worst_error_type: literal/mistranslation/unnatural/omission/good")
    print("      - notes: (optional) any observations")
    print("   3. Run analyze_error_results.py to compute statistics")


if __name__ == "__main__":
    main()
