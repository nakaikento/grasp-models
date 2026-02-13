#!/usr/bin/env python3
"""
OpenSubtitlesから評価用サンプルを抽出するスクリプト。
アニメ/ドラマの会話文を想定し、適切な長さと多様性を持つサンプルを選定。
"""

import random
import argparse
from pathlib import Path
from collections import Counter

def load_parallel_data(ko_file: Path, ja_file: Path) -> list[tuple[str, str]]:
    """並列データを読み込む"""
    with open(ko_file, 'r', encoding='utf-8') as f:
        ko_lines = [line.strip() for line in f]
    with open(ja_file, 'r', encoding='utf-8') as f:
        ja_lines = [line.strip() for line in f]
    
    assert len(ko_lines) == len(ja_lines), "行数が一致しません"
    return list(zip(ko_lines, ja_lines))

def filter_good_samples(pairs: list[tuple[str, str]], 
                        min_len: int = 10, 
                        max_len: int = 100) -> list[tuple[str, str]]:
    """品質の高いサンプルをフィルタリング"""
    filtered = []
    
    for ko, ja in pairs:
        # 長さチェック
        if not (min_len <= len(ko) <= max_len and min_len <= len(ja) <= max_len):
            continue
        
        # 空白や特殊文字だけのサンプルを除外
        if not ko.strip() or not ja.strip():
            continue
        
        # 数字だけ、記号だけを除外
        if ko.isdigit() or ja.isdigit():
            continue
        
        # ハングル/日本語が含まれているかチェック
        has_hangul = any('\uac00' <= c <= '\ud7a3' for c in ko)
        has_jp = any(('\u3040' <= c <= '\u309f') or  # ひらがな
                     ('\u30a0' <= c <= '\u30ff') or  # カタカナ
                     ('\u4e00' <= c <= '\u9fff')     # 漢字
                     for c in ja)
        
        if not has_hangul or not has_jp:
            continue
        
        filtered.append((ko, ja))
    
    return filtered

def select_diverse_samples(pairs: list[tuple[str, str]], 
                           n_samples: int = 1000,
                           seed: int = 42) -> list[tuple[str, str]]:
    """多様性を考慮してサンプルを選定"""
    random.seed(seed)
    
    # 長さで層化サンプリング
    short = [(ko, ja) for ko, ja in pairs if len(ko) < 30]    # 短文
    medium = [(ko, ja) for ko, ja in pairs if 30 <= len(ko) < 60]  # 中文
    long = [(ko, ja) for ko, ja in pairs if len(ko) >= 60]    # 長文
    
    # 比率: 短文30%, 中文50%, 長文20%
    n_short = int(n_samples * 0.3)
    n_medium = int(n_samples * 0.5)
    n_long = n_samples - n_short - n_medium
    
    samples = []
    samples.extend(random.sample(short, min(n_short, len(short))))
    samples.extend(random.sample(medium, min(n_medium, len(medium))))
    samples.extend(random.sample(long, min(n_long, len(long))))
    
    # 不足分をランダムに補充
    if len(samples) < n_samples:
        remaining = [p for p in pairs if p not in samples]
        samples.extend(random.sample(remaining, min(n_samples - len(samples), len(remaining))))
    
    random.shuffle(samples)
    return samples[:n_samples]

def save_samples(samples: list[tuple[str, str]], output_dir: Path):
    """サンプルを保存"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ko_file = output_dir / "source_ko.txt"
    ja_file = output_dir / "reference_ja.txt"
    
    with open(ko_file, 'w', encoding='utf-8') as f:
        for ko, _ in samples:
            f.write(ko + '\n')
    
    with open(ja_file, 'w', encoding='utf-8') as f:
        for _, ja in samples:
            f.write(ja + '\n')
    
    # 統計情報
    stats_file = output_dir / "stats.txt"
    ko_lengths = [len(ko) for ko, _ in samples]
    ja_lengths = [len(ja) for _, ja in samples]
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"サンプル数: {len(samples)}\n")
        f.write(f"韓国語平均長: {sum(ko_lengths)/len(ko_lengths):.1f}文字\n")
        f.write(f"韓国語最小/最大: {min(ko_lengths)}/{max(ko_lengths)}文字\n")
        f.write(f"日本語平均長: {sum(ja_lengths)/len(ja_lengths):.1f}文字\n")
        f.write(f"日本語最小/最大: {min(ja_lengths)}/{max(ja_lengths)}文字\n")
    
    print(f"✅ 保存完了: {output_dir}")
    print(f"   - {ko_file.name}: {len(samples)}行")
    print(f"   - {ja_file.name}: {len(samples)}行")
    print(f"   - {stats_file.name}")

def main():
    parser = argparse.ArgumentParser(description="評価用サンプル抽出")
    parser.add_argument("--ko", type=Path, default=Path.home() / "grasp-models/data/cleaned/cleaned.ko")
    parser.add_argument("--ja", type=Path, default=Path.home() / "grasp-models/data/cleaned/cleaned.ja")
    parser.add_argument("--output", type=Path, default=Path.home() / "grasp-models/evaluation/samples")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--min-len", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print(f"📥 データ読み込み: {args.ko}")
    pairs = load_parallel_data(args.ko, args.ja)
    print(f"   全ペア数: {len(pairs):,}")
    
    print(f"🔍 フィルタリング (長さ: {args.min_len}-{args.max_len}文字)")
    filtered = filter_good_samples(pairs, args.min_len, args.max_len)
    print(f"   フィルタ後: {len(filtered):,}")
    
    print(f"📊 多様性サンプリング (n={args.n_samples})")
    samples = select_diverse_samples(filtered, args.n_samples, args.seed)
    
    save_samples(samples, args.output)

if __name__ == "__main__":
    main()
