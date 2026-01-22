#!/usr/bin/env python3
"""
OPUSデータの品質調査スクリプト
- 基本統計
- アライメント品質のサンプル確認
- ノイズパターンの検出

Usage:
    python scripts/analyze_data.py
"""

import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

JA_FILE = PROJECT_ROOT / "data/raw/OpenSubtitles.ja-ko.ja"
KO_FILE = PROJECT_ROOT / "data/raw/OpenSubtitles.ja-ko.ko"


def load_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def basic_stats(lines, lang):
    """基本統計"""
    lengths = [len(line) for line in lines]
    word_counts = [len(line.split()) for line in lines]
    
    print(f"\n{'='*50}")
    print(f"【{lang}】基本統計")
    print(f"{'='*50}")
    print(f"総行数: {len(lines):,}")
    print(f"空行数: {sum(1 for l in lines if not l):,}")
    print(f"文字数 - 平均: {sum(lengths)/len(lengths):.1f}, 最大: {max(lengths)}, 最小: {min(lengths)}")
    print(f"単語数 - 平均: {sum(word_counts)/len(word_counts):.1f}, 最大: {max(word_counts)}")
    
    # 文字数分布
    print(f"\n文字数分布:")
    brackets = [(0, 10), (11, 30), (31, 50), (51, 100), (101, 200), (201, float('inf'))]
    for low, high in brackets:
        count = sum(1 for l in lengths if low <= l <= high)
        pct = count / len(lengths) * 100
        label = f"{low}-{int(high)}" if high != float('inf') else f"{low}+"
        print(f"  {label:>8}文字: {count:>8,} ({pct:5.1f}%)")


def detect_noise_patterns(lines, lang):
    """ノイズパターンの検出"""
    patterns = {
        'HTMLタグ': re.compile(r'<[^>]+>'),
        '音楽記号': re.compile(r'[♪♫♬♩]'),
        '括弧付き注釈': re.compile(r'[\[【\(（][^\]】\)）]*[\]】\)）]'),
        'URL': re.compile(r'https?://'),
        '...(省略記号多用)': re.compile(r'\.{4,}|…{2,}'),
        '数字のみ': re.compile(r'^[\d\s\.\-:]+$'),
        'クレジット系': re.compile(r'(翻訳|字幕|번역|자막|subtitle|translated)', re.I),
    }
    
    print(f"\n{'='*50}")
    print(f"【{lang}】ノイズパターン検出")
    print(f"{'='*50}")
    
    for name, pattern in patterns.items():
        matches = [(i, line) for i, line in enumerate(lines) if pattern.search(line)]
        print(f"\n{name}: {len(matches):,}件")
        if matches and len(matches) <= 5:
            for i, line in matches[:3]:
                print(f"  行{i+1}: {line[:60]}...")
        elif matches:
            for i, line in matches[:3]:
                print(f"  行{i+1}: {line[:60]}...")


def check_alignment_quality(ja_lines, ko_lines, sample_size=20):
    """アライメント品質のサンプル確認"""
    print(f"\n{'='*50}")
    print(f"【アライメント品質チェック】")
    print(f"{'='*50}")
    
    # 長さの比率で怪しいペアを検出
    suspicious = []
    for i, (ja, ko) in enumerate(zip(ja_lines, ko_lines)):
        if not ja or not ko:
            continue
        ratio = len(ja) / len(ko) if len(ko) > 0 else float('inf')
        # 極端な長さの違い（3倍以上 or 1/3以下）
        if ratio > 3 or ratio < 0.33:
            suspicious.append((i, ja, ko, ratio))
    
    print(f"\n長さ比率が極端なペア（怪しいアライメント）: {len(suspicious):,}件")
    print("\nサンプル（最初の10件）:")
    for i, ja, ko, ratio in suspicious[:10]:
        print(f"\n  行{i+1} (比率: {ratio:.2f}):")
        print(f"    JA: {ja[:50]}{'...' if len(ja) > 50 else ''}")
        print(f"    KO: {ko[:50]}{'...' if len(ko) > 50 else ''}")
    
    # ランダムサンプルで目視確認用
    import random
    random.seed(42)
    
    print(f"\n\nランダムサンプル（目視確認用）:")
    print("-" * 50)
    indices = random.sample(range(min(len(ja_lines), len(ko_lines))), min(sample_size, len(ja_lines)))
    
    for i in sorted(indices)[:10]:
        ja, ko = ja_lines[i], ko_lines[i]
        print(f"\n行{i+1}:")
        print(f"  JA: {ja}")
        print(f"  KO: {ko}")


def find_duplicate_pairs(ja_lines, ko_lines):
    """重複ペアの検出"""
    pairs = list(zip(ja_lines, ko_lines))
    pair_counts = Counter(pairs)
    
    duplicates = [(pair, count) for pair, count in pair_counts.items() if count > 1]
    duplicates.sort(key=lambda x: -x[1])
    
    print(f"\n{'='*50}")
    print(f"【重複ペア】")
    print(f"{'='*50}")
    print(f"ユニークペア数: {len(pair_counts):,}")
    print(f"重複ありペア数: {len(duplicates):,}")
    print(f"重複による余剰行数: {sum(c-1 for _, c in duplicates):,}")
    
    if duplicates:
        print(f"\n最も多い重複（上位10件）:")
        for (ja, ko), count in duplicates[:10]:
            print(f"  {count}回: JA「{ja[:30]}...」 / KO「{ko[:30]}...」")


def main():
    if not JA_FILE.exists():
        print(f"エラー: {JA_FILE} が見つかりません")
        print("OPUSファイルを data/raw/ に配置してください")
        return
    
    print("データを読み込み中...")
    ja_lines = load_lines(JA_FILE)
    ko_lines = load_lines(KO_FILE)
    
    print(f"\n日本語ファイル: {len(ja_lines):,}行")
    print(f"韓国語ファイル: {len(ko_lines):,}行")
    
    if len(ja_lines) != len(ko_lines):
        print(f"\n⚠️ 警告: 行数が一致しません！")
    
    basic_stats(ja_lines, "日本語")
    basic_stats(ko_lines, "韓国語")
    
    detect_noise_patterns(ja_lines, "日本語")
    detect_noise_patterns(ko_lines, "韓国語")
    
    check_alignment_quality(ja_lines, ko_lines)
    find_duplicate_pairs(ja_lines, ko_lines)
    
    print(f"\n{'='*50}")
    print("分析完了！")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()