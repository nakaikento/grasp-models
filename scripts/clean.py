#!/usr/bin/env python3
"""
OPUSデータ クレンジングスクリプト

Usage:
    python scripts/clean.py

Input:  data/raw/OpenSubtitles.ja-ko.{ja,ko}
Output: data/cleaned/cleaned.{ja,ko}
"""

import re
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass
from collections import OrderedDict

# =============================================================================
# 設定
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class CleaningConfig:
    # 入力ファイル
    ja_input: Path = PROJECT_ROOT / "data/raw/OpenSubtitles.ja-ko.ja"
    ko_input: Path = PROJECT_ROOT / "data/raw/OpenSubtitles.ja-ko.ko"
    
    # 出力ファイル
    ja_output: Path = PROJECT_ROOT / "data/cleaned/cleaned.ja"
    ko_output: Path = PROJECT_ROOT / "data/cleaned/cleaned.ko"
    
    # 統計ファイル
    stats_output: Path = PROJECT_ROOT / "data/cleaned/stats.txt"
    
    # 長さフィルタ（文字数）
    min_chars: int = 4
    max_chars_ja: int = 100
    max_chars_ko: int = 150
    
    # 長さ比率フィルタ
    min_length_ratio: float = 0.25
    max_length_ratio: float = 4.0
    
    # 重複除去
    remove_duplicates: bool = True


config = CleaningConfig()

# =============================================================================
# ノイズパターン
# =============================================================================

REMOVE_PATTERNS = [
    re.compile(r'[♪♫♬♩♭♮♯]'),
    re.compile(r'^[\d\s\.\-:：／/]+$'),
    re.compile(r'(字幕|翻訳|번역|자막|subtitle|translated|subscene|opensubtitles)', re.I),
    re.compile(r'https?://'),
    re.compile(r'^[A-Z\s\-:]+$'),
]

CLEAN_PATTERNS = [
    (re.compile(r'<[^>]+>'), ''),
    (re.compile(r'^[\-\-ー]+\s*'), ''),
    (re.compile(r'\([A-Za-z][A-Za-z\s]+\)'), ''),
    (re.compile(r'\.{4,}'), '...'),
    (re.compile(r'…{2,}'), '…'),
    (re.compile(r'\s{2,}'), ' '),
]

# =============================================================================
# クレンジング関数
# =============================================================================

def should_remove_line(ja: str, ko: str) -> Tuple[bool, str]:
    for pattern in REMOVE_PATTERNS:
        if pattern.search(ja) or pattern.search(ko):
            return True, f"pattern: {pattern.pattern[:30]}"
    return False, ""


def clean_text(text: str) -> str:
    for pattern, replacement in CLEAN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


def length_filter(ja: str, ko: str) -> Tuple[bool, str]:
    ja_len, ko_len = len(ja), len(ko)
    
    if ja_len < config.min_chars:
        return False, f"ja too short: {ja_len}"
    if ko_len < config.min_chars:
        return False, f"ko too short: {ko_len}"
    if ja_len > config.max_chars_ja:
        return False, f"ja too long: {ja_len}"
    if ko_len > config.max_chars_ko:
        return False, f"ko too long: {ko_len}"
    
    ratio = ja_len / ko_len if ko_len > 0 else float('inf')
    if ratio < config.min_length_ratio or ratio > config.max_length_ratio:
        return False, f"length ratio: {ratio:.2f}"
    
    return True, ""


def process_data(ja_lines: List[str], ko_lines: List[str]) -> Tuple[List[str], List[str], dict]:
    stats = {
        'total': len(ja_lines),
        'removed_by_pattern': 0,
        'removed_by_length': 0,
        'removed_by_duplicate': 0,
        'passed': 0,
    }
    
    cleaned_pairs = []
    
    print("Stage 1-2: ルールベース除去 + 長さフィルタ...")
    
    for i, (ja, ko) in enumerate(zip(ja_lines, ko_lines)):
        if i % 200000 == 0:
            print(f"  処理中: {i:,} / {len(ja_lines):,}")
        
        should_remove, _ = should_remove_line(ja, ko)
        if should_remove:
            stats['removed_by_pattern'] += 1
            continue
        
        ja_clean = clean_text(ja)
        ko_clean = clean_text(ko)
        
        if not ja_clean or not ko_clean:
            stats['removed_by_pattern'] += 1
            continue
        
        passed, _ = length_filter(ja_clean, ko_clean)
        if not passed:
            stats['removed_by_length'] += 1
            continue
        
        cleaned_pairs.append((ja_clean, ko_clean))
    
    if config.remove_duplicates:
        print("Stage 3: 重複除去...")
        before_dedup = len(cleaned_pairs)
        unique_pairs = list(OrderedDict.fromkeys(cleaned_pairs))
        stats['removed_by_duplicate'] = before_dedup - len(unique_pairs)
        cleaned_pairs = unique_pairs
    
    stats['passed'] = len(cleaned_pairs)
    
    ja_cleaned = [pair[0] for pair in cleaned_pairs]
    ko_cleaned = [pair[1] for pair in cleaned_pairs]
    
    return ja_cleaned, ko_cleaned, stats


def print_and_save_stats(stats: dict):
    output = []
    output.append("=" * 50)
    output.append("クレンジング結果")
    output.append("=" * 50)
    output.append(f"入力行数:           {stats['total']:>10,}")
    output.append(f"パターン除去:       {stats['removed_by_pattern']:>10,} ({stats['removed_by_pattern']/stats['total']*100:.1f}%)")
    output.append(f"長さフィルタ除去:   {stats['removed_by_length']:>10,} ({stats['removed_by_length']/stats['total']*100:.1f}%)")
    output.append(f"重複除去:           {stats['removed_by_duplicate']:>10,} ({stats['removed_by_duplicate']/stats['total']*100:.1f}%)")
    output.append("-" * 50)
    output.append(f"出力行数:           {stats['passed']:>10,} ({stats['passed']/stats['total']*100:.1f}%)")
    
    text = '\n'.join(output)
    print(text)
    
    config.stats_output.parent.mkdir(parents=True, exist_ok=True)
    with open(config.stats_output, 'w', encoding='utf-8') as f:
        f.write(text)


def show_sample(ja_lines: List[str], ko_lines: List[str], n: int = 10):
    import random
    random.seed(42)
    
    print("\n" + "=" * 50)
    print("クレンジング後のサンプル")
    print("=" * 50)
    
    indices = random.sample(range(len(ja_lines)), min(n, len(ja_lines)))
    for i in sorted(indices):
        print(f"\n行{i+1}:")
        print(f"  JA: {ja_lines[i]}")
        print(f"  KO: {ko_lines[i]}")


def main():
    print(f"入力: {config.ja_input}")
    print(f"出力: {config.ja_output}")
    print()
    
    # 入力ファイル確認
    if not config.ja_input.exists():
        print(f"エラー: {config.ja_input} が見つかりません")
        print("OPUSファイルを data/raw/ に配置してください")
        return
    
    print("データを読み込み中...")
    with open(config.ja_input, 'r', encoding='utf-8') as f:
        ja_lines = [line.strip() for line in f]
    with open(config.ko_input, 'r', encoding='utf-8') as f:
        ko_lines = [line.strip() for line in f]
    
    print(f"読み込み完了: {len(ja_lines):,}行\n")
    
    # クレンジング
    ja_cleaned, ko_cleaned, stats = process_data(ja_lines, ko_lines)
    
    # 統計
    print_and_save_stats(stats)
    
    # サンプル
    show_sample(ja_cleaned, ko_cleaned)
    
    # 保存
    print(f"\n保存中...")
    config.ja_output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config.ja_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ja_cleaned))
    with open(config.ko_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ko_cleaned))
    
    print(f"  {config.ja_output}")
    print(f"  {config.ko_output}")
    print("\n完了！")


if __name__ == "__main__":
    main()
