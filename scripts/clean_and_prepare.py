#!/usr/bin/env python3
"""
データクリーニング＆準備スクリプト

昨日の反省を踏まえた改善版：
1. FAILED_TRANSLATION除去
2. Qwenメタテキスト除去（「申し訳ありません」等）
3. 極端に短い/長いペア除去
4. 重複除去
5. Train/Val/Test分割

Usage:
    python scripts/clean_and_prepare.py
"""

import re
from pathlib import Path
from collections import Counter
import random

# パス設定
DATA_DIR = Path("data")
RAW_KO = DATA_DIR / "raw/OpenSubtitles.ja-ko.ko"
TEACHER_JA = DATA_DIR / "teacher/OpenSubtitles-Qwen72B.ja-ko.ja"
OUTPUT_DIR = DATA_DIR / "splits_v2"

# クリーニングパターン
CONTAMINATION_PATTERNS = [
    # FAILED_TRANSLATION
    r"^FAILED_TRANSLATION$",
    r"^FAILED$",
    
    # Qwenメタテキスト（翻訳拒否系）
    r"申し訳ありません",
    r"翻訳できません",
    r"翻訳する韓国語",
    r"翻訳不可",
    r"提供いただいた",
    r"テキストが必要",
    r"韓国語のテキストを",
    r"입력해 주세요",  # 韓国語混入
    
    # 空白・特殊文字のみ
    r"^[\s\.\-\?\!…♪]+$",
    r"^$",
]

# 文字数制限
MIN_CHARS = 5    # 最小文字数
MAX_CHARS = 500  # 最大文字数

# 分割比率
TRAIN_RATIO = 0.98
VAL_RATIO = 0.01
TEST_RATIO = 0.01


def load_parallel_data(ko_path: Path, ja_path: Path) -> list[tuple[str, str]]:
    """並列データをロード"""
    print(f"Loading {ko_path}...")
    with open(ko_path, 'r', encoding='utf-8') as f:
        ko_lines = [line.strip() for line in f]
    
    print(f"Loading {ja_path}...")
    with open(ja_path, 'r', encoding='utf-8') as f:
        ja_lines = [line.strip() for line in f]
    
    assert len(ko_lines) == len(ja_lines), \
        f"Line count mismatch: {len(ko_lines)} vs {len(ja_lines)}"
    
    return list(zip(ko_lines, ja_lines))


def is_contaminated(text: str) -> bool:
    """汚染データかどうかチェック"""
    for pattern in CONTAMINATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def is_valid_pair(ko: str, ja: str) -> tuple[bool, str]:
    """有効なペアかどうかチェック"""
    # 空チェック
    if not ko or not ja:
        return False, "empty"
    
    # 文字数チェック
    if len(ko) < MIN_CHARS or len(ja) < MIN_CHARS:
        return False, "too_short"
    if len(ko) > MAX_CHARS or len(ja) > MAX_CHARS:
        return False, "too_long"
    
    # 汚染チェック
    if is_contaminated(ja):
        return False, "contaminated_ja"
    if is_contaminated(ko):
        return False, "contaminated_ko"
    
    return True, "ok"


def clean_data(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """データをクリーニング"""
    cleaned = []
    stats = Counter()
    
    print("\nCleaning data...")
    for ko, ja in pairs:
        valid, reason = is_valid_pair(ko, ja)
        stats[reason] += 1
        
        if valid:
            cleaned.append((ko, ja))
    
    print("\n=== Cleaning Stats ===")
    print(f"Original: {len(pairs):,}")
    print(f"Cleaned:  {len(cleaned):,}")
    print(f"Removed:  {len(pairs) - len(cleaned):,}")
    print("\nRemoval reasons:")
    for reason, count in stats.most_common():
        if reason != "ok":
            print(f"  {reason}: {count:,}")
    
    return cleaned


def deduplicate(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """重複除去"""
    seen = set()
    unique = []
    duplicates = 0
    
    for ko, ja in pairs:
        key = (ko, ja)
        if key not in seen:
            seen.add(key)
            unique.append((ko, ja))
        else:
            duplicates += 1
    
    print(f"\nDuplicates removed: {duplicates:,}")
    return unique


def split_data(pairs: list[tuple[str, str]], seed: int = 42) -> dict:
    """データをtrain/val/testに分割"""
    random.seed(seed)
    random.shuffle(pairs)
    
    n = len(pairs)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    
    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:],
    }
    
    print("\n=== Split Stats ===")
    for name, data in splits.items():
        print(f"{name}: {len(data):,}")
    
    return splits


def save_splits(splits: dict, output_dir: Path):
    """分割データを保存"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, pairs in splits.items():
        ko_path = output_dir / f"{split_name}.ko"
        ja_path = output_dir / f"{split_name}.ja"
        
        with open(ko_path, 'w', encoding='utf-8') as f:
            for ko, _ in pairs:
                f.write(ko + '\n')
        
        with open(ja_path, 'w', encoding='utf-8') as f:
            for _, ja in pairs:
                f.write(ja + '\n')
        
        print(f"Saved: {ko_path}, {ja_path}")


def main():
    print("=" * 50)
    print("Data Cleaning & Preparation")
    print("=" * 50)
    
    # 1. データロード
    pairs = load_parallel_data(RAW_KO, TEACHER_JA)
    
    # 2. クリーニング
    cleaned = clean_data(pairs)
    
    # 3. 重複除去
    unique = deduplicate(cleaned)
    
    # 4. 分割
    splits = split_data(unique)
    
    # 5. 保存
    save_splits(splits, OUTPUT_DIR)
    
    print("\n" + "=" * 50)
    print("✅ Done!")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
