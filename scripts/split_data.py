#!/usr/bin/env python3
"""
データ分割スクリプト

train/val/test に分割

Usage:
    python scripts/split_data.py

Input:  data/cleaned/cleaned.{ja,ko}
Output: data/splits/{train,val,test}.{ja,ko}
"""

from pathlib import Path
from dataclasses import dataclass
import random

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class SplitConfig:
    # 入力
    ja_input: Path = PROJECT_ROOT / "data/cleaned/cleaned.ja"
    ko_input: Path = PROJECT_ROOT / "data/cleaned/cleaned.ko"
    
    # 出力ディレクトリ
    output_dir: Path = PROJECT_ROOT / "data/splits"
    
    # 分割サイズ
    val_size: int = 5000
    test_size: int = 5000
    
    # シード
    random_seed: int = 42


config = SplitConfig()


def load_parallel_data(ja_path: Path, ko_path: Path):
    """並列データを読み込み"""
    with open(ja_path, 'r', encoding='utf-8') as f:
        ja_lines = [line.strip() for line in f]
    with open(ko_path, 'r', encoding='utf-8') as f:
        ko_lines = [line.strip() for line in f]
    
    assert len(ja_lines) == len(ko_lines), "行数が一致しません"
    return list(zip(ja_lines, ko_lines))


def split_data(pairs, val_size, test_size, seed):
    """データを分割"""
    random.seed(seed)
    
    # シャッフル
    pairs = pairs.copy()
    random.shuffle(pairs)
    
    # 分割
    test = pairs[:test_size]
    val = pairs[test_size:test_size + val_size]
    train = pairs[test_size + val_size:]
    
    return train, val, test


def save_split(pairs, output_dir: Path, name: str):
    """分割データを保存"""
    ja_path = output_dir / f"{name}.ja"
    ko_path = output_dir / f"{name}.ko"
    
    ja_lines = [p[0] for p in pairs]
    ko_lines = [p[1] for p in pairs]
    
    with open(ja_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ja_lines))
    
    with open(ko_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ko_lines))
    
    print(f"  {name}: {len(pairs):,}行 -> {ja_path.name}, {ko_path.name}")


def main():
    print("=" * 50)
    print("データ分割")
    print("=" * 50)
    
    # 入力確認
    if not config.ja_input.exists():
        print(f"エラー: {config.ja_input} が見つかりません")
        return
    
    # 読み込み
    print(f"\nデータを読み込み中...")
    pairs = load_parallel_data(config.ja_input, config.ko_input)
    print(f"読み込み完了: {len(pairs):,}ペア")
    
    # 分割
    print(f"\n分割中 (val={config.val_size:,}, test={config.test_size:,})...")
    train, val, test = split_data(
        pairs, 
        config.val_size, 
        config.test_size, 
        config.random_seed
    )
    
    # 保存
    print(f"\n保存中...")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    save_split(train, config.output_dir, "train")
    save_split(val, config.output_dir, "val")
    save_split(test, config.output_dir, "test")
    
    # 統計
    print(f"\n" + "=" * 50)
    print("分割結果")
    print("=" * 50)
    print(f"Train: {len(train):>10,} ({len(train)/len(pairs)*100:.1f}%)")
    print(f"Val:   {len(val):>10,} ({len(val)/len(pairs)*100:.1f}%)")
    print(f"Test:  {len(test):>10,} ({len(test)/len(pairs)*100:.1f}%)")
    print(f"Total: {len(pairs):>10,}")
    
    print("\n完了！")


if __name__ == "__main__":
    main()
