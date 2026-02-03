#!/usr/bin/env python3
"""
OPUSデータの冒頭確認スクリプト

Usage:
    python scripts/inspect_data.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

JA_FILE = PROJECT_ROOT / "data/raw/OpenSubtitles.ja-ko.ja"
KO_FILE = PROJECT_ROOT / "data/raw/OpenSubtitles.ja-ko.ko"


def inspect(file_path, num_lines=10):
    print(f"--- {file_path.name} の冒頭 {num_lines} 行 ---")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            print(f"{i+1}: {line.strip()}")


def main():
    if not JA_FILE.exists():
        print(f"エラー: {JA_FILE} が見つかりません")
        print("OPUSファイルを data/raw/ に配置してください")
        return
    
    inspect(JA_FILE)
    print()
    inspect(KO_FILE)


if __name__ == "__main__":
    main()