#!/usr/bin/env python3
"""
FLORES-200から日韓並列コーパスを抽出。
高品質な人手翻訳で、評価用リファレンスとして使用。

FLORES-200:
- 200言語の並列コーパス
- devtest: 1012文
- 全言語で同じ内容（n-way並列）
"""

import argparse
from pathlib import Path
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="FLORES-200から日韓ペア抽出")
    parser.add_argument("--output-dir", type=Path, default=Path("data/flores"))
    parser.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    args = parser.parse_args()
    
    print("📥 Loading FLORES-200...")
    ds = load_dataset("facebook/flores", "all", split=args.split, trust_remote_code=True)
    
    print(f"   Total samples: {len(ds)}")
    
    # 日本語と韓国語を抽出
    ja_texts = ds["sentence_jpn_Jpan"]
    ko_texts = ds["sentence_kor_Hang"]
    
    # 保存
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    ja_file = args.output_dir / "ja_source.txt"
    ko_file = args.output_dir / "ko_reference.txt"
    
    with open(ja_file, 'w', encoding='utf-8') as f:
        for text in ja_texts:
            f.write(text.strip() + '\n')
    
    with open(ko_file, 'w', encoding='utf-8') as f:
        for text in ko_texts:
            f.write(text.strip() + '\n')
    
    # 統計
    ja_lengths = [len(t) for t in ja_texts]
    ko_lengths = [len(t) for t in ko_texts]
    
    stats_file = args.output_dir / "stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: FLORES-200 ({args.split})\n")
        f.write(f"サンプル数: {len(ja_texts)}\n")
        f.write(f"日本語平均長: {sum(ja_lengths)/len(ja_lengths):.1f}文字\n")
        f.write(f"日本語最小/最大: {min(ja_lengths)}/{max(ja_lengths)}文字\n")
        f.write(f"韓国語平均長: {sum(ko_lengths)/len(ko_lengths):.1f}文字\n")
        f.write(f"韓国語最小/最大: {min(ko_lengths)}/{max(ko_lengths)}文字\n")
    
    print(f"\n✅ Saved to {args.output_dir}")
    print(f"   - {ja_file.name}: {len(ja_texts)} lines")
    print(f"   - {ko_file.name}: {len(ko_texts)} lines")
    print(f"\n📊 Sample:")
    for i in range(3):
        print(f"   [{i}] JA: {ja_texts[i][:50]}...")
        print(f"       KO: {ko_texts[i][:50]}...")

if __name__ == "__main__":
    main()
