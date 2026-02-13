#!/usr/bin/env python3
"""
AI Hub Ko-Ja Translation データセットから評価用サンプルを抽出。

データセット: traintogpb/aihub-koja-translation-integrated-small-100k
フィルタ:
  - aihub-71263: 放送コンテンツ（ドラマ字幕）
  - aihub-546: 日常生活・口語

使用方法:
  python3 extract_aihub.py --n-samples 10000 --seed 71263
"""

import argparse
import random
from pathlib import Path
from datasets import load_dataset

# 映画・ドラマ・アニメ学習に適したソース
TARGET_SOURCES = {'aihub-71263', 'aihub-546'}

def main():
    parser = argparse.ArgumentParser(description="AI Hubから日韓ペア抽出")
    parser.add_argument("--output-dir", type=Path, default=Path("data/aihub"))
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=71263, help="ランダムシード")
    parser.add_argument("--min-len", type=int, default=5, help="最小文字数")
    parser.add_argument("--max-len", type=int, default=200, help="最大文字数")
    parser.add_argument("--all-sources", action="store_true", help="全ソースを使用")
    args = parser.parse_args()
    
    print("📥 Loading AI Hub Ko-Ja Translation dataset...")
    ds = load_dataset(
        'traintogpb/aihub-koja-translation-integrated-small-100k', 
        split='train'
    )
    print(f"   Total samples: {len(ds)}")
    
    # ソースフィルタ
    if not args.all_sources:
        print(f"🎯 Filtering by source: {TARGET_SOURCES}")
    
    # フィルタリング
    print(f"🔍 Filtering (length: {args.min_len}-{args.max_len})...")
    filtered = []
    source_counts = {}
    
    for item in ds:
        source = item['source']
        
        # ソースフィルタ
        if not args.all_sources and source not in TARGET_SOURCES:
            continue
        
        ja = item['ja'].strip()
        ko = item['ko'].strip()
        
        # 長さチェック
        if not (args.min_len <= len(ja) <= args.max_len):
            continue
        if not (args.min_len <= len(ko) <= args.max_len):
            continue
        
        # 空白チェック
        if not ja or not ko:
            continue
        
        filtered.append((ja, ko, source))
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"   Filtered: {len(filtered)} samples")
    for src, cnt in sorted(source_counts.items()):
        print(f"     {src}: {cnt}")
    
    # ソース比率を維持してサンプリング
    random.seed(args.seed)
    
    if len(filtered) <= args.n_samples:
        samples = filtered
    else:
        # ソースごとに比率を計算
        total = len(filtered)
        samples = []
        
        for source in source_counts:
            source_items = [(ja, ko) for ja, ko, src in filtered if src == source]
            n_take = int(args.n_samples * (len(source_items) / total))
            n_take = max(1, n_take)  # 最低1つは取る
            
            if len(source_items) <= n_take:
                samples.extend(source_items)
            else:
                samples.extend(random.sample(source_items, n_take))
        
        # 不足分を補充
        if len(samples) < args.n_samples:
            all_pairs = [(ja, ko) for ja, ko, _ in filtered]
            remaining = [p for p in all_pairs if p not in samples]
            random.shuffle(remaining)
            samples.extend(remaining[:args.n_samples - len(samples)])
        
        random.shuffle(samples)
        samples = samples[:args.n_samples]
    
    print(f"\n📊 Selected {len(samples)} samples")
    
    # 保存
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    ja_file = args.output_dir / "ja_source.txt"
    ko_file = args.output_dir / "ko_reference.txt"
    
    with open(ja_file, 'w', encoding='utf-8') as f:
        for ja, ko in samples:
            f.write(ja + '\n')
    
    with open(ko_file, 'w', encoding='utf-8') as f:
        for ja, ko in samples:
            f.write(ko + '\n')
    
    # 統計
    ja_lengths = [len(ja) for ja, _ in samples]
    ko_lengths = [len(ko) for _, ko in samples]
    
    stats_file = args.output_dir / "stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: AI Hub Ko-Ja Translation\n")
        f.write(f"Sources: {', '.join(sorted(source_counts.keys()))}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"サンプル数: {len(samples)}\n")
        f.write(f"日本語平均長: {sum(ja_lengths)/len(ja_lengths):.1f}文字\n")
        f.write(f"日本語最小/最大: {min(ja_lengths)}/{max(ja_lengths)}文字\n")
        f.write(f"韓国語平均長: {sum(ko_lengths)/len(ko_lengths):.1f}文字\n")
        f.write(f"韓国語最小/最大: {min(ko_lengths)}/{max(ko_lengths)}文字\n")
    
    print(f"\n✅ Saved to {args.output_dir}")
    print(f"   - {ja_file.name}: {len(samples)} lines")
    print(f"   - {ko_file.name}: {len(samples)} lines")
    
    # サンプル表示
    print(f"\n📝 Sample (first 5):")
    for i in range(min(5, len(samples))):
        ja, ko = samples[i]
        print(f"   [{i}] JA: {ja[:50]}...")
        print(f"       KO: {ko[:50]}...")

if __name__ == "__main__":
    main()
