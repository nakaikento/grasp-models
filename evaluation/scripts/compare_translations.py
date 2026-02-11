#!/usr/bin/env python3
"""
複数のLLM翻訳結果を比較評価するスクリプト。

使用方法:
  python compare_translations.py \
    --source samples/source_ko.txt \
    --reference samples/reference_ja.txt \
    --translations translations/qwen3-32b-natural.txt translations/deepseek-r1.txt \
    --names "Qwen3-32B" "DeepSeek-R1" \
    --output results/comparison.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from statistics import mean, stdev

@dataclass
class Metrics:
    chrf_pp: Optional[float] = None
    bleu: Optional[float] = None
    bertscore_f1: Optional[float] = None
    comet: Optional[float] = None

def calculate_metrics_single(source: str, reference: str, hypothesis: str) -> Metrics:
    """単一の翻訳ペアに対するメトリクスを計算"""
    metrics = Metrics()
    
    # chrF++
    try:
        from sacrebleu import sentence_chrf
        result = sentence_chrf(hypothesis, [reference], word_order=2)
        metrics.chrf_pp = round(result.score, 2)
    except Exception as e:
        print(f"⚠️ chrF++ error: {e}")
    
    # BLEU
    try:
        from sacrebleu import sentence_bleu
        result = sentence_bleu(hypothesis, [reference], tokenize='char')
        metrics.bleu = round(result.score, 2)
    except Exception as e:
        print(f"⚠️ BLEU error: {e}")
    
    return metrics

def calculate_corpus_metrics(sources: list[str], references: list[str], hypotheses: list[str]) -> dict:
    """コーパス全体に対するメトリクスを計算"""
    metrics = {}
    
    # chrF++ (corpus-level)
    try:
        from sacrebleu import corpus_chrf
        result = corpus_chrf(hypotheses, [references], word_order=2)
        metrics['chrf_pp'] = round(result.score, 2)
    except Exception as e:
        print(f"⚠️ corpus chrF++ error: {e}")
    
    # BLEU (corpus-level)
    try:
        from sacrebleu import corpus_bleu
        result = corpus_bleu(hypotheses, [references], tokenize='char')
        metrics['bleu'] = round(result.score, 2)
    except Exception as e:
        print(f"⚠️ corpus BLEU error: {e}")
    
    # COMET (optional, requires GPU)
    try:
        from comet import load_from_checkpoint, download_model
        
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        
        output = model.predict(data, batch_size=8, gpus=1)
        metrics['comet'] = round(output['system_score'], 4)
        
    except ImportError:
        pass  # COMET is optional
    except Exception as e:
        print(f"⚠️ COMET error: {e}")
    
    return metrics

def load_lines(path: Path) -> list[str]:
    """ファイルから行を読み込む"""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def print_comparison_table(results: list[dict]):
    """比較結果をテーブル形式で表示"""
    print("\n" + "="*70)
    print("📊 TRANSLATION QUALITY COMPARISON")
    print("="*70)
    
    # ヘッダー
    header = f"{'Model':<25} {'chrF++':>10} {'BLEU':>10}"
    if any(r.get('comet') for r in results):
        header += f" {'COMET':>10}"
    print(header)
    print("-"*70)
    
    # ソート（chrF++降順）
    sorted_results = sorted(results, key=lambda x: x.get('chrf_pp', 0), reverse=True)
    
    for i, r in enumerate(sorted_results):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        row = f"{rank} {r['name']:<22} {r.get('chrf_pp', 'N/A'):>10}"
        row += f" {r.get('bleu', 'N/A'):>10}"
        if any(x.get('comet') for x in results):
            row += f" {r.get('comet', 'N/A'):>10}"
        print(row)
    
    print("="*70)
    
    # 分析
    print("\n🔍 ANALYSIS")
    best = sorted_results[0]
    print(f"   Best model: {best['name']} (chrF++ {best.get('chrf_pp', 'N/A')})")
    
    chrf_scores = [r['chrf_pp'] for r in results if r.get('chrf_pp')]
    if len(chrf_scores) > 1:
        print(f"   chrF++ range: {min(chrf_scores):.1f} - {max(chrf_scores):.1f} (Δ{max(chrf_scores)-min(chrf_scores):.1f})")
    
    # 品質判定
    if best.get('chrf_pp', 0) >= 50:
        print("   ✅ Top model meets production quality threshold (chrF++ ≥ 50)")
    else:
        print("   ⚠️ No model meets production quality threshold (chrF++ < 50)")

def main():
    parser = argparse.ArgumentParser(description="LLM翻訳比較評価")
    parser.add_argument("--source", type=Path, required=True, help="ソース韓国語ファイル")
    parser.add_argument("--reference", type=Path, required=True, help="リファレンス日本語ファイル")
    parser.add_argument("--translations", type=Path, nargs='+', required=True, help="翻訳結果ファイル群")
    parser.add_argument("--names", type=str, nargs='+', help="モデル名（翻訳ファイルと同数）")
    parser.add_argument("--output", type=Path, help="結果出力ファイル (JSON)")
    args = parser.parse_args()
    
    # 読み込み
    sources = load_lines(args.source)
    references = load_lines(args.reference)
    
    print(f"📥 Source: {len(sources)} lines")
    print(f"📥 Reference: {len(references)} lines")
    
    # 各翻訳結果を評価
    all_results = []
    
    for i, trans_path in enumerate(args.translations):
        name = args.names[i] if args.names and i < len(args.names) else trans_path.stem
        print(f"\n🔄 Evaluating: {name}")
        
        hypotheses = load_lines(trans_path)
        
        # 行数チェック
        min_len = min(len(sources), len(references), len(hypotheses))
        if min_len < len(sources):
            print(f"   ⚠️ Truncating to {min_len} lines")
        
        src = sources[:min_len]
        ref = references[:min_len]
        hyp = hypotheses[:min_len]
        
        # コーパスレベル評価
        metrics = calculate_corpus_metrics(src, ref, hyp)
        metrics['name'] = name
        metrics['file'] = str(trans_path)
        metrics['n_samples'] = min_len
        
        all_results.append(metrics)
        print(f"   chrF++: {metrics.get('chrf_pp', 'N/A')}, BLEU: {metrics.get('bleu', 'N/A')}")
    
    # 比較表示
    print_comparison_table(all_results)
    
    # 結果保存
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                'source_file': str(args.source),
                'reference_file': str(args.reference),
                'n_samples': len(sources),
                'results': all_results
            }, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Results saved: {args.output}")

if __name__ == "__main__":
    main()
