#!/usr/bin/env python3
"""
12条件のLLM翻訳を評価するスクリプト。

評価指標:
- chrF++ (参照あり)
- BLEU (参照あり)
- COMET (参照あり、GPU推奨)
- COMET-QE (参照なし、GPU推奨)

使用方法:
  python3 evaluate_all.py \
    --source data/flores/ja_source.txt \
    --reference data/flores/ko_reference.txt \
    --translations-dir translations/ \
    --output results/metrics.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings("ignore")

@dataclass
class Metrics:
    name: str
    n_samples: int
    chrf_pp: Optional[float] = None
    bleu: Optional[float] = None
    comet: Optional[float] = None
    comet_qe: Optional[float] = None


def load_lines(path: Path) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def calculate_surface_metrics(references: list[str], hypotheses: list[str]) -> dict:
    """chrF++とBLEUを計算"""
    metrics = {}
    
    try:
        from sacrebleu import corpus_chrf, corpus_bleu
        
        # chrF++ (word_order=2)
        chrf = corpus_chrf(hypotheses, [references], word_order=2)
        metrics['chrf_pp'] = round(chrf.score, 2)
        
        # BLEU (character-level for Korean)
        bleu = corpus_bleu(hypotheses, [references], tokenize='char')
        metrics['bleu'] = round(bleu.score, 2)
        
    except Exception as e:
        print(f"  ⚠️ Surface metrics error: {e}")
    
    return metrics


def calculate_comet(sources: list[str], references: list[str], 
                   hypotheses: list[str]) -> Optional[float]:
    """COMET (参照あり) を計算"""
    try:
        from comet import download_model, load_from_checkpoint
        
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        
        output = model.predict(data, batch_size=16, gpus=1)
        return round(output.system_score, 4)
        
    except ImportError:
        print("  ⚠️ COMET not installed (pip install unbabel-comet)")
        return None
    except Exception as e:
        print(f"  ⚠️ COMET error: {e}")
        return None


def calculate_comet_qe(sources: list[str], hypotheses: list[str]) -> Optional[float]:
    """COMET-QE (参照なし) を計算"""
    try:
        from comet import download_model, load_from_checkpoint
        
        # QEモデル（参照不要）
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        model = load_from_checkpoint(model_path)
        
        data = [
            {"src": s, "mt": h}
            for s, h in zip(sources, hypotheses)
        ]
        
        output = model.predict(data, batch_size=16, gpus=1)
        return round(output.system_score, 4)
        
    except ImportError:
        print("  ⚠️ COMET-QE not installed")
        return None
    except Exception as e:
        print(f"  ⚠️ COMET-QE error: {e}")
        return None


def paired_bootstrap_test(scores_a: list[float], scores_b: list[float], 
                          n_bootstrap: int = 1000) -> dict:
    """Paired bootstrap resampling で有意差検定"""
    import random
    
    n = len(scores_a)
    assert len(scores_b) == n
    
    diff = sum(a - b for a, b in zip(scores_a, scores_b)) / n
    
    wins_a = 0
    for _ in range(n_bootstrap):
        indices = [random.randint(0, n-1) for _ in range(n)]
        sample_a = [scores_a[i] for i in indices]
        sample_b = [scores_b[i] for i in indices]
        if sum(sample_a) > sum(sample_b):
            wins_a += 1
    
    p_value = 1 - (wins_a / n_bootstrap)
    
    return {
        'mean_diff': round(diff, 4),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05
    }


def main():
    parser = argparse.ArgumentParser(description="LLM翻訳評価")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--translations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--skip-comet", action="store_true", help="COMET計算をスキップ")
    args = parser.parse_args()
    
    # 読み込み
    sources = load_lines(args.source)
    references = load_lines(args.reference)
    
    print(f"📥 Source: {len(sources)} lines")
    print(f"📥 Reference: {len(references)} lines")
    
    # 翻訳ファイルを探す
    trans_files = list(args.translations_dir.glob("*.txt")) + \
                  list(args.translations_dir.glob("*.jsonl"))
    
    if not trans_files:
        print(f"❌ No translation files found in {args.translations_dir}")
        return
    
    print(f"📁 Found {len(trans_files)} translation files")
    
    all_results = []
    
    # COMETモデルを一度だけ読み込み
    comet_model = None
    comet_qe_model = None
    
    if not args.skip_comet:
        try:
            from comet import download_model, load_from_checkpoint
            print("\n📊 Loading COMET models...")
            
            comet_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(comet_path)
            
            comet_qe_path = download_model("Unbabel/wmt22-cometkiwi-da")
            comet_qe_model = load_from_checkpoint(comet_qe_path)
            
            print("   ✅ COMET models loaded")
        except Exception as e:
            print(f"   ⚠️ COMET load failed: {e}")
    
    for trans_file in sorted(trans_files):
        name = trans_file.stem
        print(f"\n{'='*60}")
        print(f"🔄 Evaluating: {name}")
        print(f"{'='*60}")
        
        # 翻訳読み込み
        if trans_file.suffix == '.jsonl':
            import json
            hypotheses = []
            with open(trans_file, 'r') as f:
                for line in f:
                    obj = json.loads(line)
                    hypotheses.append(obj.get('translation', obj.get('hypothesis', '')))
        else:
            hypotheses = load_lines(trans_file)
        
        # 行数を揃える
        min_len = min(len(sources), len(references), len(hypotheses))
        src = sources[:min_len]
        ref = references[:min_len]
        hyp = hypotheses[:min_len]
        
        print(f"   Samples: {min_len}")
        
        # 表層指標
        surface = calculate_surface_metrics(ref, hyp)
        print(f"   chrF++: {surface.get('chrf_pp', 'N/A')}")
        print(f"   BLEU:   {surface.get('bleu', 'N/A')}")
        
        # COMET
        comet_score = None
        comet_qe_score = None
        
        if comet_model:
            data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyp, ref)]
            output = comet_model.predict(data, batch_size=16, gpus=1)
            comet_score = round(output.system_score, 4)
            print(f"   COMET:  {comet_score}")
        
        if comet_qe_model:
            data = [{"src": s, "mt": h} for s, h in zip(src, hyp)]
            output = comet_qe_model.predict(data, batch_size=16, gpus=1)
            comet_qe_score = round(output.system_score, 4)
            print(f"   COMET-QE: {comet_qe_score}")
        
        result = Metrics(
            name=name,
            n_samples=min_len,
            chrf_pp=surface.get('chrf_pp'),
            bleu=surface.get('bleu'),
            comet=comet_score,
            comet_qe=comet_qe_score
        )
        all_results.append(asdict(result))
    
    # 結果まとめ
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON (sorted by chrF++)")
    print(f"{'='*80}")
    
    header = f"{'Model':<30} {'chrF++':>10} {'BLEU':>10} {'COMET':>10} {'COMET-QE':>10}"
    print(header)
    print("-"*80)
    
    sorted_results = sorted(all_results, key=lambda x: x.get('chrf_pp', 0) or 0, reverse=True)
    for i, r in enumerate(sorted_results):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        
        def fmt(v):
            return f"{v:.2f}" if isinstance(v, float) else "N/A"
        
        print(f"{rank} {r['name']:<27} {fmt(r.get('chrf_pp')):>10} "
              f"{fmt(r.get('bleu')):>10} {fmt(r.get('comet')):>10} "
              f"{fmt(r.get('comet_qe')):>10}")
    
    # 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_file': str(args.source),
            'reference_file': str(args.reference),
            'results': all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
