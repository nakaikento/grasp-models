#!/usr/bin/env python3
"""
RunPodでLLM翻訳評価を一括実行するスクリプト。

使用方法:
  # vLLMサーバーが起動済みの状態で
  python3 run_full_evaluation.py --base-url http://localhost:8000/v1
  
  # OpenRouter経由
  OPENROUTER_API_KEY=xxx python3 run_full_evaluation.py --provider openrouter
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

# ===== 設定 =====
MODELS = [
    # (名前, プロバイダーでのモデル名)
    ("qwen3-32b", "Qwen/Qwen3-32B"),
]

STRATEGIES = ["baseline", "natural", "few_shot"]

# プロンプト戦略
PROMPT_STRATEGIES = {
    "baseline": {
        "system": "あなたは韓国語から日本語への翻訳者です。",
        "user": "次の韓国語を日本語に翻訳してください。翻訳のみを出力してください。\n\n{text}"
    },
    "natural": {
        "system": """あなたは韓国語から日本語への翻訳者です。
自然で流暢な日本語に翻訳してください。
韓国語特有の表現は日本語として自然な言い回しに置き換えてください。
アニメやドラマのセリフのような口語表現を意識してください。""",
        "user": "次の韓国語を自然な日本語に翻訳してください。翻訳のみを出力してください。\n\n{text}"
    },
    "few_shot": {
        "system": "あなたは韓国語から日本語への翻訳者です。以下の例を参考に翻訳してください。",
        "user": "次の韓国語を日本語に翻訳してください。翻訳のみを出力してください。\n\n{text}",
        "examples": [
            ("뭐 하는 거야?", "何してるの？"),
            ("진짜 미치겠다", "本当にどうかしてる"),
            ("괜찮아, 걱정하지 마", "大丈夫、心配しないで"),
        ]
    }
}


def load_lines(path: Path) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def build_messages(text: str, strategy: str) -> list[dict]:
    config = PROMPT_STRATEGIES[strategy]
    messages = [{"role": "system", "content": config["system"]}]
    
    if "examples" in config:
        for ko, ja in config["examples"]:
            messages.append({"role": "user", "content": f"韓国語: {ko}"})
            messages.append({"role": "assistant", "content": ja})
    
    messages.append({"role": "user", "content": config["user"].format(text=text)})
    return messages


def translate_text(client: httpx.Client, base_url: str, model: str, 
                   text: str, strategy: str) -> str:
    messages = build_messages(text, strategy)
    
    resp = client.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.3,
        },
        timeout=60.0
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def translate_batch(texts: list[str], base_url: str, model: str, 
                   strategy: str, max_workers: int = 8) -> list[str]:
    results = [""] * len(texts)
    
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, text in enumerate(texts):
                future = executor.submit(
                    translate_text, client, base_url, model, text, strategy
                )
                futures[future] = i
            
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"  ⚠️ Error at {idx}: {e}")
                    results[idx] = ""
                done += 1
                if done % 100 == 0:
                    print(f"  Progress: {done}/{len(texts)}")
    
    return results


def evaluate_translations(sources: list[str], references: list[str], 
                         hypotheses: list[str]) -> dict:
    metrics = {}
    
    # chrF++
    try:
        from sacrebleu import corpus_chrf
        result = corpus_chrf(hypotheses, [references], word_order=2)
        metrics['chrf_pp'] = round(result.score, 2)
    except Exception as e:
        print(f"  ⚠️ chrF++ error: {e}")
    
    # BLEU
    try:
        from sacrebleu import corpus_bleu
        result = corpus_bleu(hypotheses, [references], tokenize='char')
        metrics['bleu'] = round(result.score, 2)
    except Exception as e:
        print(f"  ⚠️ BLEU error: {e}")
    
    # COMET (GPU required)
    try:
        from comet import download_model, load_from_checkpoint
        
        print("  📊 Loading COMET model...")
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        
        output = model.predict(data, batch_size=16, gpus=1)
        metrics['comet'] = round(output.system_score, 4)
        
    except ImportError:
        print("  ⚠️ COMET not installed")
    except Exception as e:
        print(f"  ⚠️ COMET error: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="LLM翻訳評価一括実行")
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--provider", choices=["vllm", "openrouter"], default="vllm")
    parser.add_argument("--limit", type=int, help="処理行数制限 (デバッグ用)")
    parser.add_argument("--skip-translate", action="store_true", help="翻訳をスキップ（評価のみ）")
    args = parser.parse_args()
    
    # プロバイダー設定
    if args.provider == "openrouter":
        args.base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("❌ OPENROUTER_API_KEY not set")
            sys.exit(1)
    
    # 入力読み込み
    source_file = args.samples_dir / "source_ko.txt"
    reference_file = args.samples_dir / "reference_ja.txt"
    
    sources = load_lines(source_file)
    references = load_lines(reference_file)
    
    if args.limit:
        sources = sources[:args.limit]
        references = references[:args.limit]
    
    print(f"📥 Loaded {len(sources)} samples")
    
    # 出力ディレクトリ
    args.output_dir.mkdir(parents=True, exist_ok=True)
    translations_dir = Path("translations")
    translations_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    for model_name, model_id in MODELS:
        for strategy in STRATEGIES:
            run_name = f"{model_name}-{strategy}"
            trans_file = translations_dir / f"{run_name}.txt"
            
            print(f"\n{'='*60}")
            print(f"🔄 {run_name}")
            print(f"{'='*60}")
            
            # 翻訳実行
            if not args.skip_translate or not trans_file.exists():
                print(f"  🌐 Translating with {model_id}...")
                start = time.time()
                hypotheses = translate_batch(sources, args.base_url, model_id, strategy)
                elapsed = time.time() - start
                print(f"  ⏱️ Translation took {elapsed:.1f}s ({len(sources)/elapsed:.1f} samples/s)")
                
                # 保存
                with open(trans_file, 'w', encoding='utf-8') as f:
                    for h in hypotheses:
                        f.write(h + '\n')
                print(f"  💾 Saved to {trans_file}")
            else:
                print(f"  📂 Loading existing translations from {trans_file}")
                hypotheses = load_lines(trans_file)
            
            # 評価
            print(f"  📊 Evaluating...")
            metrics = evaluate_translations(sources, references, hypotheses)
            metrics['model'] = model_name
            metrics['strategy'] = strategy
            metrics['n_samples'] = len(sources)
            
            all_results.append(metrics)
            
            print(f"  📈 Results:")
            print(f"     chrF++: {metrics.get('chrf_pp', 'N/A')}")
            print(f"     BLEU:   {metrics.get('bleu', 'N/A')}")
            print(f"     COMET:  {metrics.get('comet', 'N/A')}")
    
    # 結果まとめ
    print(f"\n{'='*70}")
    print("📊 FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Strategy':<12} {'chrF++':>10} {'BLEU':>10} {'COMET':>10}")
    print("-"*70)
    
    sorted_results = sorted(all_results, key=lambda x: x.get('chrf_pp', 0), reverse=True)
    for i, r in enumerate(sorted_results):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{rank} {r['model']:<17} {r['strategy']:<12} "
              f"{r.get('chrf_pp', 'N/A'):>10} {r.get('bleu', 'N/A'):>10} "
              f"{r.get('comet', 'N/A'):>10}")
    
    # JSON保存
    result_file = args.output_dir / "comparison.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_samples': len(sources),
            'results': all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to {result_file}")


if __name__ == "__main__":
    main()
