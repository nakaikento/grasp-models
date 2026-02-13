#!/usr/bin/env python3
"""
LLM翻訳スクリプト（12条件対応）。

条件:
- モデル: Qwen3-32B, Qwen3-235B-A22B, DeepSeek-R1-Distill-32B
- プロンプト: zero_shot, few_shot, thinking, natural

使用方法:
  # 単一条件
  python3 translate_with_llm.py \
    --input data/flores/ja_source.txt \
    --output translations/qwen3-32b-natural.txt \
    --model qwen3-32b \
    --strategy natural \
    --base-url http://localhost:8000/v1

  # 全12条件一括（vLLM使用時）
  python3 translate_with_llm.py --run-all --base-url http://localhost:8000/v1
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

# ===== モデル設定 =====
MODELS = {
    "qwen3-32b": {
        "vllm": "Qwen/Qwen3-32B",
        "openrouter": "qwen/qwen3-32b"
    },
    "qwen3-235b": {
        "openrouter": "qwen/qwen3-235b-a22b"
    },
    "deepseek-r1": {
        "vllm": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "openrouter": "deepseek/deepseek-r1-distill-qwen-32b"
    }
}

# ===== プロンプト戦略 =====
STRATEGIES = {
    "zero_shot": {
        "system": "あなたは日本語から韓国語への翻訳者です。",
        "user": "次の日本語を韓国語に翻訳してください。翻訳のみを出力してください。\n\n{text}",
        "examples": None,
        "thinking": False
    },
    "few_shot": {
        "system": "あなたは日本語から韓国語への翻訳者です。以下の例を参考に翻訳してください。",
        "user": "次の日本語を韓国語に翻訳してください。翻訳のみを出力してください。\n\n{text}",
        "examples": [
            ("今日は天気がいいですね。", "오늘 날씨가 좋네요."),
            ("ちょっと待ってください。", "잠깐만 기다려 주세요."),
            ("本当にありがとうございます。", "정말 감사합니다."),
        ],
        "thinking": False
    },
    "thinking": {
        "system": """あなたは日本語から韓国語への翻訳者です。
翻訳する前に、以下を考慮してください：
1. 文脈と話者の意図
2. 日本語特有の表現や文化的背景
3. 韓国語として自然な言い回し

考えた後、最終的な翻訳のみを出力してください。""",
        "user": "次の日本語を韓国語に翻訳してください。\n\n{text}",
        "examples": None,
        "thinking": True
    },
    "natural": {
        "system": """あなたは日本語から韓国語への翻訳者です。
自然で流暢な韓国語に翻訳してください。
直訳を避け、韓国語として自然な表現を使ってください。
敬語レベルは原文に合わせてください。""",
        "user": "次の日本語を自然な韓国語に翻訳してください。翻訳のみを出力してください。\n\n{text}",
        "examples": None,
        "thinking": False
    }
}


def build_messages(text: str, strategy: dict) -> list[dict]:
    """プロンプト戦略に基づいてメッセージを構築"""
    messages = [{"role": "system", "content": strategy["system"]}]
    
    if strategy["examples"]:
        for ja, ko in strategy["examples"]:
            messages.append({"role": "user", "content": f"日本語: {ja}"})
            messages.append({"role": "assistant", "content": ko})
    
    messages.append({"role": "user", "content": strategy["user"].format(text=text)})
    return messages


def translate_single(client: httpx.Client, base_url: str, api_key: Optional[str],
                    model: str, text: str, strategy: dict) -> str:
    """単一テキストを翻訳"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    messages = build_messages(text, strategy)
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.3,
    }
    
    resp = client.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60.0
    )
    resp.raise_for_status()
    
    content = resp.json()["choices"][0]["message"]["content"]
    
    # Thinking タグを除去（DeepSeek-R1等）
    if "<think>" in content:
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    return content.strip()


def translate_batch(texts: list[str], base_url: str, api_key: Optional[str],
                   model: str, strategy: dict, max_workers: int = 8) -> list[str]:
    """バッチ翻訳"""
    results = [""] * len(texts)
    
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, text in enumerate(texts):
                future = executor.submit(
                    translate_single, client, base_url, api_key, model, text, strategy
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


def run_single_condition(input_file: Path, output_file: Path,
                        model_key: str, strategy_key: str,
                        base_url: str, provider: str,
                        api_key: Optional[str] = None,
                        limit: Optional[int] = None):
    """単一条件を実行"""
    # モデルID取得
    model_config = MODELS.get(model_key, {})
    model_id = model_config.get(provider) or model_config.get("vllm") or model_key
    
    # 戦略取得
    strategy = STRATEGIES.get(strategy_key, STRATEGIES["zero_shot"])
    
    # 入力読み込み
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]
    
    if limit:
        texts = texts[:limit]
    
    print(f"\n{'='*60}")
    print(f"🔄 {model_key} + {strategy_key}")
    print(f"   Model ID: {model_id}")
    print(f"   Samples: {len(texts)}")
    print(f"{'='*60}")
    
    start = time.time()
    results = translate_batch(texts, base_url, api_key, model_id, strategy)
    elapsed = time.time() - start
    
    print(f"   ⏱️ Took {elapsed:.1f}s ({len(texts)/elapsed:.1f} samples/s)")
    
    # 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(r + '\n')
    
    print(f"   💾 Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="LLM翻訳")
    parser.add_argument("--input", type=Path, default=Path("data/flores/ja_source.txt"))
    parser.add_argument("--output", type=Path, help="出力ファイル（単一条件時）")
    parser.add_argument("--output-dir", type=Path, default=Path("translations"))
    parser.add_argument("--model", choices=list(MODELS.keys()), help="モデル（単一条件時）")
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), help="戦略（単一条件時）")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--provider", choices=["vllm", "openrouter"], default="vllm")
    parser.add_argument("--limit", type=int, help="処理行数制限")
    parser.add_argument("--run-all", action="store_true", help="全12条件を実行")
    parser.add_argument("--models", nargs='+', help="実行するモデル（run-all時）")
    parser.add_argument("--strategies", nargs='+', help="実行する戦略（run-all時）")
    args = parser.parse_args()
    
    # APIキー
    api_key = None
    if args.provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("❌ OPENROUTER_API_KEY not set")
            return
        args.base_url = "https://openrouter.ai/api/v1"
    
    if args.run_all:
        # 全条件実行
        models = args.models or list(MODELS.keys())
        strategies = args.strategies or list(STRATEGIES.keys())
        
        print(f"📋 Running {len(models)} models × {len(strategies)} strategies = {len(models)*len(strategies)} conditions")
        
        for model_key in models:
            for strategy_key in strategies:
                output_file = args.output_dir / f"{model_key}-{strategy_key}.txt"
                
                # すでに存在する場合はスキップ
                if output_file.exists():
                    print(f"⏭️ Skipping {model_key}-{strategy_key} (already exists)")
                    continue
                
                run_single_condition(
                    args.input, output_file,
                    model_key, strategy_key,
                    args.base_url, args.provider, api_key,
                    args.limit
                )
    else:
        # 単一条件実行
        if not args.model or not args.strategy:
            parser.error("--model and --strategy required (or use --run-all)")
        
        output_file = args.output or (args.output_dir / f"{args.model}-{args.strategy}.txt")
        
        run_single_condition(
            args.input, output_file,
            args.model, args.strategy,
            args.base_url, args.provider, api_key,
            args.limit
        )


if __name__ == "__main__":
    main()
