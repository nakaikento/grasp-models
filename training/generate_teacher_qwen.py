#!/usr/bin/env python3
"""
Qwen2.5-7B + vLLM を使用した教師データ生成スクリプト

使い方:
1. vLLMサーバー起動:
   python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-7B-Instruct --port 8000

2. 教師データ生成:
   python generate_teacher_qwen.py \
     --src_lang ko --tgt_lang ja \
     --src_file data/raw/source.ko \
     --output_file data/teacher/train.ja
"""

import os
import re
import json
import time
import argparse
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# vLLM APIエンドポイント
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 言語名マッピング
LANG_NAMES = {
    "ja": "Japanese",
    "ko": "Korean"
}

# 言語検出用パターン
JP_PATTERN = re.compile(r'[ぁ-んァ-ヶ一-龠]')
KO_PATTERN = re.compile(r'[가-힣]')

def contains_language(text, lang_code):
    """指定された言語が含まれているかチェック"""
    if lang_code == "ja":
        return bool(JP_PATTERN.search(text))
    elif lang_code == "ko":
        return bool(KO_PATTERN.search(text))
    return False

def create_prompt(src_text, src_lang, tgt_lang):
    """翻訳プロンプトを生成"""
    src_name = LANG_NAMES[src_lang]
    tgt_name = LANG_NAMES[tgt_lang]
    
    return f"""Translate the following {src_name} text to {tgt_name}. Output ONLY the translation, nothing else.

{src_name}: {src_text}

{tgt_name}:"""

def translate_single(src_text, src_lang, tgt_lang, timeout=30):
    """単一テキストを翻訳"""
    prompt = create_prompt(src_text, src_lang, tgt_lang)
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.1,
    }
    
    try:
        resp = requests.post(VLLM_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        
        # 改行を除去
        result = result.replace("\n", " ").strip()
        
        # ソース言語が残っている場合は失敗
        if contains_language(result, src_lang):
            return "FAILED_TRANSLATION"
        
        return result
    except Exception as e:
        return f"ERROR: {e}"

def translate_batch_parallel(batch, src_lang, tgt_lang, max_workers=8):
    """並列でバッチ翻訳"""
    results = [""] * len(batch)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(translate_single, text, src_lang, tgt_lang): i 
            for i, text in enumerate(batch)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"ERROR: {e}"
    
    return results

def check_vllm_server():
    """vLLMサーバーの起動確認"""
    try:
        resp = requests.get("http://localhost:8000/health", timeout=5)
        return resp.status_code == 200
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B + vLLMを使用した教師データ生成")
    
    # 言語設定
    parser.add_argument("--src_lang", type=str, required=True, 
                        choices=["ja", "ko"],
                        help="ソース言語 (ja: 日本語, ko: 韓国語)")
    parser.add_argument("--tgt_lang", type=str, required=True,
                        choices=["ja", "ko"],
                        help="ターゲット言語 (ja: 日本語, ko: 韓国語)")
    
    # ファイルパス
    parser.add_argument("--src_file", type=str, required=True,
                        help="入力ファイルパス")
    parser.add_argument("--output_file", type=str, required=True,
                        help="出力ファイルパス")
    
    # 処理設定
    parser.add_argument("--batch_size", type=int, default=32,
                        help="バッチサイズ（並列リクエスト数）")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="並列ワーカー数")
    parser.add_argument("--sample_interval", type=int, default=1000,
                        help="進捗表示の間隔")
    parser.add_argument("--limit", type=int, default=None,
                        help="処理行数の制限（デバッグ用）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Qwen2.5-7B 教師データ生成")
    print("=" * 60)
    print(f"  ソース言語: {args.src_lang} ({LANG_NAMES[args.src_lang]})")
    print(f"  ターゲット言語: {args.tgt_lang} ({LANG_NAMES[args.tgt_lang]})")
    print(f"  入力ファイル: {args.src_file}")
    print(f"  出力ファイル: {args.output_file}")
    print(f"  バッチサイズ: {args.batch_size}")
    print(f"  並列ワーカー: {args.max_workers}")
    print()
    
    # vLLMサーバー確認
    print("🔍 vLLMサーバー確認中...")
    if not check_vllm_server():
        print("❌ vLLMサーバーが起動していません。")
        print()
        print("以下のコマンドでサーバーを起動してください:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("    --model Qwen/Qwen2.5-7B-Instruct --port 8000 \\")
        print("    --gpu-memory-utilization 0.9")
        return 1
    print("✅ vLLMサーバー接続OK")
    print()
    
    # 入力データ読み込み
    print(f"📖 入力ファイル読み込み中: {args.src_file}")
    if not os.path.exists(args.src_file):
        raise FileNotFoundError(f"入力ファイルが見つかりません: {args.src_file}")
    
    with open(args.src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]
    
    if args.limit:
        src_lines = src_lines[:args.limit]
    
    print(f"✅ {len(src_lines):,}行読み込み完了")
    
    # 再開処理
    start_idx = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            start_idx = sum(1 for _ in f)
        print(f"🔄 {start_idx:,}行目から再開します...")
    else:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 統計
    total_processed = start_idx
    total_failed = 0
    start_time = time.time()
    
    # 翻訳実行
    print(f"\n🔥 翻訳開始...")
    with open(args.output_file, 'a', encoding='utf-8') as f:
        pbar = tqdm(
            range(start_idx, len(src_lines), args.batch_size),
            initial=start_idx // args.batch_size,
            total=len(src_lines) // args.batch_size,
            desc="翻訳中"
        )
        
        for i in pbar:
            batch = src_lines[i : i + args.batch_size]
            
            # 空行対策
            batch = [line if line else "。" for line in batch]
            
            # 並列翻訳
            results = translate_batch_parallel(
                batch, args.src_lang, args.tgt_lang, args.max_workers
            )
            
            # 統計更新
            for res in results:
                if res.startswith("FAILED") or res.startswith("ERROR"):
                    total_failed += 1
            total_processed += len(results)
            
            # サンプル表示
            if i % args.sample_interval < args.batch_size:
                elapsed = time.time() - start_time
                speed = total_processed / elapsed if elapsed > 0 else 0
                eta = (len(src_lines) - total_processed) / speed if speed > 0 else 0
                
                print(f"\n--- [進捗: {total_processed:,}/{len(src_lines):,}] ---")
                print(f"原文 ({args.src_lang}): {batch[0]}")
                print(f"翻訳 ({args.tgt_lang}): {results[0]}")
                print(f"速度: {speed:.1f}行/秒, 失敗: {total_failed:,}, ETA: {eta/60:.1f}分")
                print("-" * 50)
            
            # 結果を書き込み
            for res in results:
                f.write(res + "\n")
            f.flush()
            
            # プログレスバー更新
            pbar.set_postfix({
                "speed": f"{total_processed / (time.time() - start_time):.1f}/s",
                "failed": total_failed
            })
    
    # 完了レポート
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("✅ 翻訳完了")
    print("=" * 60)
    print(f"  総処理行数: {total_processed:,}")
    print(f"  失敗数: {total_failed:,} ({100*total_failed/total_processed:.1f}%)")
    print(f"  所要時間: {elapsed/60:.1f}分")
    print(f"  平均速度: {total_processed/elapsed:.1f}行/秒")
    print(f"  出力ファイル: {args.output_file}")
    print()
    
    return 0

if __name__ == "__main__":
    exit(main())
