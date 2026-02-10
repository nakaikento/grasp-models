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

※ v2: 行順序を保証（並列処理でも正しい順序で出力）
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

def translate_single(args_tuple):
    """単一テキストを翻訳（インデックス付き）"""
    global_idx, src_text, src_lang, tgt_lang, timeout = args_tuple
    
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
            return (global_idx, "FAILED_TRANSLATION")
        
        return (global_idx, result)
    except Exception as e:
        return (global_idx, f"ERROR: {e}")

def translate_batch_parallel_ordered(batch_with_indices, src_lang, tgt_lang, max_workers=16, timeout=30):
    """
    並列でバッチ翻訳（順序保証版）
    
    Args:
        batch_with_indices: [(global_idx, text), ...] のリスト
        src_lang: ソース言語
        tgt_lang: ターゲット言語
        max_workers: 並列ワーカー数
        timeout: タイムアウト秒数
    
    Returns:
        [(global_idx, translation), ...] のリスト（ソート済み）
    """
    results = []
    
    # タスクリスト作成
    tasks = [
        (global_idx, text, src_lang, tgt_lang, timeout)
        for global_idx, text in batch_with_indices
    ]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(translate_single, task) for task in tasks]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # このケースは通常発生しないが念のため
                pass
    
    # インデックスでソート
    results.sort(key=lambda x: x[0])
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
    parser.add_argument("--sample_interval", type=int, default=5000,
                        help="進捗表示の間隔")
    parser.add_argument("--limit", type=int, default=None,
                        help="処理行数の制限（デバッグ用）")
    parser.add_argument("--checkpoint_interval", type=int, default=10000,
                        help="チェックポイント保存間隔（行数）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Qwen2.5-7B 教師データ生成 (v2: 順序保証)")
    print("=" * 60)
    print(f"  ソース言語: {args.src_lang} ({LANG_NAMES[args.src_lang]})")
    print(f"  ターゲット言語: {args.tgt_lang} ({LANG_NAMES[args.tgt_lang]})")
    print(f"  入力ファイル: {args.src_file}")
    print(f"  出力ファイル: {args.output_file}")
    print(f"  バッチサイズ: {args.batch_size}")
    print(f"  並列ワーカー: {args.max_workers}")
    print(f"  チェックポイント間隔: {args.checkpoint_interval}")
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
    
    total_lines = len(src_lines)
    if args.limit:
        src_lines = src_lines[:args.limit]
        total_lines = len(src_lines)
    
    print(f"✅ {total_lines:,}行読み込み完了")
    
    # 再開処理: チェックポイントファイルを確認
    checkpoint_file = args.output_file + ".checkpoint"
    start_idx = 0
    all_results = []  # (idx, translation) のリスト
    
    if os.path.exists(checkpoint_file):
        print(f"🔄 チェックポイントファイル発見: {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    idx, translation = int(parts[0]), parts[1]
                    all_results.append((idx, translation))
        
        if all_results:
            start_idx = max(idx for idx, _ in all_results) + 1
            print(f"✅ {len(all_results):,}件ロード、{start_idx:,}行目から再開")
    else:
        os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    
    # 統計
    total_processed = len(all_results)
    total_failed = sum(1 for _, t in all_results if t.startswith("FAILED") or t.startswith("ERROR"))
    start_time = time.time()
    
    # チェックポイントファイルを追記モードで開く
    checkpoint_f = open(checkpoint_file, 'a', encoding='utf-8')
    
    try:
        # 翻訳実行
        print(f"\n🔥 翻訳開始 ({start_idx:,}行目から)...")
        
        pbar = tqdm(
            range(start_idx, total_lines, args.batch_size),
            initial=start_idx // args.batch_size,
            total=(total_lines + args.batch_size - 1) // args.batch_size,
            desc="翻訳中"
        )
        
        for batch_start in pbar:
            batch_end = min(batch_start + args.batch_size, total_lines)
            
            # (global_idx, text) のリストを作成
            batch_with_indices = []
            for i in range(batch_start, batch_end):
                text = src_lines[i] if src_lines[i] else "。"
                batch_with_indices.append((i, text))
            
            # 並列翻訳（順序保証）
            results = translate_batch_parallel_ordered(
                batch_with_indices, args.src_lang, args.tgt_lang, args.max_workers
            )
            
            # 結果を保存
            for idx, translation in results:
                all_results.append((idx, translation))
                # チェックポイントに即座に書き込み
                checkpoint_f.write(f"{idx}\t{translation}\n")
                
                if translation.startswith("FAILED") or translation.startswith("ERROR"):
                    total_failed += 1
            
            checkpoint_f.flush()
            total_processed = len(all_results)
            
            # サンプル表示
            if batch_start % args.sample_interval < args.batch_size:
                elapsed = time.time() - start_time
                processed_this_run = total_processed - (start_idx if start_idx > 0 else 0)
                speed = processed_this_run / elapsed if elapsed > 0 else 0
                remaining = total_lines - total_processed
                eta = remaining / speed if speed > 0 else 0
                
                sample_idx, sample_trans = results[0] if results else (0, "N/A")
                sample_src = src_lines[sample_idx] if sample_idx < len(src_lines) else "N/A"
                
                print(f"\n--- [進捗: {total_processed:,}/{total_lines:,} ({100*total_processed/total_lines:.1f}%)] ---")
                print(f"原文 ({args.src_lang}): {sample_src[:60]}")
                print(f"翻訳 ({args.tgt_lang}): {sample_trans[:60]}")
                print(f"速度: {speed:.1f}行/秒, 失敗: {total_failed:,}, ETA: {eta/60:.1f}分")
                print("-" * 50)
            
            # プログレスバー更新
            elapsed = time.time() - start_time
            processed_this_run = total_processed - start_idx
            speed = processed_this_run / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                "speed": f"{speed:.1f}/s",
                "failed": total_failed,
                "done": f"{total_processed:,}"
            })
    
    finally:
        checkpoint_f.close()
    
    # 最終出力: ソートして書き込み
    print(f"\n📝 最終出力ファイル生成中...")
    all_results.sort(key=lambda x: x[0])
    
    # 欠損チェック
    expected_indices = set(range(total_lines))
    actual_indices = set(idx for idx, _ in all_results)
    missing = expected_indices - actual_indices
    
    if missing:
        print(f"⚠️ 欠損行が{len(missing)}件あります。FAILED_TRANSLATIONで埋めます...")
        for idx in missing:
            all_results.append((idx, "FAILED_TRANSLATION"))
        all_results.sort(key=lambda x: x[0])
    
    # ファイル書き込み
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for idx, translation in all_results:
            f.write(translation + "\n")
    
    # チェックポイントファイル削除
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✅ チェックポイントファイル削除: {checkpoint_file}")
    
    # 完了レポート
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("✅ 翻訳完了")
    print("=" * 60)
    print(f"  総処理行数: {len(all_results):,}")
    print(f"  失敗数: {total_failed:,} ({100*total_failed/len(all_results):.1f}%)")
    print(f"  所要時間: {elapsed/60:.1f}分")
    print(f"  平均速度: {(len(all_results) - start_idx)/elapsed:.1f}行/秒")
    print(f"  出力ファイル: {args.output_file}")
    print()
    
    # 検証
    print("🔍 アラインメント検証...")
    with open(args.output_file, 'r', encoding='utf-8') as f:
        output_lines = f.readlines()
    
    if len(output_lines) == total_lines:
        print(f"✅ 行数一致: {len(output_lines):,} = {total_lines:,}")
    else:
        print(f"❌ 行数不一致: 出力{len(output_lines):,} != 入力{total_lines:,}")
    
    return 0

if __name__ == "__main__":
    exit(main())
