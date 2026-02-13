#!/usr/bin/env python3
"""
Qwen2.5-72B-AWQ 教師データ生成スクリプト

vLLM + AWQ量子化で高品質な韓国語→日本語翻訳データを生成する。
約100万サンプルの大規模生成を想定。

使い方:
    # 基本実行（AI Hub全データ）
    python generate_qwen72b_awq.py \
        --input /path/to/aihub \
        --output /path/to/output/teacher_data.jsonl

    # サンプル数指定
    python generate_qwen72b_awq.py \
        --input /path/to/aihub \
        --output output.jsonl \
        --limit 100000

    # 再開（前回の続きから）
    python generate_qwen72b_awq.py \
        --input /path/to/aihub \
        --output output.jsonl \
        --resume

必要環境:
    - NVIDIA GPU (RTX A6000 48GB+ 推奨)
    - vLLM 0.15.0+
    - 環境変数: HF_HOME, XDG_CACHE_HOME を /workspace 等に設定

推定処理時間:
    - RTX A6000 (batch=128): ~12-14 samples/s → 100万件 ≈ 20時間
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("❌ vLLMがインストールされていません")
    print("   pip install vllm")
    exit(1)

from tqdm import tqdm


# === 設定 ===
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-AWQ"

# Few-shot付きプロンプト（評価で最も効果的だったもの）
SYSTEM_PROMPT = """あなたは韓国ドラマ・映画・アニメの字幕翻訳を専門とする翻訳者です。
視聴者が画面を見ながら自然に理解できる字幕を作成してください。

【翻訳方針】
- 韓国語の意味とニュアンスを正確に伝える自然な日本語に翻訳
- 文化的な背景を考慮し、日本人視聴者に違和感なく伝わる表現を使用
- 敬語・タメ口のレベルは原文のキャラクター性に合わせる
- 字幕として読みやすい簡潔な表現を心がける

【厳守事項】
- 翻訳文のみを出力（説明・補足は不要）
- 通貨・単位はそのまま維持（ウォン→円への変換禁止）
- 固有名詞は原音に近いカタカナ表記
- 日本語のみで出力（中国語混入禁止）

【翻訳例】
韓: 경기 당일에는 날씨가 좋네.
日: 競技当日には天気がいいね。

韓: 당신은 무슨 맛 아이스크림을 원하나요?
日: あなたは何味のアイスクリームがほしいですか？

韓: 고객님 책제목을 말씀해 주시면 바로 안내해 드리겠습니다.
日: お客様、本のタイトルをおっしゃっていただければ、すぐにご案内します。"""


def load_source_data(input_path: Path, limit: int = None) -> Generator[dict, None, None]:
    """
    ソースデータを読み込む（ストリーミング）
    
    対応フォーマット:
    - ディレクトリ (ko_reference.txt + ja_source.txt): AI Hub形式
    - .jsonl ファイル: {"ko": "...", "ja": "..."} 形式
    - .txt ファイル: 1行1文（韓国語のみ）
    """
    input_path = Path(input_path)
    count = 0
    
    if input_path.is_dir():
        # AI Hub形式（txt並列ファイル）
        ko_file = input_path / "ko_reference.txt"
        ja_file = input_path / "ja_source.txt"
        
        if ko_file.exists():
            with open(ko_file, "r", encoding="utf-8") as f_ko:
                ja_lines = None
                if ja_file.exists():
                    with open(ja_file, "r", encoding="utf-8") as f_ja:
                        ja_lines = f_ja.readlines()
                
                for i, ko_line in enumerate(f_ko):
                    if limit and count >= limit:
                        return
                    
                    ko_text = ko_line.strip()
                    if not ko_text:
                        continue
                    
                    item = {"ko": ko_text, "idx": count}
                    if ja_lines and i < len(ja_lines):
                        item["ja_ref"] = ja_lines[i].strip()
                    
                    yield item
                    count += 1
    
    elif input_path.suffix == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if limit and count >= limit:
                    return
                
                item = json.loads(line.strip())
                item["idx"] = count
                yield item
                count += 1
    
    elif input_path.suffix == ".txt":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if limit and count >= limit:
                    return
                
                ko_text = line.strip()
                if not ko_text:
                    continue
                
                yield {"ko": ko_text, "idx": count}
                count += 1
    
    else:
        raise ValueError(f"Unsupported input format: {input_path}")


def count_source_lines(input_path: Path) -> int:
    """ソースデータの行数をカウント"""
    input_path = Path(input_path)
    
    if input_path.is_dir():
        ko_file = input_path / "ko_reference.txt"
        if ko_file.exists():
            with open(ko_file, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
    elif input_path.exists():
        with open(input_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    
    return 0


def get_processed_count(output_path: Path) -> int:
    """既に処理済みのサンプル数を取得"""
    if not output_path.exists():
        return 0
    
    with open(output_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def build_prompt(ko_text: str) -> str:
    """翻訳用プロンプトを構築（Qwen chat format）"""
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{ko_text}<|im_end|>\n<|im_start|>assistant\n"


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-72B-AWQ 教師データ生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
    # 基本実行
    python generate_qwen72b_awq.py -i data/aihub -o output/teacher.jsonl
    
    # 10万件のみ生成
    python generate_qwen72b_awq.py -i data/aihub -o output/teacher.jsonl -n 100000
    
    # 中断後の再開
    python generate_qwen72b_awq.py -i data/aihub -o output/teacher.jsonl --resume
"""
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="入力データパス（ディレクトリ or ファイル）")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="出力ファイルパス (.jsonl)")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="最大サンプル数")
    parser.add_argument("--batch-size", "-b", type=int, default=128,
                        help="バッチサイズ（デフォルト: 128）")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="前回の続きから再開")
    parser.add_argument("--model", "-m", type=str, default=MODEL_ID,
                        help=f"モデルID（デフォルト: {MODEL_ID}）")
    parser.add_argument("--gpu-memory", type=float, default=0.9,
                        help="GPU メモリ使用率（デフォルト: 0.9）")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="最大出力トークン数（デフォルト: 256）")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # 出力ディレクトリ作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📚 Qwen2.5-72B-AWQ 教師データ生成")
    print("=" * 60)
    print(f"モデル:       {args.model}")
    print(f"入力:         {input_path}")
    print(f"出力:         {output_path}")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"GPUメモリ:    {args.gpu_memory * 100:.0f}%")
    print()

    # ソースデータ数をカウント
    total_lines = count_source_lines(input_path)
    if args.limit:
        total_lines = min(total_lines, args.limit)
    print(f"📊 ソースデータ: {total_lines:,} 件")

    # 再開モード
    skip_count = 0
    if args.resume:
        skip_count = get_processed_count(output_path)
        if skip_count > 0:
            print(f"⏩ 再開モード: {skip_count:,} 件スキップ")
    
    remaining = total_lines - skip_count
    if remaining <= 0:
        print("✅ 既に完了しています")
        return
    
    print(f"📝 処理対象: {remaining:,} 件")
    print()

    # モデル読み込み
    print("🤖 モデル読み込み中...")
    start_load = time.time()
    
    llm = LLM(
        model=args.model,
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=2048,
        enforce_eager=True,  # ディスク容量問題回避（torch.compile無効化）
    )
    
    load_time = time.time() - start_load
    print(f"   完了 ({load_time:.1f}秒)")
    print()

    # サンプリングパラメータ
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding（一貫性のため）
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "\n\n"],
    )

    # 処理開始
    print("🔄 翻訳生成中...")
    start_time = time.time()
    processed = 0
    
    # 出力ファイルを追記モードで開く
    mode = "a" if args.resume and skip_count > 0 else "w"
    
    with open(output_path, mode, encoding="utf-8") as f_out:
        # データをバッチで処理
        batch = []
        batch_items = []
        
        data_iter = load_source_data(input_path, args.limit)
        
        # 再開時はスキップ
        for _ in range(skip_count):
            next(data_iter, None)
        
        pbar = tqdm(total=remaining, desc="生成中", unit="samples")
        
        for item in data_iter:
            batch.append(build_prompt(item["ko"]))
            batch_items.append(item)
            
            if len(batch) >= args.batch_size:
                # バッチ処理
                outputs = llm.generate(batch, sampling_params)
                
                for item, output in zip(batch_items, outputs):
                    ja_text = output.outputs[0].text.strip()
                    
                    result = {
                        "ko": item["ko"],
                        "ja": ja_text,
                    }
                    if "ja_ref" in item:
                        result["ja_ref"] = item["ja_ref"]
                    
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed += 1
                
                pbar.update(len(batch))
                f_out.flush()  # 定期的にフラッシュ
                
                # 速度表示更新
                elapsed = time.time() - start_time
                speed = processed / elapsed if elapsed > 0 else 0
                eta_seconds = (remaining - processed) / speed if speed > 0 else 0
                pbar.set_postfix({
                    "speed": f"{speed:.1f}/s",
                    "ETA": str(timedelta(seconds=int(eta_seconds)))
                })
                
                batch = []
                batch_items = []
        
        # 残りを処理
        if batch:
            outputs = llm.generate(batch, sampling_params)
            
            for item, output in zip(batch_items, outputs):
                ja_text = output.outputs[0].text.strip()
                
                result = {
                    "ko": item["ko"],
                    "ja": ja_text,
                }
                if "ja_ref" in item:
                    result["ja_ref"] = item["ja_ref"]
                
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed += 1
            
            pbar.update(len(batch))
        
        pbar.close()

    # 結果サマリー
    elapsed = time.time() - start_time
    speed = processed / elapsed if elapsed > 0 else 0
    
    print()
    print("=" * 60)
    print("📊 完了サマリー")
    print("=" * 60)
    print(f"処理件数:     {processed:,} 件")
    print(f"処理時間:     {timedelta(seconds=int(elapsed))}")
    print(f"速度:         {speed:.2f} samples/s")
    print(f"出力ファイル: {output_path}")
    print(f"ファイルサイズ: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 60)

    # 推定時間（100万件の場合）
    if processed > 0:
        est_1m = 1_000_000 / speed / 3600
        print(f"\n💡 100万件の推定処理時間: {est_1m:.1f} 時間")


if __name__ == "__main__":
    main()
