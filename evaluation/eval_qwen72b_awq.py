#!/usr/bin/env python3
"""
Qwen2.5-72B-Instruct-AWQ 翻訳品質評価スクリプト

使い方:
    python eval_qwen72b_awq.py --samples 100

必要環境:
    - NVIDIA GPU (RTX 6000 Blackwell 97GB推奨)
    - vLLM 0.6.0+
    - sacrebleu
"""

import argparse
import json
import time
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("❌ vLLMがインストールされていません")
    print("   pip install vllm")
    exit(1)

try:
    import sacrebleu
except ImportError:
    print("❌ sacrebleuがインストールされていません")
    print("   pip install sacrebleu")
    exit(1)


# === 設定 ===
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-AWQ"
DATA_FILE = Path(__file__).parent / "data" / "aihub"  # AI Hub directory
OUTPUT_FILE = Path(__file__).parent / "results" / "qwen72b_awq_results.json"

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


def load_data(filepath: Path, max_samples: int = 100) -> list[dict]:
    """テストデータを読み込む（jsonl or txt pair対応）"""
    data = []
    
    # Check if it's a directory with txt files (AI Hub format)
    if filepath.is_dir():
        ko_file = filepath / "ko_reference.txt"
        ja_file = filepath / "ja_source.txt"
        if ko_file.exists() and ja_file.exists():
            with open(ko_file, "r", encoding="utf-8") as f_ko, \
                 open(ja_file, "r", encoding="utf-8") as f_ja:
                for i, (ko_line, ja_line) in enumerate(zip(f_ko, f_ja)):
                    if i >= max_samples:
                        break
                    data.append({
                        "ko": ko_line.strip(),
                        "ja_ref": ja_line.strip(),
                    })
            return data
    
    # jsonl format
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            item = json.loads(line.strip())
            data.append({
                "ko": item["ko"],
                "ja_ref": item["ja"],
            })
    return data


def translate_batch(llm: LLM, texts: list[str], sampling_params: SamplingParams) -> list[str]:
    """バッチ翻訳"""
    prompts = []
    for text in texts:
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)
    
    outputs = llm.generate(prompts, sampling_params)
    results = []
    for output in outputs:
        generated = output.outputs[0].text.strip()
        results.append(generated)
    return results


def calculate_metrics(hypotheses: list[str], references: list[str]) -> dict:
    """評価指標を計算"""
    # chrF++
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    
    # BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    
    return {
        "chrf++": round(chrf.score, 2),
        "bleu": round(bleu.score, 2),
        "num_samples": len(hypotheses),
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-72B-AWQ Ko→Ja翻訳評価")
    parser.add_argument("--samples", type=int, default=100, help="評価サンプル数")
    parser.add_argument("--batch-size", type=int, default=10, help="バッチサイズ")
    parser.add_argument("--model", type=str, default=MODEL_ID, help="モデルID")
    parser.add_argument("--data", type=str, default=None, help="データファイルパス")
    args = parser.parse_args()

    data_file = Path(args.data) if args.data else DATA_FILE
    
    print("=" * 60)
    print(f"Qwen2.5-72B-AWQ 翻訳品質評価")
    print("=" * 60)
    print(f"モデル: {args.model}")
    print(f"サンプル数: {args.samples}")
    print(f"データ: {data_file}")
    print()

    # データ読み込み
    print("📂 データ読み込み中...")
    if not data_file.exists():
        print(f"❌ データファイルが見つかりません: {data_file}")
        print("   先にデータを準備してください:")
        print("   python prepare_eval_data.py")
        return
    
    data = load_data(data_file, args.samples)
    print(f"   {len(data)} サンプル読み込み完了")
    print()

    # モデル読み込み
    print("🤖 モデル読み込み中...")
    start_load = time.time()
    llm = LLM(
        model=args.model,
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        enforce_eager=True,  # Skip torch.compile to avoid disk space issues
    )
    load_time = time.time() - start_load
    print(f"   モデル読み込み完了 ({load_time:.1f}秒)")
    print()

    # サンプリングパラメータ
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        max_tokens=256,
        stop=["<|im_end|>", "\n\n"],
    )

    # 翻訳実行
    print("🔄 翻訳中...")
    ko_texts = [d["ko"] for d in data]
    start_translate = time.time()
    
    hypotheses = translate_batch(llm, ko_texts, sampling_params)
    
    translate_time = time.time() - start_translate
    speed = len(data) / translate_time
    print(f"   翻訳完了: {translate_time:.1f}秒 ({speed:.2f} samples/s)")
    print()

    # 評価
    print("📊 評価中...")
    references = [d["ja_ref"] for d in data]
    metrics = calculate_metrics(hypotheses, references)
    print(f"   chrF++: {metrics['chrf++']}")
    print(f"   BLEU:   {metrics['bleu']}")
    print()

    # サンプル表示
    print("📝 サンプル翻訳 (最初の5件):")
    print("-" * 60)
    for i in range(min(5, len(data))):
        print(f"[{i+1}]")
        print(f"  KO:  {data[i]['ko']}")
        print(f"  REF: {data[i]['ja_ref']}")
        print(f"  HYP: {hypotheses[i]}")
        print()

    # 代表サンプル3文を選択（短文・中文・長文）
    samples_by_len = sorted(enumerate(data), key=lambda x: len(x[1]["ko"]))
    representative = [
        samples_by_len[len(samples_by_len) // 4][0],      # 短め
        samples_by_len[len(samples_by_len) // 2][0],      # 中間
        samples_by_len[3 * len(samples_by_len) // 4][0],  # 長め
    ]

    # 結果保存
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.model,
        "num_samples": len(data),
        "metrics": metrics,
        "timing": {
            "load_time_sec": round(load_time, 1),
            "translate_time_sec": round(translate_time, 1),
            "speed_samples_per_sec": round(speed, 2),
        },
        "representative_samples": [
            {
                "type": label,
                "ko": data[idx]["ko"],
                "ja_ref": data[idx]["ja_ref"],
                "ja_hyp": hypotheses[idx],
            }
            for label, idx in zip(["short", "medium", "long"], representative)
        ],
        "translations": [
            {
                "ko": data[i]["ko"],
                "ja_ref": data[i]["ja_ref"],
                "ja_hyp": hypotheses[i],
            }
            for i in range(len(data))
        ],
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 結果保存: {OUTPUT_FILE}")

    # サマリー
    print()
    print("=" * 60)
    print("📊 サマリー")
    print("=" * 60)
    print(f"モデル:     {args.model}")
    print(f"サンプル数: {len(data)}")
    print(f"chrF++:     {metrics['chrf++']}")
    print(f"BLEU:       {metrics['bleu']}")
    print(f"処理時間:   {translate_time:.1f}秒")
    print(f"速度:       {speed:.2f} samples/s")
    print()
    print("📝 代表サンプル (短/中/長):")
    print("-" * 60)
    for idx in representative:
        print(f"KO:  {data[idx]['ko']}")
        print(f"REF: {data[idx]['ja_ref']}")
        print(f"HYP: {hypotheses[idx]}")
        print()
    print("=" * 60)


if __name__ == "__main__":
    main()
