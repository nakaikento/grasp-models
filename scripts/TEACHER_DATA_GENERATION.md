# 教師データ大規模生成

Qwen2.5-72B-AWQ を使って韓国語→日本語の翻訳教師データを生成する。

## 概要

- **モデル**: Qwen/Qwen2.5-72B-Instruct-AWQ (INT4量子化, ~40GB VRAM)
- **速度**: ~7 samples/s (RTX A6000)
- **100万件**: 約40時間

## クイックスタート

### 1. RunPod環境セットアップ

```bash
# GPUインスタンス起動 (RTX A6000 48GB+ 推奨)
# Container: 20GB, Volume: 100GB+

# SSH接続
ssh root@<IP> -p <PORT>

# リポジトリ取得
cd /workspace
git clone https://github.com/nakaikento/grasp-models.git
cd grasp-models

# 依存関係
pip install vllm tqdm

# キャッシュディレクトリ設定（重要！）
export HF_HOME=/workspace/cache
export XDG_CACHE_HOME=/workspace/cache
```

### 2. 実行

```bash
# 基本実行（AI Hub 1万件）
python scripts/generate_teacher_data.py \
    --input evaluation/data/aihub \
    --output output/teacher_1m.jsonl \
    --limit 10000

# 全件実行（再開可能）
python scripts/generate_teacher_data.py \
    --input evaluation/data/aihub \
    --output output/teacher_1m.jsonl \
    --resume
```

## オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--input`, `-i` | 入力データパス | 必須 |
| `--output`, `-o` | 出力ファイル (.jsonl) | 必須 |
| `--limit`, `-n` | 最大サンプル数 | 全件 |
| `--batch-size`, `-b` | バッチサイズ | 64 |
| `--resume`, `-r` | 前回の続きから再開 | - |
| `--model`, `-m` | モデルID | Qwen2.5-72B-AWQ |
| `--gpu-memory` | GPUメモリ使用率 | 0.9 |

## 入力フォーマット

### AI Hub形式（ディレクトリ）
```
data/aihub/
├── ko_reference.txt   # 韓国語（1行1文）
└── ja_source.txt      # 日本語参照（オプション）
```

### JSONL形式
```jsonl
{"ko": "안녕하세요", "ja_ref": "こんにちは"}
{"ko": "감사합니다", "ja_ref": "ありがとうございます"}
```

### プレーンテキスト
```
안녕하세요
감사합니다
```

## 出力フォーマット

```jsonl
{"ko": "안녕하세요", "ja": "こんにちは", "ja_ref": "こんにちは"}
{"ko": "감사합니다", "ja": "ありがとうございます", "ja_ref": "ありがとうございます"}
```

- `ko`: 韓国語原文
- `ja`: モデル生成翻訳
- `ja_ref`: 参照翻訳（入力に含まれていた場合）

## 推定処理時間

| サンプル数 | RTX A6000 | A100 80GB* |
|-----------|-----------|------------|
| 10,000 | 24分 | ~15分 |
| 100,000 | 4時間 | ~2.5時間 |
| 1,000,000 | 40時間 | ~25時間 |

*A100は推定値（tensor parallel使用時）

## コスト推定 (RunPod)

| GPU | 料金/h | 100万件 | コスト |
|-----|--------|---------|--------|
| RTX A6000 | $0.79 | 40h | ~$32 |
| A100 80GB | $1.99 | 25h | ~$50 |

## トラブルシューティング

### ディスク容量不足
```bash
# キャッシュを /workspace に設定
export HF_HOME=/workspace/cache
export XDG_CACHE_HOME=/workspace/cache
```

### OOMエラー
```bash
# バッチサイズを下げる
python scripts/generate_teacher_data.py --batch-size 32 ...

# GPUメモリ使用率を下げる
python scripts/generate_teacher_data.py --gpu-memory 0.8 ...
```

### 中断からの再開
```bash
# --resume オプションで続きから
python scripts/generate_teacher_data.py --resume ...
```

## プロンプト

Few-shot例を含む最適化済みプロンプトを使用:

```
あなたは韓国ドラマ・映画・アニメの字幕翻訳を専門とする翻訳者です。
視聴者が画面を見ながら自然に理解できる字幕を作成してください。

【翻訳方針】
- 韓国語の意味とニュアンスを正確に伝える自然な日本語に翻訳
- 文化的な背景を考慮し、日本人視聴者に違和感なく伝わる表現を使用
...

【翻訳例】
韓: 경기 당일에는 날씨가 좋네.
日: 競技当日には天気がいいね。
...
```

## 評価結果

| プロンプト | chrF++ | BLEU |
|------------|--------|------|
| 基本 | 38.43 | 3.2 |
| +字幕コンテキスト | 34.72 | 0.0 |
| **+Few-shot** | **37.95** | **5.16** |
