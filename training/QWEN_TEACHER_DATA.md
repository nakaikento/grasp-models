# Qwen2.5-7B 教師データ生成ガイド

## 概要
Qwen2.5-7B-Instruct + vLLM を使用して、高品質な翻訳教師データを生成します。

**性能 (RTX 4090):**
- chrF++: 49.29 (M2M100比 3.3倍)
- 速度: ~6行/秒 (並列処理時)
- 100万行: 約20-25時間

## 1. RunPod セットアップ

### Pod作成
| 項目 | 推奨値 |
|------|--------|
| Template | `runpod/pytorch:2.4.0-py3.11-cuda12.4.0-cudnn-devel-ubuntu22.04` |
| GPU | RTX 4090 (24GB) |
| Container Disk | 100GB |

### 環境構築
```bash
# SSH接続
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519

# vLLMインストール
pip install vllm requests tqdm

# リポジトリクローン
git clone https://github.com/nakaikento/grasp-models.git
cd grasp-models/training
```

## 2. vLLMサーバー起動

```bash
# ターミナル1: サーバー起動
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.9

# 起動確認（別ターミナル）
curl http://localhost:8000/health
```

## 3. 教師データ生成

### 韓国語→日本語
```bash
python generate_teacher_qwen.py \
  --src_lang ko \
  --tgt_lang ja \
  --src_file data/raw/source.ko \
  --output_file data/teacher/train.ja \
  --batch_size 32 \
  --max_workers 16
```

### 日本語→韓国語
```bash
python generate_teacher_qwen.py \
  --src_lang ja \
  --tgt_lang ko \
  --src_file data/raw/source.ja \
  --output_file data/teacher/train.ko \
  --batch_size 32 \
  --max_workers 16
```

## 4. オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--batch_size` | 32 | 並列リクエスト数 |
| `--max_workers` | 16 | スレッド数 |
| `--sample_interval` | 1000 | サンプル表示間隔 |
| `--limit` | None | 処理行数制限（デバッグ用） |

## 5. 中断・再開

スクリプトは自動的に出力ファイルの行数を確認し、中断したところから再開します。

```bash
# 中断した場合、同じコマンドを再実行するだけでOK
python generate_teacher_qwen.py \
  --src_lang ko --tgt_lang ja \
  --src_file data/raw/source.ko \
  --output_file data/teacher/train.ja
```

## 6. 速度見積もり

| 設定 | 速度 | 100万行 |
|------|------|---------|
| batch=32, workers=16 | ~6行/秒 | ~46時間 |
| batch=64, workers=32 | ~10行/秒 | ~28時間 |
| batch=128, workers=64 | ~15行/秒 | ~18時間 |

※ GPUメモリに余裕があれば並列数を増やせます

## 7. トラブルシューティング

### vLLMサーバーに接続できない
```bash
# サーバーが起動しているか確認
ps aux | grep vllm

# ログ確認
tail -f /tmp/vllm.log
```

### メモリ不足
```bash
# gpu-memory-utilizationを下げる
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.8
```

### 失敗が多い場合
- ネットワーク遅延: `--batch_size` を下げる
- タイムアウト: スクリプト内の `timeout` を増やす

## 8. 出力フォーマット

入力ファイルと同じ行数で、1行1翻訳結果。
失敗した行は `FAILED_TRANSLATION` または `ERROR: ...` になります。

```
# 入力 (source.ko)
안녕하세요
감사합니다

# 出力 (train.ja)
こんにちは
ありがとうございます
```
