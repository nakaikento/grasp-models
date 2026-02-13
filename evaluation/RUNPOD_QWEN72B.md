# Qwen2.5-72B-AWQ 評価手順 (RunPod)

## 必要環境
- **GPU**: RTX PRO 6000 Blackwell 97GB（推奨）または A100 80GB
- **VRAM**: ~45GB（AWQ INT4量子化）
- **ディスク**: ~50GB（モデル + 依存関係）

## 1. RunPodインスタンス起動

```bash
# RTX PRO 6000 Blackwell 97GB を選択
# テンプレート: RunPod PyTorch 2.1
# ディスク: 100GB以上
```

## 2. SSH接続

```bash
ssh root@<IP> -p <PORT>
```

## 3. 環境セットアップ

```bash
# リポジトリクローン
cd /workspace
git clone https://github.com/nakaikento/grasp-models.git
cd grasp-models/evaluation

# 依存関係インストール
pip install vllm sacrebleu

# 確認
python -c "from vllm import LLM; print('vLLM OK')"
```

## 4. 評価実行

```bash
cd /workspace/grasp-models/evaluation
python eval_qwen72b_awq.py --samples 100
```

### オプション

```bash
# サンプル数変更
python eval_qwen72b_awq.py --samples 50

# 別のモデル（例: 32B）
python eval_qwen72b_awq.py --model Qwen/Qwen2.5-32B-Instruct-AWQ

# カスタムデータ
python eval_qwen72b_awq.py --data /path/to/data.jsonl
```

## 5. 期待される出力

```
============================================================
📊 サマリー
============================================================
モデル:     Qwen/Qwen2.5-72B-Instruct-AWQ
サンプル数: 100
chrF++:     XX.XX
BLEU:       XX.XX
処理時間:   XX.X秒
速度:       X.XX samples/s
============================================================
```

## 6. 結果ファイル

```
evaluation/results/qwen72b_awq_results.json
```

## 推定時間

| サンプル数 | 推定時間 |
|-----------|---------|
| 100 | 15-25分 |
| 500 | 1-2時間 |
| 1000 | 2-4時間 |

## トラブルシューティング

### OOMエラー
```bash
# gpu_memory_utilizationを下げる
# eval_qwen72b_awq.py の gpu_memory_utilization=0.9 → 0.8
```

### モデルダウンロードが遅い
```bash
# HFキャッシュを永続ボリュームに設定
export HF_HOME=/workspace/cache
```

## 比較用: 過去の結果

| モデル | chrF++ | BLEU | データ |
|--------|--------|------|--------|
| Qwen2.5-7B | 49.29 | - | OpenSubs 20 |
| Qwen2.5-7B | 30.01 | 41.23 | AI Hub 1000 |
| Qwen3-32B | 35.39 | 11.50 | AI Hub 1000 |
| **Qwen2.5-72B-AWQ** | **???** | **???** | **Ko-Ja 100** |
