# RunPod MarianMT学習ガイド (2026-02-15)

## 命名規則
- **データ**: `splits_{教師モデル}_{日付}/`
- **スクリプト**: `train_{学習モデル}_{教師}_{日付}.py`
- **出力モデル**: `models/{学習モデル}-{方向}-{教師}-{日付}/`

## 今回の学習
- **データ**: `data/splits_qwen72b_20260215/`
- **スクリプト**: `scripts/train_marian_qwen72b_20260215.py`
- **出力**: `models/marian-ko-ja-qwen72b-20260215/`

## 改善点（昨日の反省）
- ✅ データ汚染除去（FAILED_TRANSLATION、メタテキスト）
- ✅ chrF++評価追加
- ✅ より頻繁な評価（2000ステップ毎）
- ✅ L4最適化（バッチ64、勾配累積2）

## RunPodセットアップ

### 1. Pod作成
- **GPU**: NVIDIA L4（24GB VRAM）
- **Template**: RunPod PyTorch 2.0
- **Disk**: 50GB

### 2. 環境構築
```bash
# リポジトリクローン
git clone https://github.com/nakaikento/grasp-models.git
cd grasp-models

# 依存インストール
pip install transformers datasets sentencepiece evaluate sacrebleu

# GPU確認
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 3. 学習実行
```bash
# 学習開始
python scripts/train_marian_qwen72b_20260215.py

# 予想時間: L4で約2-3時間（5エポック）
```

### 4. チェックポイント確認
```bash
# 学習中の確認
ls -la models/marian-ko-ja-qwen72b-20260215/

# 途中経過確認
cat models/marian-ko-ja-qwen72b-20260215/trainer_state.json | jq '.best_metric'
```

## 学習完了後

### ONNX変換
```bash
python scripts/convert_to_onnx.py --model-dir models/marian-ko-ja-qwen72b-20260215

# 出力: encoder.onnx, decoder.onnx
```

### INT8量子化
```bash
python scripts/quantize_onnx.py --model-dir models/marian-ko-ja-qwen72b-20260215

# 出力: models/marian-ko-ja-qwen72b-20260215-int8/
```

### GitHubにプッシュ
```bash
git add models/marian-ko-ja-qwen72b-20260215/
git commit -m "MarianMT ko-ja trained with Qwen72B teacher (2026-02-15)"
git push
```

## 期待される結果

| 指標 | 昨日 (2/14) | 今回目標 |
|------|------------|---------|
| Test BLEU | 38.51 | > 40 |
| Test chrF++ | N/A | > 50 |
| データ汚染 | 16,000件 | 0件 |

## トラブルシューティング

### OOM (Out of Memory)
```python
# スクリプトの TRAIN_CONFIG を調整
"per_device_train_batch_size": 32,  # 64 → 32
"gradient_accumulation_steps": 4,   # 2 → 4
```

### 学習が遅い
```bash
# DataLoaderワーカー数を確認
python -c "import os; print(os.cpu_count())"
# 必要に応じてスクリプト内で調整
```
