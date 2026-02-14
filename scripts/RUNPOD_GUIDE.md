# RunPod MarianMT学習ガイド v2

## 改善点（昨日の反省）
- ✅ データ汚染除去（FAILED_TRANSLATION、メタテキスト）
- ✅ chrF++評価追加
- ✅ より頻繁な評価（2000ステップ毎）
- ✅ L4最適化（バッチ64、勾配累積2）

## 事前準備（NUCで実行）

```bash
cd ~/.openclaw/workspace/grasp-models

# 1. クリーニング実行
python scripts/clean_and_prepare.py

# 2. データ確認
wc -l data/splits_v2/*.ko data/splits_v2/*.ja

# 3. コミット&プッシュ
git add data/splits_v2/ scripts/
git commit -m "Add cleaned data v2 and training scripts"
git push
```

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
python scripts/train_v2.py

# 予想時間: L4で約2-3時間（5エポック）
```

### 4. チェックポイント確認
```bash
# 学習中の確認
ls -la models/ko-ja-v2/

# 途中経過確認
tail -f models/ko-ja-v2/trainer_state.json
```

## 学習完了後

### ONNX変換
```bash
python training/convert_to_onnx.py --model-dir models/ko-ja-v2

# 出力: models/ko-ja-v2/encoder.onnx, decoder.onnx
```

### INT8量子化
```bash
python training/quantize_onnx.py --model-dir models/ko-ja-v2

# 出力: models/ko-ja-v2-int8/
```

### GitHubにプッシュ
```bash
git add models/ko-ja-v2/
git commit -m "MarianMT ko-ja v2 trained model"
git push
```

## 期待される結果

| 指標 | v1（昨日） | v2（目標） |
|------|-----------|-----------|
| Test BLEU | 38.51 | > 40 |
| Test chrF++ | - | > 50 |
| データ汚染 | 16,000件 | 0件 |

## トラブルシューティング

### OOM (Out of Memory)
```python
# train_v2.py の TRAIN_CONFIG を調整
"per_device_train_batch_size": 32,  # 64 → 32
"gradient_accumulation_steps": 4,   # 2 → 4
```

### 学習が遅い
```bash
# DataLoaderワーカー数を確認
python -c "import os; print(os.cpu_count())"

# 必要に応じて調整
# train_v2.py: "dataloader_num_workers": 2
```
