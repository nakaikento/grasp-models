# AWS 実行ガイド

## 概要

このガイドでは、AWS EC2でMarianMT（日韓翻訳モデル）を学習する手順を説明します。

## 推奨インスタンス

| インスタンス | GPU | メモリ | 料金目安 | 用途 |
|-------------|-----|--------|---------|------|
| g4dn.xlarge | T4 x1 | 16GB | $0.5/h | 開発・小規模学習 |
| g5.xlarge | A10G x1 | 24GB | $1.0/h | 推奨（高速） |
| g5.2xlarge | A10G x1 | 32GB | $1.2/h | 大規模学習 |
| p3.2xlarge | V100 x1 | 16GB | $3.0/h | 高性能 |

**推奨**: g5.xlarge（コストパフォーマンス良好）

**Spot Instance**: 60-70%のコスト削減が可能。中断リスクあり。

---

## セットアップ手順

### 1. EC2インスタンス作成

```
AMI: Deep Learning AMI (Ubuntu 22.04)
インスタンス: g5.xlarge
ストレージ: 100GB以上
セキュリティグループ: SSH (22) を許可
```

### 2. SSH接続

```bash
ssh -i your-key.pem ubuntu@<instance-ip>
```

### 3. リポジトリセットアップ

```bash
git clone https://github.com/nakaikento/mt-ja-ko.git
cd mt-ja-ko
chmod +x scripts/run_aws.sh
./scripts/run_aws.sh setup
```

### 4. データ準備

**方法A: ローカルからアップロード**

```bash
# ローカルで実行
scp -i your-key.pem -r data/ ubuntu@<instance-ip>:~/mt-ja-ko/
```

**方法B: S3経由**

```bash
# AWSインスタンスで実行
export S3_DATA_PATH="s3://your-bucket/mt-ja-ko/data"
./scripts/run_aws.sh data
```

---

## 学習実行

### Phase 1: 教師データ生成

NLLB-200を使って高品質な韓国語翻訳を生成（約6-8時間）

```bash
./scripts/run_aws.sh teacher
```

**出力**: `data/teacher/train.ko`

### Phase 2: MarianMT学習

```bash
./scripts/run_aws.sh train
```

**推定時間**: 
- g4dn.xlarge: 24-48時間
- g5.xlarge: 12-24時間

**モニタリング**:
```bash
# GPU使用率
watch -n 1 nvidia-smi

# 学習ログ
tail -f models/ja-ko/trainer_log.txt

# Wandb（設定済みの場合）
# https://wandb.ai で確認
```

### Phase 3: ONNXエクスポート

```bash
./scripts/run_aws.sh export
```

**出力**:
- `models/ja-ko/encoder.onnx`
- `models/ja-ko/decoder.onnx`
- `models/ja-ko/encoder_int8.onnx`（量子化版）
- `models/ja-ko/decoder_int8.onnx`（量子化版）

---

## 一括実行

全ステップを連続実行:

```bash
./scripts/run_aws.sh all
```

---

## コスト見積もり

| フェーズ | 時間（g5.xlarge） | コスト |
|---------|-----------------|--------|
| 教師データ生成 | 6-8時間 | $6-8 |
| モデル学習 | 12-24時間 | $12-24 |
| エクスポート | 10分 | $0.2 |
| **合計** | **20-32時間** | **$20-32** |

Spot Instanceなら約$8-13で実行可能。

---

## トラブルシューティング

### CUDA out of memory

バッチサイズを下げる:
```bash
python training/train.py --batch-size 16
```

### 学習が遅い

1. gradient_accumulation_stepsを増やしてバッチサイズ削減
2. fp16が有効か確認
3. dataloader_num_workersを調整

### 教師データ生成が中断

途中から再開:
```bash
python training/generate_teacher_data.py --resume-from 500000
```

---

## 結果の取得

### S3にアップロード

```bash
export S3_MODEL_PATH="s3://your-bucket/mt-ja-ko/models"
./scripts/run_aws.sh upload
```

### ローカルにダウンロード

```bash
# ローカルで実行
scp -i your-key.pem -r ubuntu@<instance-ip>:~/mt-ja-ko/models/ ./
```

---

## インスタンス停止

学習完了後は必ずインスタンスを停止:

```bash
# インスタンス内で
sudo shutdown -h now
```

または AWS コンソールから停止。

---

## 次のステップ

1. **評価**: testセットでBLEUスコアを確認
2. **Android統合**: ONNXモデルをGraspアプリに統合
3. **LoRAチューニング**: アニメ・K-POP等のジャンル特化
