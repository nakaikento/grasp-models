# 🔐 RunPod アクセスガイド

再起動後も作業を継続できるように、アクセス方法とよく使うコマンドをまとめました。

---

## 📡 SSH接続方法

### 秘密鍵の配置
```bash
# ホストマシンでの操作
ls ~/.ssh/id_ed25519
# 秘密鍵が存在することを確認
# ※秘密鍵の内容はこのドキュメントに記載しません
```

### SSH接続コマンド
```bash
# RunPod インスタンスへの接続
ssh root@157.157.221.29 -p 32309 -i ~/.ssh/id_ed25519

# 接続確認
nvidia-smi
pwd
# 期待: /root
```

### プロジェクトディレクトリ
```bash
cd /workspace/mt-ja-ko
```

---

## 🔑 WandB認証

### APIトークンの取得方法
```bash
# ホストマシンでトークンを確認
cat ~/openclaw/wandb.txt
# または
cat ~/.openclaw/workspace/wandb.txt
```

### RunPodでのログイン
```bash
# 手動ログイン
wandb login

# または環境変数で設定
export WANDB_API_KEY=$(cat ~/openclaw/wandb.txt)
wandb login $WANDB_API_KEY
```

**重要:** APIトークン自体はこのドキュメントに記載しません。

---

## 🚀 訓練再開方法

### 現在の訓練状況確認

```bash
# プロセス確認
ps aux | grep 'python3.*train.py' | grep -v grep

# 最新ログ確認
tail -100 /workspace/mt-ja-ko/train_bs112.log

# GPU使用状況
nvidia-smi
```

### 訓練ログからWandB URL取得

```bash
grep 'View run at' /workspace/mt-ja-ko/train_bs112.log
# 出力例: https://wandb.ai/okamoto2okamoto-personal/huggingface/runs/09s29z5s
```

### 訓練実行コマンド（記録用）

#### 現在実行中（2026-02-03）
```bash
cd /workspace/mt-ja-ko

nohup python3 training/train.py \
    --data-dir data/matched \
    --tokenizer data/tokenized/spm.model \
    --output-dir /workspace/models/ja-ko-final \
    --epochs 10 \
    --batch-size 112 \
    --learning-rate 3e-4 \
    --num-workers 12 > train_bs112.log 2>&1 &

# WandB: https://wandb.ai/okamoto2okamoto-personal/huggingface/runs/09s29z5s
# 設定: Batch 112 x 2 = 224, 全24,420ステップ, 推定6.8時間
```

#### 過去に成功した設定（BLEU 33達成時）
```bash
python3 training/train.py \
    --data-dir data/clean \
    --tokenizer data/tokenized/spm.model \
    --output-dir /workspace/models/ja-ko \
    --epochs 10 \
    --batch-size 128 \
    --learning-rate 3e-4 \
    --num-workers 12

# 結果: Test BLEU 33.03
# WandB: https://wandb.ai/okamoto2okamoto-personal/huggingface/runs/zsh840l3
```

---

## 📊 よく使うSSHコマンド集

### 訓練進捗確認
```bash
# 最新100行
tail -100 /workspace/mt-ja-ko/train_bs112.log

# epochとBLEUスコアのみ抽出
grep -E "'epoch':|eval_bleu" /workspace/mt-ja-ko/train_bs112.log | tail -20

# リアルタイム監視
tail -f /workspace/mt-ja-ko/train_bs112.log
```

### GPU/メモリ監視
```bash
# GPU使用状況
nvidia-smi

# リアルタイム監視（1秒ごと更新）
watch -n 1 nvidia-smi

# メモリ使用量
free -h

# ディスク使用量
df -h /workspace
```

### プロセス管理
```bash
# 訓練プロセス確認
ps aux | grep train.py | grep -v grep

# プロセス強制終了（必要な場合のみ）
pkill -9 -f 'python3.*train.py'

# GPUプロセス確認
nvidia-smi --query-compute-apps=pid --format=csv,noheader
```

### ファイル確認
```bash
# データファイル確認
ls -lh /workspace/mt-ja-ko/data/matched/
wc -l /workspace/mt-ja-ko/data/matched/*.ja

# モデルチェックポイント確認
ls -lh /workspace/models/ja-ko-final/checkpoint-*/
du -sh /workspace/models/ja-ko-final/

# 最新のチェックポイントのBLEUスコア確認
find /workspace/models/ja-ko-final -name 'trainer_state.json' -exec tail {} \; | grep eval_bleu
```

### ログ分析
```bash
# 訓練速度（it/s）の推移
grep 'it/s' /workspace/mt-ja-ko/train_bs112.log | tail -50

# エラー確認
grep -i error /workspace/mt-ja-ko/train_bs112.log
grep -i 'out of memory' /workspace/mt-ja-ko/train_bs112.log

# 評価結果一覧
grep 'eval_bleu' /workspace/mt-ja-ko/train_bs112.log
```

---

## 🔧 データとモデルの場所

### データディレクトリ構造
```
/workspace/mt-ja-ko/
├── data/
│   ├── splits/          # 元データ（1,035,749ペア）
│   ├── cleaned/         # クリーニング後（1,025,781ペア）
│   └── matched/         # 訓練用（train: 546,881, val/test: 14,392）
│       ├── train.ja
│       ├── train.ko
│       ├── val.ja
│       ├── val.ko
│       ├── test.ja
│       └── test.ko
└── data/tokenized/
    └── spm.model        # SentencePiece トークナイザー
```

### モデル出力
```
/workspace/models/
├── ja-ko-final/         # 現在訓練中
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   └── ...
└── ja-ko-onnx-int8/     # 以前のONNXモデル
```

---

## 📝 訓練設定メモ

### 現在の訓練（2026-02-03）
- **方向:** 日本語 → 韓国語
- **データ:** OPUS OpenSubtitles（クリーニング済み）
- **モデル:** MarianMT (61M params)
- **設定:**
  - Batch size: 112 per device × 2 accumulation = 224 effective
  - Learning rate: 3e-4
  - Epochs: 10
  - Total steps: 24,420
- **推定時間:** 約6.8時間
- **GPU:** RTX 4090
- **VRAM使用:** 9.5GB / 24.5GB (39%)
- **GPU使用率:** 50%

### 過去の成功事例（BLEU 33達成）
- **日時:** 2026-01-25
- **Batch size:** 128 × 2 = 256
- **データ:** data/clean/ (OPUS韓国語)
- **結果:** Test BLEU 33.03
- **訓練時間:** 約50分（10エポック）

---

## 🐛 トラブルシューティング

### CUDA Out of Memory エラー
```bash
# 現在のプロセスを停止
pkill -9 -f 'python3.*train.py'

# Batch sizeを減らして再開
# 112 → 96 → 80 → 64 と段階的に
nohup python3 training/train.py \
    --batch-size 96 \
    ... > train_bs96.log 2>&1 &
```

### 訓練が止まっている
```bash
# プロセス確認
ps aux | grep train.py

# GPU確認
nvidia-smi

# ログの最後を確認
tail -50 /workspace/mt-ja-ko/train_bs112.log

# 必要なら強制終了して再開
pkill -9 python3
# 再度訓練コマンド実行
```

### SSH接続が切れた
```bash
# 再接続
ssh root@157.157.221.29 -p 32309 -i ~/.ssh/id_ed25519

# 訓練プロセスが生きているか確認
ps aux | grep train.py

# ログで進捗確認
tail -100 /workspace/mt-ja-ko/train_bs112.log
```

### WandBでリアルタイム確認
```bash
# ログからWandB URLを抽出
grep 'View run at' /workspace/mt-ja-ko/train_bs112.log

# ブラウザで開いて進捗確認
# loss, learning_rate, eval_bleu などをモニタリング
```

---

## 📚 参考リンク

- **WandB プロジェクト:** https://wandb.ai/okamoto2okamoto-personal/huggingface
- **実験ログ:** EXPERIMENTS.md
- **訓練ガイド:** TRAINING_GUIDE.md
- **RunPodワークフロー:** RUNPOD_WORKFLOW.md

---

**最終更新:** 2026-02-03  
**作成者:** Sora (OpenClaw)
