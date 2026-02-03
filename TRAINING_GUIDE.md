# 🚀 RunPod学習ガイド

MarianMT翻訳モデルのRunPod学習手順

## 前提条件

- RunPod GPU Pod（RTX 4090推奨）
- このリポジトリがクローン済み
- データが準備済み（`data/splits/`, `data/tokenized/spm.model`）

## 📋 クイックスタート

### 1. RunPodにログイン

```bash
# SSH接続
ssh root@<pod-ip> -p <port>

# リポジトリクローン
git clone https://github.com/nakaikento/mt-ja-ko.git
cd mt-ja-ko
```

### 2. 依存関係インストール

```bash
pip install -r requirements.txt
```

### 3. 学習実行

#### **韓国語 → 日本語**（教師データあり）

```bash
python training/train_pair.py \
  --src-lang ko \
  --tgt-lang ja \
  --use-teacher \
  --epochs 10 \
  --batch-size 64
```

#### **韓国語 → 日本語**（教師データ自動生成）

```bash
python training/train_pair.py \
  --src-lang ko \
  --tgt-lang ja \
  --generate-teacher \
  --epochs 10 \
  --batch-size 64
```

#### **日本語 → 韓国語**（既存データで確認）

```bash
python training/train_pair.py \
  --src-lang ja \
  --tgt-lang ko \
  --use-teacher \
  --epochs 10 \
  --batch-size 64
```

## 🔧 オプション

### 基本オプション

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `--src-lang` | ソース言語（ja/ko） | **必須** |
| `--tgt-lang` | ターゲット言語（ja/ko） | **必須** |
| `--epochs` | エポック数 | 10 |
| `--batch-size` | バッチサイズ | 64 |
| `--learning-rate` | 学習率 | 3e-4 |

### データオプション

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `--use-teacher` | 教師データを使用 | True |
| `--no-teacher` | OPUS生データで学習 | False |
| `--generate-teacher` | 教師データを自動生成 | False |
| `--data-dir` | データディレクトリ | `data/splits` |
| `--teacher-dir` | 教師データディレクトリ | `data/teacher` |
| `--tokenizer` | トークナイザーパス | `data/tokenized/spm.model` |

### その他

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `--resume` | チェックポイントから再開 | なし |
| `--num-workers` | DataLoaderワーカー数 | 4 |

## 📊 進捗確認

スクリプト実行中、以下の情報がリアルタイムで表示されます：

```
Training: 42%|████████▎         | 8401/20000 [12:45<17:32, 11.03step/s, loss=1.2345, BLEU=28.50]
```

- **進捗バー**: 全ステップ中の現在位置
- **loss**: 現在の損失値（低いほど良い）
- **BLEU**: 評価セットのBLEUスコア（高いほど良い、目標: >30）

## 🔄 チェックポイント＆再開

### 中断した学習を再開

```bash
# 最新のチェックポイントを確認
ls -lt models/ko-ja/

# 再開（例: checkpoint-8000）
python training/train_pair.py \
  --src-lang ko \
  --tgt-lang ja \
  --resume models/ko-ja/checkpoint-8000
```

### 定期保存

- デフォルトで1000ステップごとに保存
- 最新3つのチェックポイントのみ保持（ディスク節約）
- Early Stopping: 3回連続でBLEUが改善しなければ停止

## 📁 出力構成

```
models/
  ko-ja/                          # 韓日翻訳モデル
    checkpoint-1000/              # 中間チェックポイント
    checkpoint-2000/
    checkpoint-8000/              # 最新
    config.json                   # モデル設定
    pytorch_model.bin             # 学習済み重み
    spm.model                     # トークナイザー
    training_args.bin             # 学習設定
  ja-ko/                          # 日韓翻訳モデル（既存）
```

## 🎯 目標BLEU

| 言語ペア | 目標BLEU | 達成基準 |
|---------|---------|---------|
| ja → ko | 30+ | Grasp v1.0.0で達成済み |
| ko → ja | 30+ | 今回の目標 |

## ⚡ パフォーマンス最適化

### RTX 4090（24GB VRAM）

```bash
python training/train_pair.py \
  --src-lang ko \
  --tgt-lang ja \
  --batch-size 64 \
  --num-workers 8
```

### RTX 3090 / A5000（16GB VRAM）

```bash
python training/train_pair.py \
  --src-lang ko \
  --tgt-lang ja \
  --batch-size 32 \
  --num-workers 4
```

### Google Colab（T4 16GB）

```bash
python training/train_pair.py \
  --src-lang ko \
  --tgt-lang ja \
  --batch-size 32 \
  --num-workers 2
```

## 🐛 トラブルシューティング

### CUDA Out of Memory

```bash
# バッチサイズを減らす
--batch-size 32

# または
--batch-size 16
```

### 教師データが見つからない

```bash
# 自動生成
python training/train_pair.py \
  --src-lang ko \
  --tgt-lang ja \
  --generate-teacher
```

### 中断時の対処

1. Ctrl+Cで安全に停止（最新チェックポイントまで保存済み）
2. `--resume` で再開

## 📝 次のステップ

学習完了後：

1. **ONNX変換**
   ```bash
   python training/convert_to_onnx.py --model-dir models/ko-ja
   ```

2. **量子化**
   ```bash
   python training/quantize_onnx.py --model-dir models/ko-ja
   ```

3. **GitHub Releaseにアップロード**
   ```bash
   # ko-ja-onnx.zip を作成
   cd models/ko-ja
   zip -r ../../ko-ja-onnx.zip encoder_model.onnx decoder_model_merged.onnx spm.model
   
   # Grasp リポジトリでリリース作成
   gh release create v2.0.0 ko-ja-onnx.zip --title "v2.0.0 - 双方向翻訳" --notes "韓日翻訳モデル追加"
   ```

## 💡 Tips

- **wandb無効化済み**: ログはローカルのみ（`report_to=["none"]`）
- **tqdm進捗バー**: リアルタイム表示でCLI実行が快適
- **エラーハンドリング**: Ctrl+Cやエラー時も安全に停止
- **教師データ自動生成**: `--generate-teacher` で1コマンド完結

---

**質問・問題があれば**: Kentoに連絡 👋
