# 🚀 RunPod CLIワークフロー

韓→日翻訳モデルのタスク別実行ガイド

## 📋 タスク一覧

| # | タスク | スクリプト | 所要時間 |
|---|--------|-----------|---------|
| 0 | セットアップ | `scripts/setup.sh` | 2分 |
| 1 | 教師データ生成 | `training/generate_teacher_data.py` | 6-8時間 |
| 2 | 学習 | `training/train_pair.py` | 2-4時間 |
| 3 | ONNX変換 | `scripts/convert_to_onnx.py` | 5分 |
| 4 | 量子化 | `scripts/quantize_onnx.py` | 2分 |
| 5 | ZIP作成＆アップロード | 手動 | 5分 |

---

## 🔧 タスク0: セットアップ

**初回のみ実行**

```bash
cd /workspace/grasp-models
bash scripts/setup.sh
```

**確認項目：**
- ✅ GPU利用可能
- ✅ `data/splits/` にデータ存在
- ✅ `data/tokenized/spm.model` 存在
- ✅ 依存関係インストール完了

---

## 📝 タスク1: 教師データ生成

**目的:** NLLB-200で高品質な日本語翻訳を生成（Knowledge Distillation用）

```bash
python training/generate_teacher_data.py \
  --src-lang ko \
  --tgt-lang ja \
  --src-file data/splits/train.ko \
  --output-file data/teacher/train_ko_ja.ja \
  --batch-size 40 \
  --num-beams 3
```

**オプション調整:**
- `--batch-size 64` - RTX 4090ならより大きく（高速化）
- `--batch-size 32` - VRAM不足時は小さく
- `--num-beams 5` - 品質重視（遅くなる）

**所要時間:**
- RTX 4090: 約6時間（1,025,749文）
- RTX 3090: 約8時間

**確認:**
```bash
wc -l data/teacher/train_ko_ja.ja
# 期待: 1025749 data/teacher/train_ko_ja.ja
```

**中断＆再開:**
- ログに `[XXXX/1025749]` と進捗表示
- 中断時は `--resume` で再開（未実装なので要注意）
- 長時間実行なので `screen` または `tmux` 推奨

---

## 🎓 タスク2: MarianMT学習

**目的:** 軽量＆高速な翻訳モデルを学習

```bash
python training/train_pair.py \
  --src-lang ko \
  --tgt-lang ja \
  --use-teacher \
  --epochs 10 \
  --batch-size 64 \
  --learning-rate 3e-4
```

**進捗表示（tqdm）:**
```
Training: 42%|████████▎         | 8401/20000 [12:45<17:32, 11.03step/s, loss=1.2345, BLEU=28.50]
```

**チェックポイント:**
- 1000ステップごとに保存: `models/ko-ja/checkpoint-XXXX/`
- 最新3つのみ保持（ディスク節約）
- BLEU改善が3回連続で止まればEarly Stopping

**中断＆再開:**
```bash
# 最新チェックポイント確認
ls -lt models/ko-ja/

# 再開（例: checkpoint-8000から）
python training/train_pair.py \
  --src-lang ko \
  --tgt-lang ja \
  --resume models/ko-ja/checkpoint-8000
```

**所要時間:**
- RTX 4090: 約2-3時間（エポック10、バッチ64）
- RTX 3090: 約4時間

**確認:**
```bash
# 最終評価結果
cat models/ko-ja/trainer_state.json | grep eval_bleu
# 期待: BLEU > 30
```

---

## 🔄 タスク3: ONNX変換

**目的:** PyTorchモデル → ONNX（Android用）

```bash
python scripts/convert_to_onnx.py \
  --model-dir models/ko-ja
```

**出力:**
```
models/ko-ja-onnx/
  encoder_model.onnx          (~136 MB)
  decoder_model.onnx          (~223 MB)
  decoder_with_past_model.onnx (~211 MB)
  spm.model                   (~807 KB)
  config.json
  generation_config.json
```

**合計サイズ:** 約570 MB

**所要時間:** 約5分

**確認:**
```bash
ls -lh models/ko-ja-onnx/*.onnx
```

---

## 📦 タスク4: 量子化

**目的:** INT8量子化でサイズ削減（推論速度も向上）

```bash
python scripts/quantize_onnx.py \
  --model-dir models/ko-ja-onnx
```

**効果:**
- サイズ: 570 MB → 約285 MB (50%)
- 精度低下: ほぼなし（BLEU -0.5以内）

**出力:**
```
models/ko-ja-onnx-quantized/
  encoder_model_quantized.onnx
  decoder_model_quantized.onnx
  decoder_with_past_model_quantized.onnx
  spm.model
```

**所要時間:** 約2分

**確認:**
```bash
ls -lh models/ko-ja-onnx-quantized/*.onnx
```

---

## 📤 タスク5: ZIP作成＆GitHub Releaseアップロード

**ZIP作成:**
```bash
cd models/ko-ja-onnx-quantized
zip -r ../../ko-ja-onnx.zip *_quantized.onnx spm.model
cd ../..
ls -lh ko-ja-onnx.zip
```

**GitHub Releaseにアップロード:**
```bash
# Graspリポジトリに移動
cd /path/to/Grasp

# Release作成
gh release create v2.0.0 \
  /workspace/grasp-models/ko-ja-onnx.zip \
  --title "v2.0.0 - 双方向翻訳" \
  --notes "韓→日翻訳モデル追加（BLEU: XX.XX）"
```

---

## 🔁 全体ワークフロー（一気通貫）

```bash
# 0. セットアップ
bash scripts/setup.sh

# 1. 教師データ生成（6-8時間）
python training/generate_teacher_data.py \
  --src-lang ko --tgt-lang ja \
  --src-file data/splits/train.ko \
  --output-file data/teacher/train_ko_ja.ja \
  --batch-size 40 --num-beams 3

# 2. 学習（2-4時間）
python training/train_pair.py \
  --src-lang ko --tgt-lang ja \
  --use-teacher --epochs 10 --batch-size 64

# 3. ONNX変換（5分）
python scripts/convert_to_onnx.py --model-dir models/ko-ja

# 4. 量子化（2分）
python scripts/quantize_onnx.py --model-dir models/ko-ja-onnx

# 5. ZIP作成
cd models/ko-ja-onnx-quantized
zip -r ../../ko-ja-onnx.zip *_quantized.onnx spm.model
cd ../..

echo "✅ 完了！ko-ja-onnx.zip をGitHub Releaseにアップロードしてください"
```

**合計所要時間:** 約8-12時間（教師データ生成がボトルネック）

---

## 💡 Tips

### screen/tmuxを使う（長時間実行）

```bash
# screenセッション開始
screen -S mt-training

# タスク実行
python training/generate_teacher_data.py ...

# デタッチ: Ctrl+A → D
# 再アタッチ: screen -r mt-training
```

### ログを保存

```bash
python training/train_pair.py ... 2>&1 | tee train.log
```

### GPU使用率モニタリング

```bash
# 別ターミナルで
watch -n 1 nvidia-smi
```

### ディスク容量確認

```bash
df -h /workspace
```

---

## 🐛 トラブルシューティング

### CUDA Out of Memory

```bash
# バッチサイズを減らす
--batch-size 32  # または 16
```

### 教師データ生成が遅い

```bash
# ビーム数を減らす（品質はやや下がる）
--num-beams 1  # または 2
```

### チェックポイントから再開できない

```bash
# 最新チェックポイントを指定
python training/train_pair.py --resume models/ko-ja/checkpoint-XXXX
```

---

**質問・問題があれば:** Kentoに連絡 👋
