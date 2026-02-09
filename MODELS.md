# Grasp Models

Graspで使用する全モデルの一覧と性能評価結果

## モデル一覧

| # | 種類 | 方向 | モデル | ソース |
|---|------|------|--------|--------|
| 1 | ASR | 日本語 | Sherpa-ONNX Zipformer | k2-fsa (サードパーティ) |
| 2 | ASR | 韓国語 | Sherpa-ONNX Zipformer | k2-fsa (サードパーティ) |
| 3 | 翻訳 | 日→韓 | MarianMT INT8 | 自作 |
| 4 | 翻訳 | 韓→日 | MarianMT INT8 | 自作 |

---

## 1. 日本語ASR

- **種類**: Transducer (Zipformer)
- **ソース**: [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **場所**: `grasp-ja-ko/app/src/main/assets/models/`
- **ファイル**:
  - `encoder-epoch-99-avg-1.int8.onnx`
  - `decoder-epoch-99-avg-1.int8.onnx`
  - `joiner-epoch-99-avg-1.int8.onnx`
  - `tokens.txt` (日本語トークン)
  - `silero_vad.onnx` (VAD)

### 性能評価

| 指標 | スコア | 評価データ | 日付 |
|------|--------|-----------|------|
| CER | TBD | - | - |
| WER | TBD | - | - |

---

## 2. 韓国語ASR

- **種類**: Transducer (Zipformer)
- **ソース**: [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **HuggingFace**: [sherpa-onnx-streaming-zipformer-korean-2024-06-16](https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16)
- **場所**: `grasp-ko-ja/app/src/main/assets/models/`
- **ファイル**:
  - `encoder-epoch-99-avg-1.int8.onnx`
  - `decoder-epoch-99-avg-1.int8.onnx`
  - `joiner-epoch-99-avg-1.int8.onnx`
  - `tokens.txt` (韓国語BPEトークン)
  - `silero_vad.onnx` (VAD)

### 性能評価

| 指標 | スコア | 評価データ | 日付 |
|------|--------|-----------|------|
| CER | TBD | grasp-datasets (YouTube KO字幕) | - |
| WER | TBD | - | - |

---

## 3. 日本語→韓国語 翻訳

- **種類**: MarianMT (Seq2Seq)
- **量子化**: INT8
- **ソース**: 自作 (`grasp-models/training`)
- **場所**: `grasp-models/models/ja-ko-onnx-int8/`
- **ファイル**:
  - `encoder_model_quantized.onnx`
  - `decoder_model_quantized.onnx`
  - `decoder_with_past_model_quantized.onnx`
  - `spm.model`

### 性能評価

| 指標 | スコア | 評価データ | 日付 |
|------|--------|-----------|------|
| chrF++ | TBD | grasp-datasets (10,519ペア) | - |
| BLEU | TBD | grasp-datasets (10,519ペア) | - |

---

## 4. 韓国語→日本語 翻訳

- **種類**: MarianMT (Seq2Seq)
- **量子化**: INT8
- **ソース**: 自作 (`grasp-models/training`)
- **場所**: `grasp-models/models/ko-ja-onnx-int8/`
- **ファイル**:
  - `encoder_model_quantized.onnx`
  - `decoder_model_quantized.onnx`
  - `decoder_with_past_model_quantized.onnx`
  - `spm.model`

### 性能評価

| 指標 | スコア | 評価データ | 日付 |
|------|--------|-----------|------|
| chrF++ | **7.38** | grasp-datasets (10,519ペア) | 2026-02-09 |
| BLEU | **1.28** | grasp-datasets (10,519ペア) | 2026-02-09 |

**備考**: 短文制限で学習したため長文翻訳が苦手。改善が必要。

---

## 評価データセット

### grasp-datasets
- **リポジトリ**: [nakaikento/grasp-datasets](https://github.com/nakaikento/grasp-datasets)
- **内容**: YouTube KO+JA手動字幕
- **規模**: 19動画、10.5時間、10,519対訳ペア
- **用途**: ASR評価、翻訳評価、E2E評価

---

## 目標スコア

| 評価 | 指標 | 目標 | 現状 |
|------|------|------|------|
| ASR | CER | < 15% | TBD |
| 翻訳 | chrF++ | > 50 | 7.38 |
| 翻訳 | BLEU | > 30 | 1.28 |

---

## 更新履歴

- **2026-02-09**: 初版作成、ko→ja翻訳評価実施 (chrF++ 7.38)
