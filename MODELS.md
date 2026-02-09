# Grasp Models

Graspで使用するモデルの一覧と取得方法

## ASR (音声認識)

### Sherpa-ONNX Korean ASR (k2-fsa)
- **種類**: Transducer (Zipformer)
- **言語**: 韓国語 (ko)
- **ソース**: [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **HuggingFace**: [sherpa-onnx-streaming-zipformer-korean-2024-06-16](https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16)
- **モデルファイル**:
  - `encoder-epoch-99-avg-1.int8.onnx` - Encoder
  - `decoder-epoch-99-avg-1.int8.onnx` - Decoder
  - `joiner-epoch-99-avg-1.int8.onnx` - Joiner
  - `tokens.txt` - 韓国語BPEトークン
- **場所**: `grasp-ko-ja/app/src/main/assets/models/`

### SileroVAD
- **種類**: Voice Activity Detection
- **ソース**: [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- **モデルファイル**: `silero_vad.onnx`

## 翻訳

### 韓国語 → 日本語 (ko-ja)
- **種類**: MarianMT (Seq2Seq)
- **量子化**: INT8
- **ソース**: 自作 (grasp-models/training)
- **モデルファイル**:
  - `models/ko-ja-onnx-int8/encoder_model_quantized.onnx`
  - `models/ko-ja-onnx-int8/decoder_model_quantized.onnx`
  - `models/ko-ja-onnx-int8/decoder_with_past_model_quantized.onnx`
  - `models/ko-ja-onnx-int8/spm.model`

### 日本語 → 韓国語 (ja-ko)
- **種類**: MarianMT (Seq2Seq)
- **量子化**: INT8
- **ソース**: 自作 (grasp-models/training)
- **モデルファイル**:
  - `models/ja-ko-onnx-int8/encoder_model_quantized.onnx`
  - `models/ja-ko-onnx-int8/decoder_model_quantized.onnx`
  - `models/ja-ko-onnx-int8/decoder_with_past_model_quantized.onnx`
  - `models/ja-ko-onnx-int8/spm.model`

## 評価結果

### 翻訳モデル (2026-02-09)

| 方向 | chrF++ | BLEU | 備考 |
|------|--------|------|------|
| ko→ja | 7.38 | 1.28 | 短文制限の影響で長文が苦手 |
| ja→ko | TBD | TBD | 未評価 |

**評価データ**: grasp-datasets (YouTube KO+JA手動字幕, 10,519ペア)

## ディレクトリ構造

```
grasp-models/
├── models/
│   ├── ko-ja-onnx-int8/    # 韓→日翻訳
│   └── ja-ko-onnx-int8/    # 日→韓翻訳
├── training/                # 学習スクリプト
├── export/                  # ONNX変換スクリプト
└── evaluation/              # 評価パイプライン

grasp-ko-ja/app/src/main/assets/models/
├── encoder-epoch-99-avg-1.int8.onnx   # ASR
├── decoder-epoch-99-avg-1.int8.onnx
├── joiner-epoch-99-avg-1.int8.onnx
├── bpe.model
├── tokens.txt
├── silero_vad.onnx
├── ko-ja-onnx/              # 翻訳
└── ja-ko/                   # 翻訳
```

## TODO

- [ ] ASRモデルをgrasp-modelsにコピーまたはシンボリックリンク
- [ ] ASR評価パイプライン構築
- [ ] E2E評価（音声→翻訳）実装
