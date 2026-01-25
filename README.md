# mt-ja-ko

日本語→韓国語のオンデバイス翻訳モデル。字幕・ドラマ・アニメに特化した軽量高速エンジン。

## 背景

Helsinki-NLP (Hugging Face) に日韓翻訳モデルが存在しなかったため、自作しました。

## モデル性能

| 指標 | 値 |
|------|-----|
| **Test BLEU** | 33.03 |
| Final Loss | 0.97 |
| パラメータ数 | 61M |

※ BLEU 30-40 は商用翻訳サービス（Google翻訳等）に匹敵するレベル

## モデルサイズ

| バージョン | サイズ | 用途 |
|------------|--------|------|
| PyTorch (fp32) | ~245MB | 学習・評価 |
| ONNX (fp32) | 572MB | 推論 |
| **ONNX (INT8量子化)** | **148MB** | **Android推奨** |

### ONNX INT8 内訳

| ファイル | サイズ |
|----------|--------|
| encoder_model_quantized.onnx | 35MB |
| decoder_model_quantized.onnx | 57MB |
| decoder_with_past_model_quantized.onnx | 54MB |
| spm.model | 807KB |

## 学習設定

| 項目 | 値 |
|------|-----|
| ベースアーキテクチャ | MarianMT (Transformer) |
| Encoder/Decoder Layers | 6 / 6 |
| d_model | 512 |
| Attention Heads | 8 |
| Vocab Size | 32,000 (SentencePiece) |
| 学習データ | OPUS OpenSubtitles (~55万ペア) |
| GPU | NVIDIA RTX 4090 |
| バッチサイズ | 128 |
| エポック | 10 |
| 学習時間 | 約50分 |

## 翻訳例

```
🇯🇵 こんにちは、元気ですか？
🇰🇷 안녕하세요?

🇯🇵 逃げろ！
🇰🇷 도망쳐!

🇯🇵 君のことが好きだ。
🇰🇷 너에 대해 좋아해

🇯🇵 絶対に諦めない。
🇰🇷 절대 포기하지 않을거야.

🇯🇵 君がいないと生きていけない。
🇰🇷 네가 없으면 살아갈 수 없어.
```

## 使用方法

### Python (ONNX Runtime)

```python
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import sentencepiece as spm

# モデルロード
model = ORTModelForSeq2SeqLM.from_pretrained("models/ja-ko-onnx-int8")

# トークナイザー
sp = spm.SentencePieceProcessor()
sp.load("models/ja-ko-onnx-int8/spm.model")

# 翻訳
text = "こんにちは"
inputs = sp.encode(text, out_type=int)
# ... generate ...
```

### Android (sherpa-onnx)

ONNX Runtime + NNAPI で Tensor G2/G3 のNPUを活用可能。

## プロジェクト構成

```
mt-ja-ko/
├── training/
│   ├── train.py          # 学習スクリプト
│   ├── config.py         # モデル・学習設定
│   └── generate_teacher_data.py  # Knowledge Distillation用
├── scripts/
│   └── train_tokenizer.py
├── export/
│   └── to_onnx.py
├── data/
│   ├── tokenized/        # SentencePieceモデル
│   └── splits/           # train/val/test分割
└── models/               # 学習済みモデル（gitignore）
```

## 今後の改善点

- [ ] 否定表現（〜ないで）の精度向上
- [ ] 敬語/タメ口の一貫性
- [ ] LoRAによるジャンル特化（K-POP、ゲーム等）

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) - Non-commercial use only.

- ✅ 個人利用・学習目的 OK
- ✅ 改変・再配布 OK（クレジット表記必要）
- ❌ 商用利用 不可

Note: 学習にNLLB-200を使用しているため、同ライセンスに準拠。
