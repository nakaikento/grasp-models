# mt-ja-ko

日本語→韓国語のオンデバイス翻訳モデル。字幕・アニメ・オタク文化に特化した軽量高速エンジンを目指します。

## 背景

Helsinki-NLP (Hugging Face) に日韓翻訳モデルが存在しなかったため、自作することにしました。リアルタイム字幕翻訳アプリ [Grasp](https://github.com/user/grasp) のコア翻訳エンジンとして使用予定です。

## 目標

| 項目 | スペック |
|------|----------|
| モデルサイズ | 100MB 〜 300MB |
| 推論速度 | CPU: 50-100ms / GPU: 15ms以下 |
| 翻訳の質 | 字幕特有の意訳・口語表現に強い |
| 運用コスト | ローカル実行（0円） |

ASR（音声認識）と組み合わせて、**合計0.3秒以内**の表示を目指します。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   [OPUS OpenSubtitles]                                  │
│          │                                              │
│          ▼                                              │
│   [Data Cleaning] ─── HTMLタグ・ノイズ除去              │
│          │                                              │
│          ▼                                              │
│   [SentencePiece] ─── 日韓最適化Vocab作成               │
│          │                                              │
│          ▼                                              │
│   [Teacher Model] ─── M2M100 / NLLB-200                 │
│          │                                              │
│          │  Knowledge Distillation                      │
│          ▼                                              │
│   [Student Model] ─── MarianMT (Transformer-Base)       │
│          │                                              │
│          ▼                                              │
│   [ONNX Export] ─── 推論最適化                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### なぜ MarianMT か

- **高速推論**: 翻訳タスクに特化した軽量Transformerアーキテクチャ
- **実績**: Helsinki-NLPの多言語モデルで広く採用
- **ONNX互換**: モバイル/エッジデバイスへのデプロイが容易

### Knowledge Distillation

巨大LLM（M2M100等）を「教師」として、その高品質な出力を「生徒（MarianMT）」に学習させることで、軽量ながらも自然な翻訳を実現します。

## ロードマップ

### 第1段階：データ準備とクレンジング `← 現在ここ`

- [ ] OPUS (OpenSubtitles) から日韓対訳ペアを取得
- [ ] 前処理スクリプト作成（HTMLタグ・音楽記号・クレジット除去）
- [ ] SentencePieceによる語彙辞書（Vocab）作成
- [ ] データ品質の検証

### 第2段階：モデル学習

- [ ] AWS GPUインスタンス（g4dn.xlarge等）環境構築
- [ ] 教師モデルによる擬似正解データ生成
- [ ] Back-translationスコアによるフィルタリング（上位95%抽出）
- [ ] MarianMT学習実行
- [ ] 評価（BLEU, COMET等）

### 第3段階：推論最適化と実装

- [ ] ONNX形式への変換
- [ ] 量子化（INT8）による高速化
- [ ] Androidアプリ（Grasp）への統合
- [ ] ストリーミング翻訳の実装

### 将来の拡張

- [ ] LoRAによるジャンル特化（K-POP、ゲーム実況、アニメ等）

## ディレクトリ構成

```
mt-ja-ko/
├── data/
│   ├── raw/              # 生データ
│   ├── cleaned/          # クレンジング済みデータ
│   └── tokenized/        # トークナイズ済みデータ
├── scripts/
│   ├── download.py       # データダウンロード
│   ├── clean.py          # 前処理
│   └── tokenize.py       # SentencePiece学習
├── training/
│   ├── distillation.py   # 知識蒸留
│   └── train.py          # モデル学習
├── export/
│   └── to_onnx.py        # ONNX変換
├── models/               # 学習済みモデル
└── README.md
```

## 必要要件

```
Python >= 3.10
PyTorch >= 2.0
transformers >= 4.30
sentencepiece >= 0.1.99
onnx >= 1.14
onnxruntime >= 1.15
```

## 参考

- [OPUS OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles.php)
- [MarianMT](https://huggingface.co/docs/transformers/model_doc/marian)
- [Helsinki-NLP Models](https://huggingface.co/Helsinki-NLP)
- [Knowledge Distillation for NMT](https://arxiv.org/abs/1606.07947)

## ライセンス

MIT License