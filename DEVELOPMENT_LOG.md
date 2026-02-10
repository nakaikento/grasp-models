# Grasp Models - 開発ログ

Korean↔Japanese リアルタイム翻訳アプリ開発の記録

---

## 📅 2026-02-08: 翻訳精度問題の特定とTED評価パイプライン構築

### 🎯 目標
- ASRと翻訳、どちらに精度問題があるか特定する
- 自動評価パイプラインを構築する

### 🔧 実施内容

#### 1. 問題の分析
**状況:**
- 翻訳速度は良好（46-111ms、平均70ms）✅
- しかし翻訳精度がPrime Video正規字幕と比べて満足できないレベル

**課題:**
- ASR（音声認識）の問題？
- 翻訳モデルの問題？
- どちらか、または両方かを特定する必要がある

#### 2. 評価パイプラインの構築

**データソース: TED Talks**
- 理由:
  - 韓国語スピーカーのトークあり
  - プロ翻訳の字幕（韓国語・日本語）
  - 音声品質が高い
  - CCライセンス、ダウンロード可能

**パイプライン構成:**
```
evaluation/
├── download_ted.py           # TED Talkダウンロード（音声+字幕）
├── vtt_to_text.py           # VTT字幕→プレーンテキスト変換
├── evaluate_asr.py          # ASR精度評価（WER/CER）
├── evaluate_translation.py  # 翻訳精度評価（BLEU/chrF）
├── evaluate_e2e.py          # E2E評価（音声→翻訳）
├── requirements.txt         # 依存パッケージ
└── README.md               # 使い方ガイド
```

**評価メトリクス:**
| 指標 | 何を測る | 目標値 |
|------|---------|--------|
| **WER** (Word Error Rate) | ASR単語誤り率 | < 20% |
| **CER** (Character Error Rate) | ASR文字誤り率 | < 15% |
| **BLEU** | 翻訳品質（n-gram一致） | > 30 |
| **chrF** | 翻訳品質（文字n-gram） | > 50 |

#### 3. テストデータのダウンロード

**動画:**
- タイトル: 'HOW'가 아닌 'WHY'로 'Smart'하게 공부하기
- スピーカー: Sungpyo Hong | TEDxCheongdam
- URL: https://www.youtube.com/watch?v=-WVhcG0rauI

**ダウンロード結果:**
```
✅ 音声: data/audio/-WVhcG0rauI.wav (176MB)
✅ 韓国語字幕: data/subtitles/-WVhcG0rauI.ko.vtt (96KB)
⏳ 日本語字幕: YouTubeレート制限（30分後に再試行）
```

### 📊 次のステップ

1. **日本語字幕取得** (14:19頃にcronリトライ)
2. **評価実行:**
   ```bash
   # ASR評価
   python3 evaluate_asr.py \
     data/audio/-WVhcG0rauI.wav \
     data/subtitles/-WVhcG0rauI.ko.vtt \
     -o data/results/asr_results.json
   
   # 翻訳評価
   python3 evaluate_translation.py \
     data/subtitles/-WVhcG0rauI.ko.vtt \
     data/subtitles/-WVhcG0rauI.ja.vtt \
     -o data/results/translation_results.json
   
   # E2E評価
   python3 evaluate_e2e.py \
     data/audio/-WVhcG0rauI.wav \
     data/subtitles/-WVhcG0rauI.ja.vtt \
     --reference-ko data/subtitles/-WVhcG0rauI.ko.vtt \
     -o data/results/e2e_results.json
   ```

3. **結果分析:**
   - WER/CERが高い → ASRモデル改善が必要
   - BLEUが低い → 翻訳モデル改善が必要
   - 両方問題 → 両方改善

4. **改善サイクル:**
   - ボトルネック特定 → モデル改善 → 再評価 → デプロイ

---

## 📅 2026-02-08: ASRパイプラインのロギング強化

### 🎯 目標
翻訳遅延の原因特定（ASR vs 翻訳 vs バッファリング）

### 🔧 実施内容

**coderエージェントに委譲して実装:**
- OnnxAsrManager.kt - パイプライン全体ログ
- SileroVad.kt - VAD詳細ログ
- TransducerDecoder.kt - ASR推論詳細ログ
- MelSpectrogram.kt - Mel変換ログ
- AudioCaptureService.kt - 音声キャプチャログ
- MainActivity.kt - E2Eパイプラインログ

**ログ出力例:**
```
📊 END-TO-END PIPELINE SUMMARY
   Buffer accumulation: 4003ms  ← 最大の遅延要因
   Processing (total):   590ms
     ├─ WAV読込:     4ms
     ├─ VAD:        65ms
     ├─ ASR推論:   439ms
     │   ├─ Mel:     31ms
     │   ├─ Encoder: 225ms
     │   └─ Decoder: 183ms
     └─ 翻訳:       56ms ✅ 超高速！
   E2E (trigger→done): 4593ms (~4.6秒)
```

### 📊 結果

**ボトルネック特定:**
- ✅ 翻訳は全く問題なし（46-111ms）
- ✅ ASR推論も高速（520ms、RTF 0.11x）
- 📌 遅延の主要因はバッファ待ち時間（設計仕様: 5秒バッファ）

**最適化の余地:**
- バッファサイズ 5秒→3秒 で遅延短縮可能（精度とのトレードオフ）

---

## 📅 2026-02-08: 翻訳モデル精度問題の解決

### 🐛 問題
- テスト環境（Python）: 良好な翻訳
- 実機（Android）: 空文字列、"?"、意味不明な出力

### 🔍 根本原因

#### 1. decoder_start_token が間違っていた
```python
❌ decoder_start_token = PAD_ID (0)
✅ decoder_start_token = BOS_ID (2)  # MarianMTモデルの仕様
```

#### 2. デコーディング手法の違い
- テスト環境: ビームサーチ（num_beams=4）
- 実機: Greedy decoding

### 🔧 解決プロセス

1. **Pythonスクリプトで問題切り分け** (`test_beam_search.py`)
   ```python
   for token_id in [PAD_ID, BOS_ID, EOS_ID]:
       result = translate(text, decoder_start_token=token_id)
       print(f"Token {token_id}: {result}")
   ```
   → BOS_IDで正しく翻訳できることを確認（30秒で問題特定）

2. **速度最適化**
   - ビームサーチ（num_beams=4）: 26秒 ❌
   - Greedy decoding: 30-50ms ✅

3. **品質確認**
   - Greedyでも短文・中文では実用的な品質
   - 速度優先でGreedy採用

### 📝 学んだこと

**ベストプラクティス:**
- Seq2Seqモデルの特殊トークンは**モデルアーキテクチャ依存**
- ビームサーチは品質向上するが、モバイルでは**遅すぎる**
- 実装前にPythonで検証すると**劇的に効率化**（30min実機ビルド → 30sec Python検証）

**Androidデバッグの教訓:**
1. 複雑な問題は**Pythonスクリプトで先に実験**
2. 解決策を確認してからAndroidに実装
3. 実機テストは最後の1回で成功させる

### 🚀 成果

両アプリ（grasp-ja-ko, grasp-ko-ja）で:
- ✅ decoder_start_token修正済み
- ✅ Greedy decoding実装
- ✅ 翻訳速度: 30-50ms（リアルタイム対応）
- ✅ Pixel 7aで動作確認済み

---

## 🗂️ リポジトリ構成

```
grasp-models/          # モデル開発・テスト（このリポジトリ）
├── models/
│   ├── ja-ko-onnx-int8/  # 日→韓翻訳（145MB）
│   └── ko-ja-onnx-int8/  # 韓→日翻訳（145MB）
├── test_beam_search.py   # デコーダーテスト
├── evaluation/           # 評価パイプライン（NEW）
│   ├── download_ted.py
│   ├── evaluate_*.py
│   └── data/
└── DEVELOPMENT_LOG.md   # このファイル

grasp-ja-ko/           # 日本語→韓国語アプリ
└── app/src/main/assets/models/

grasp-ko-ja/           # 韓国語→日本語アプリ
└── app/src/main/assets/models/
```

---

## 📊 パフォーマンスサマリー

### 翻訳速度（実機計測）
| Component | Speed | Status |
|-----------|-------|--------|
| 翻訳（ONNX） | 46-111ms (avg 70ms) | ✅ 目標達成 |
| ASR推論 | 439ms (RTF 0.11x) | ✅ 高速 |
| バッファ待機 | ~4秒 | 📌 設計仕様 |

### 翻訳精度（要評価）
- TED Talks評価パイプライン構築完了
- 次回セッションで実測予定

---

## 🎯 今後の改善計画

### 短期（1週間）
1. ✅ 評価パイプライン完成
2. ⏳ TED Talksで精度測定
3. ⏳ ボトルネック特定（ASR vs 翻訳）
4. ⏳ 優先度付けして改善

### 中期（1ヶ月）
1. 精度改善（WER < 20%, BLEU > 30 目標）
2. バッファサイズ最適化
3. 複数のTED Talksで統計的評価
4. ユーザーフィードバック収集

### 長期（3ヶ月）
1. モデル軽量化（INT8 → INT4検討）
2. 他言語ペア対応（英語↔日本語など）
3. リアルタイム会話対応
4. Google Play公開

---

## 📚 参考リンク

- **MarianMT**: https://huggingface.co/Helsinki-NLP/opus-mt-ko-ja
- **sherpa-onnx**: https://github.com/k2-fsa/sherpa-onnx
- **TED Talks**: https://www.ted.com/
- **BLEU/chrF**: https://github.com/mjpost/sacrebleu
- **WER/CER**: https://github.com/jitsi/jiwer

---

## 🤝 開発体制

- **Main**: Kento (ユーザー)
- **Sora**: AI開発アシスタント
- **Coder**: Claude Opus 4.5（複雑な実装タスク）

**委譲ルール:**
- 簡単な編集・確認 → Sora
- 複雑な実装・デバッグ → Coder
- 方針決定・レビュー → Kento

---

**最終更新:** 2026-02-08 13:50 JST
