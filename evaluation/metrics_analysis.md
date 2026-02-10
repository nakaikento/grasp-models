# 評価指標の詳細ガイド

## 📊 指標の選定理由

### ASR評価: CER（文字誤り率）をメインに

**理由:**
- 韓国語は分かち書きが曖昧（띄어쓰기）
- WERは分かち書きに敏感すぎる
- CERの方が人間の認識に近い

**例:**
```
正解: "나는 남자 친구를 아는 줄 알았어"
ASR:  "나는 남자친구를 아는줄 알았어"

WER: 40% (띄어쓰기 2箇所で4単語中2単語エラー)
CER:  0% (文字は完全一致、띄어쓰기だけ違う)
```

### 翻訳評価: chrF をメインに

**理由:**
- 日本語は文字単位の評価が適している
- BLEUは単語分割に敏感（日本語の分かち書きは不安定）
- chrFは文字n-gramなので安定

**例:**
```
正解: "彼氏を知っていると思っていました"
翻訳: "彼氏を知ってると思ってた"

BLEU: 低い（単語分割の違いで大きく下がる）
chrF: 高い（文字レベルでは類似）
```

---

## 🎯 目標値の根拠

### CER (Character Error Rate)

| 範囲 | 評価 | 例 |
|------|------|-----|
| 0-5% | 🏆 優秀 | プロレベル、商用ASR |
| 5-10% | ✅ 良好 | 実用レベル、ほぼ理解可能 |
| 10-20% | ⚠️ 要改善 | 意味は取れるが誤字多い |
| 20%+ | ❌ 不合格 | 使い物にならない |

**参考データ:**
- Google Speech API (韓国語): CER ~3-5%
- Whisper large-v3 (韓国語): CER ~5-8%
- Sherpa-onnx (韓国語): CER ~8-15% (モデル次第)

### chrF (Character n-gram F-score)

| 範囲 | 評価 | 例 |
|------|------|-----|
| 60+ | 🏆 優秀 | プロ翻訳に近い |
| 50-60 | ✅ 良好 | 十分実用的 |
| 40-50 | ⚠️ 要改善 | 意味は通じる |
| 40未満 | ❌ 不合格 | 意味不明 |

**参考データ:**
- Google Translate (韓→日): chrF ~60-65
- DeepL (韓→日): chrF ~65-70
- MarianMT (ko-ja): chrF ~45-55 (一般的)

### BLEU

| 範囲 | 評価 | 意味 |
|------|------|------|
| 40+ | 🏆 優秀 | 人間に近い |
| 30-40 | ✅ 良好 | 実用的 |
| 20-30 | ⚠️ 要改善 | 理解可能 |
| 20未満 | ❌ 不合格 | 品質低い |

---

## 📈 評価ワークフロー

### ステップ1: 個別コンポーネント評価

```bash
# ASR精度
python evaluate_asr.py \
  data/audio/VIDEO_ID.wav \
  data/subtitles/VIDEO_ID.ko.vtt \
  -o data/results/asr_results.json

# 翻訳精度（クリーンな入力）
python evaluate_translation.py \
  data/subtitles/VIDEO_ID.ko.vtt \
  data/subtitles/VIDEO_ID.ja.vtt \
  -o data/results/translation_results.json
```

**期待される結果:**
```json
{
  "asr": {
    "cer": 8.5,
    "wer": 12.3
  },
  "translation": {
    "bleu": 35.2,
    "chrf": 52.8
  }
}
```

### ステップ2: E2E評価

```bash
# E2E（音声→翻訳）
python evaluate_e2e.py \
  data/audio/VIDEO_ID.wav \
  data/subtitles/VIDEO_ID.ja.vtt \
  --reference-ko data/subtitles/VIDEO_ID.ko.vtt \
  -o data/results/e2e_results.json
```

**期待される結果:**
```json
{
  "e2e": {
    "bleu": 31.5,
    "chrf": 48.3
  },
  "degradation": {
    "bleu_loss": 3.7,  // (35.2 - 31.5)
    "chrf_loss": 4.5   // (52.8 - 48.3)
  }
}
```

### ステップ3: 問題診断

```python
# 自動診断ロジック
if results["asr"]["cer"] > 15:
    print("❌ 優先課題: ASR精度改善")
    print("   - より良い音響モデル")
    print("   - 韓国語特化のファインチューニング")
    
elif results["translation"]["chrf"] < 45:
    print("❌ 優先課題: 翻訳モデル改善")
    print("   - より大きなモデル")
    print("   - 追加の学習データ")
    
elif results["degradation"]["chrf_loss"] > 10:
    print("⚠️ ASRエラーが翻訳に悪影響")
    print("   - 翻訳モデルのロバスト性向上")
    print("   - エラー訂正機構の追加")
    
else:
    print("✅ 全体的に良好")
    print("   - 細かい改善で完成度向上")
```

---

## 🔬 詳細分析

### エラーパターン分析

**ASRエラーの分類:**
```python
def analyze_asr_errors(reference, hypothesis):
    errors = {
        'phonetic': 0,      # 音韻的に似た誤り (예 vs 애)
        'spacing': 0,       # 띄어쓰기エラー
        'missing': 0,       # 聞き逃し
        'hallucination': 0  # 幻聴
    }
    # 実装...
    return errors
```

**翻訳エラーの分類:**
```python
def analyze_translation_errors(reference, hypothesis):
    errors = {
        'mistranslation': [],  # 誤訳の例
        'omission': [],        # 訳抜けの例
        'fluency': [],         # 不自然な表現
        'formatting': []       # 体裁の問題
    }
    # 実装...
    return errors
```

---

## 📊 レポート生成

### 評価レポートのフォーマット

```markdown
# 評価レポート: VIDEO_ID

## 📅 基本情報
- 動画: [タイトル]
- 長さ: 10:25
- 評価日: 2026-02-08

## 📊 結果サマリー

| 指標 | 値 | 評価 | 目標 |
|------|-----|------|------|
| ASR CER | 8.5% | ✅ | < 10% |
| 翻訳 chrF | 52.8 | ✅ | > 50 |
| E2E chrF | 48.3 | ⚠️ | > 50 |

## 🎯 総合評価: ✅ 良好

### 強み
- ASR精度は実用レベル
- 翻訳品質も十分

### 弱み
- ASRエラーが翻訳品質を4.5ポイント下げている

### 推奨アクション
1. ASRのロバスト性向上（優先度: 中）
2. 翻訳モデルのエラー耐性強化（優先度: 低）

## 📝 詳細データ
[JSON出力へのリンク]
```

---

## 🚀 次のステップ

1. **ベースライン測定** - 現状の性能を記録
2. **目標設定** - 達成したい指標値を決定
3. **改善サイクル** - モデル改善 → 再評価 → 比較
4. **ユーザー評価** - 定量指標と主観評価の相関確認

---

**参考文献:**
- BLEU: https://aclanthology.org/P02-1040/
- chrF: https://aclanthology.org/W15-3049/
- WER/CER: https://en.wikipedia.org/wiki/Word_error_rate
- COMET: https://aclanthology.org/2020.emnlp-main.213/
