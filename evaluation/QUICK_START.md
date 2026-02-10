# 📊 評価システム クイックスタート

推奨評価指標（chrF++ + BLEU）でテストする方法

---

## ✅ セットアップ完了状況

```
✅ chrF++ (PRIMARY) - 文字+単語 n-gram評価
✅ BLEU (BASELINE) - 業界標準（文字レベル）
⚠️ BERTScore (SEMANTIC) - 現在の環境では動作しない（オプション）
```

---

## 🚀 クイックテスト

### 方法1: サンプルデータでテスト

```bash
cd evaluation
python3 quick_test.py --test
```

**出力例:**
```
============================================================
Test Case 1: Good translation
============================================================
Korean:     나는 남자 친구를 아는 줄 알았어
Reference:  彼氏を知っていると思っていました
Hypothesis: 彼氏を知ってると思ってた

📊 Results:
   chrF++:     39.00 ❌
   BLEU:       53.82 🏆
```

### 方法2: カスタムテキストでテスト

```bash
python3 quick_test.py "한국어 텍스트" "日本語参照翻訳"
```

---

## 📊 評価指標の解釈

### chrF++ (PRIMARY METRIC)

**計算方法:**
- 文字n-gram (1-6) + 単語n-gram (1-2)
- Precision と Recall の調和平均（F1）

**目標値:**
| スコア | 評価 | 説明 |
|--------|------|------|
| 60+ | 🏆 優秀 | プロ翻訳に近い |
| 50-60 | ✅ 良好 | **実用レベル（目標）** |
| 40-50 | ⚠️ 要改善 | 意味は通じる |
| < 40 | ❌ 不合格 | 品質低い |

**例:**
```
Reference:  彼氏を知っていると思っていました
Good (55):  彼氏を知ってると思ってた
Poor (35):  男の友達を知る思った
```

### BLEU (BASELINE)

**計算方法:**
- 文字n-gram (1-4) の一致率
- 日本語用に文字レベルで計算

**目標値:**
| スコア | 評価 | 説明 |
|--------|------|------|
| 40+ | 🏆 優秀 | 人間に近い |
| 30-40 | ✅ 良好 | 実用的 |
| 20-30 | ⚠️ 要改善 | 理解可能 |
| < 20 | ❌ 不合格 | 品質低い |

**注意:**
- BLEUは厳しい指標（完全一致を重視）
- chrF++の方が人間評価に近い
- 両方を参考にする

---

## 📁 TED Talk評価の実行

### 準備（日本語字幕取得後）

```bash
cd ~/.openclaw/workspace/grasp-models/evaluation

# ファイル確認
ls data/audio/      # -WVhcG0rauI.wav (176MB)
ls data/subtitles/  # -WVhcG0rauI.ko.vtt, -WVhcG0rauI.ja.vtt
```

### 評価実行

```bash
# 翻訳精度評価（韓国語テキスト → 日本語）
python3 evaluate_translation.py \
  data/subtitles/-WVhcG0rauI.ko.vtt \
  data/subtitles/-WVhcG0rauI.ja.vtt \
  --model-dir ../models/ko-ja-onnx-int8 \
  -o data/results/translation_results.json
```

**期待される出力:**
```
============================================================
📊 EVALUATION RESULTS (WMT2023 Recommended Metrics)
============================================================

🎯 PRIMARY: chrF++ (character + word n-gram)
   Score: 52.30 ✅
   Target: > 50 (Good), > 60 (Excellent)

📏 BASELINE: BLEU (industry standard)
   Score: 35.20 ✅
   Target: > 30 (Good), > 40 (Excellent)

============================================================
🎯 OVERALL ASSESSMENT
============================================================
   ✅ GOOD - Practical quality
```

---

## 🔧 トラブルシューティング

### ImportError: No module named 'sacrebleu'

```bash
pip install sacrebleu>=2.3.0
```

### BERTScore が動作しない

→ **正常です**。BERTScoreは現在オプション機能で、動作しなくても評価可能です。
→ chrF++とBLEUで十分な評価ができます。

### 翻訳モデルが見つからない

```bash
# モデルディレクトリを確認
ls ../models/ko-ja-onnx-int8/

# 必要なファイル:
# - encoder_model_quantized.onnx
# - decoder_model_quantized.onnx
```

---

## 📈 結果の解釈例

### ケース1: 良好な翻訳

```
chrF++: 55.0 ✅
BLEU:   38.5 ✅
→ 実用レベル、デプロイ可能
```

### ケース2: 改善が必要

```
chrF++: 42.0 ⚠️
BLEU:   25.0 ⚠️
→ 意味は通じるが、改善の余地あり
```

### ケース3: 問題あり

```
chrF++: 30.0 ❌
BLEU:   15.0 ❌
→ モデル改善が必要
```

---

## 🎯 次のステップ

### 1. ベースライン測定
現在のモデルの性能を記録

### 2. 改善サイクル
- モデル改善 → 再評価 → 比較

### 3. 複数データでテスト
- TED Talk 5-10本で平均スコア
- より信頼性の高い評価

---

## 📚 参考指標

### WMT2023 (機械翻訳国際会議) 基準

| システム | chrF++ | BLEU |
|---------|--------|------|
| 人間翻訳 | ~80 | ~60 |
| **目標（実用）** | **> 50** | **> 30** |
| Google Translate | ~65 | ~45 |
| DeepL | ~70 | ~50 |
| MarianMT (平均) | ~50 | ~32 |

---

**最終更新:** 2026-02-08 14:10 JST
**推奨指標:** chrF++ (PRIMARY) + BLEU (BASELINE)
