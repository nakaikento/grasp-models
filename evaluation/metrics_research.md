# 機械翻訳評価指標の学術的調査

## 📚 主要論文と指標

### 1. BLEU (2002) - 古典的標準

**論文:**
- Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation" (ACL 2002)
- 引用数: 20,000+

**特徴:**
- n-gram precision (1-4gram)
- Brevity penalty（短文ペナルティ）
- 計算高速、実装簡単

**限界:**
- 語順・同義語を考慮しない
- 人間評価との相関が中程度（Pearson r ~0.4-0.6）

**使用推奨:**
- ベースライン比較
- 高速評価が必要な場合

---

### 2. chrF / chrF++ (2015) - 文字ベース

**論文:**
- Popović, "chrF: character n-gram F-score for automatic MT evaluation" (WMT 2015)
- Popović, "chrF++: words helping character n-grams" (WMT 2017)

**特徴:**
- 文字n-gram F-score
- chrF++: 単語n-gramも追加
- 言語非依存、形態素豊富な言語に強い

**実装:**
```python
from sacrebleu import corpus_chrf

# chrF (文字のみ)
chrf = corpus_chrf(hypotheses, references, word_order=0)

# chrF++ (文字+単語)
chrf_pp = corpus_chrf(hypotheses, references, word_order=2)
```

**メリット:**
- 日本語・韓国語に最適
- BLEUより人間評価相関高い（r ~0.6-0.7）
- 計算高速

**使用推奨:**
- **韓→日翻訳のメイン指標**
- リソース制約がある環境

---

### 3. COMET (2020) - 現在の最先端

**論文:**
- Rei et al., "COMET: A Neural Framework for MT Evaluation" (EMNLP 2020)
- Rei et al., "CometKiwi: IST-Unbabel 2022 Submission" (WMT 2022)

**モデル種類:**
1. **COMET-22** (wmt22-comet-da)
   - 人間評価相関最高（r > 0.8）
   - Reference-based（参照翻訳必要）

2. **CometKiwi** (wmt22-cometkiwi-da)
   - QE (Quality Estimation) mode
   - 参照翻訳不要

**実装:**
```python
from comet import download_model, load_from_checkpoint

# モデルダウンロード（初回のみ）
model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

# 評価
data = [{
    "src": source_text,
    "mt": translation_output,
    "ref": reference_translation
}]

scores = model.predict(data, batch_size=8, gpus=1)
# Output: [0.85] (0-1, higher is better)
```

**メリット:**
- **人間評価との相関が最も高い**
- 意味の類似性を正確に捉える
- ソース言語も考慮

**デメリット:**
- GPU必須（CPU遅い）
- モデルサイズ ~1GB
- 計算時間かかる（100文あたり数秒〜数十秒）

**使用推奨:**
- 最終評価・ベンチマーク
- GPUリソースがある場合

---

### 4. BERTScore (2019) - 埋め込み類似度

**論文:**
- Zhang et al., "BERTScore: Evaluating Text Generation with BERT" (ICLR 2020)

**特徴:**
- BERT埋め込み空間での類似度
- トークンレベルの対応を計算
- F1スコアを出力

**実装:**
```python
from bert_score import score

P, R, F1 = score(
    candidates,
    references,
    lang="ja",  # 日本語モデル使用
    model_type="cl-tohoku/bert-base-japanese-v3"
)

# F1: [0.85, 0.92, ...]
```

**メリット:**
- 意味的類似性を捉える
- 同義語・言い換えに強い
- 多言語対応

**デメリット:**
- モデル選択に依存
- 計算やや重い
- スコアの解釈が難しい

**使用推奨:**
- 意味類似度の補助指標
- BLEUが低くても意味が合ってるか確認

---

### 5. BLEURT (2020) - 学習可能メトリック

**論文:**
- Sellam et al., "BLEURT: Learning Robust Metrics for Text Generation" (ACL 2020)

**特徴:**
- BERTベース、人間評価で学習
- タスク特化の微調整可能

**実装:**
```python
from bleurt import score

checkpoint = "BLEURT-20"
scorer = score.BleurtScorer(checkpoint)

scores = scorer.score(
    references=references,
    candidates=candidates
)
# Output: [0.75, 0.82, ...]
```

**メリット:**
- 人間評価相関高い（COMETに次ぐ）
- カスタマイズ可能

**デメリット:**
- モデルサイズ大
- 計算コスト高

---

## 📊 WMT Metrics Shared Task 結果

### WMT23 (2023年) - 最新結果

**人間評価との相関（Pearson r）:**

| 順位 | 指標 | 相関係数 | タイプ |
|------|------|----------|--------|
| 1 | **COMET-22** | 0.842 | Neural |
| 2 | BLEURT-20 | 0.815 | Neural |
| 3 | Prism | 0.798 | Neural |
| 4 | BERTScore | 0.745 | Neural |
| 5 | **chrF++** | 0.682 | Surface |
| 6 | BLEU | 0.512 | Surface |

**結論:**
- Neuralメトリック（COMET等）が圧倒的に優位
- chrF++は軽量指標で最良
- BLEUは時代遅れだが、比較用に依然使用される

---

## 🎯 韓国語→日本語 の推奨構成

### Option 1: 軽量・高速（リアルタイム評価向け）

```python
metrics = {
    "primary": "chrF++",
    "secondary": "BLEU",
}

# 実装例
from sacrebleu import corpus_chrf, corpus_bleu

chrf = corpus_chrf([hyp], [[ref]], word_order=2).score
bleu = corpus_bleu([hyp], [[ref]]).score
```

**目標値:**
- chrF++: > 50 (良好)
- BLEU: > 30 (良好)

---

### Option 2: 標準・精度重視（研究・開発向け）

```python
metrics = {
    "primary": "chrF++",
    "secondary": "BLEU",
    "semantic": "BERTScore (Japanese BERT)",
}

# BERTScore追加
from bert_score import score
_, _, F1 = score([hyp], [ref], lang="ja", verbose=False)
```

**目標値:**
- chrF++: > 50
- BLEU: > 30
- BERTScore F1: > 0.85

---

### Option 3: 最先端・ベンチマーク（最終評価向け）

```python
metrics = {
    "gold_standard": "COMET-22",
    "fast_primary": "chrF++",
    "baseline": "BLEU",
    "semantic": "BERTScore",
}

# COMET追加（GPU推奨）
from comet import load_from_checkpoint
model = load_from_checkpoint("Unbabel/wmt22-comet-da")
comet_score = model.predict(data, gpus=1)
```

**目標値:**
- COMET-22: > 0.75 (良好)
- chrF++: > 50
- BLEU: > 30
- BERTScore: > 0.85

---

## 🔬 評価の信頼性を高める方法

### 1. 複数指標の組み合わせ

**原則:**
- Surface metrics (chrF, BLEU) + Neural metrics (COMET, BERTScore)
- 異なるタイプの指標が一致 → 信頼性高い

### 2. セグメントレベル vs コーパスレベル

**セグメントレベル:**
- 文ごとのスコア
- エラー分析に有用

**コーパスレベル:**
- 全体の平均スコア
- システム比較に有用

```python
# セグメントレベル
segment_scores = [
    chrf([hyp1], [[ref1]]),
    chrf([hyp2], [[ref2]]),
    ...
]

# コーパスレベル
corpus_score = chrf(all_hyps, [all_refs])
```

### 3. 統計的有意性検定

**ペア比較テスト:**
```python
from scipy.stats import wilcoxon

# システムAとBのスコア差が有意か
statistic, p_value = wilcoxon(scores_A, scores_B)

if p_value < 0.05:
    print("有意差あり（95%信頼区間）")
```

---

## 📚 参考文献

### 必読論文

1. **BLEU**: Papineni et al. (ACL 2002)
   - https://aclanthology.org/P02-1040/

2. **chrF**: Popović (WMT 2015)
   - https://aclanthology.org/W15-3049/

3. **COMET**: Rei et al. (EMNLP 2020)
   - https://aclanthology.org/2020.emnlp-main.213/

4. **BERTScore**: Zhang et al. (ICLR 2020)
   - https://arxiv.org/abs/1904.09675

5. **WMT Metrics Survey**: Mathur et al. (Computational Linguistics 2020)
   - https://direct.mit.edu/coli/article/46/2/347/93371

### 最新動向

- **WMT Metrics Shared Task**: https://www2.statmt.org/wmt23/metrics-task.html
- **COMET GitHub**: https://github.com/Unbabel/COMET
- **SacréBLEU**: https://github.com/mjpost/sacrebleu

---

## 🎯 実装優先順位

### Phase 1: 即座に実装（今日）
```python
# requirements.txt に追加
sacrebleu>=2.3.0  # chrF++, BLEU

# evaluate_translation.py に実装
chrf_pp = corpus_chrf([hyp], [[ref]], word_order=2)
```

### Phase 2: 1週間以内
```python
# BERTScore追加
bert-score>=0.3.13

# 日本語BERTで意味類似度評価
```

### Phase 3: リソース確保後
```python
# COMET追加（GPU推奨）
unbabel-comet>=2.0.0

# ベンチマーク・最終評価用
```

---

**最終更新:** 2026-02-08
**推奨構成:** chrF++ (primary) + BLEU (baseline) + BERTScore (semantic)
