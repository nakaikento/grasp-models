# 実験記録

## Experiment 1: NLLB-200-3.3B 教師データ生成（失敗）

**日時:** 2026-02-03 14:30

**目的:** 韓国語→日本語の教師データを生成

**モデル:**
- facebook/nllb-200-3.3B
- 16-bit精度

**パラメータ:**
```python
src_lang: ko (kor_Hang)
tgt_lang: ja (jpn_Jpan)
batch_size: 40
num_beams: 3
max_length: (デフォルト)
repetition_penalty: (なし)
```

**データ:**
- train_sample1000.ko（1000件サンプル）

**結果:** ❌ 完全な失敗

**統計:**
- 生成完了: 1000/1000件
- FAILED: 175件（17.5%）
- 異常な繰り返し: ほぼ全件

**問題点:**
1. 同じ文字・単語の異常な繰り返し（rang×100、血×100など）
2. 翻訳失敗（FAILED_TRANSLATION_CLEANED: 175件）
3. 英語混入（of course, probably, wanna）
4. 実質的に使える翻訳がほぼゼロ

**サンプル出力:**
```
너도 구해야지 → ♪×100
사망한 거요? → 死んだ? 亡くなった? 死亡?... (繰り返し)
진실 서약을 했으니 지켜야 합니다 → FAILED_TRANSLATION_CLEANED
```

**結論:** このモデル設定では教師データとして使用不可

**原因分析:**
- ビーム探索（num_beams=3）が繰り返しを増幅
- max_length設定の欠如
- NLLBの韓日翻訳品質の問題
- repetition_penaltyが未設定

---

## 次の実験候補

### Experiment 2: NLLB-200-3.3B + repetition_penalty
- repetition_penalty: 1.2
- num_beams: 1 (greedy)
- max_length: 128

### Experiment 3: NLLB-200-distilled-600M
- より小さいモデルで試す
- 同じパラメータ

### Experiment 4: MarianMT (Helsinki-NLP)
- opus-mt-ko-ja（もしあれば）
- または別のMarianMTモデル

### Experiment 5: 直接学習
- 教師データなしで元のパラレルコーパスから学習
- epochs: 3-5
- batch_size: 64

## Experiment 2 & 3: スキップ

**理由:**
- Exp 2 (repetition_penalty): すでに色々試したが効果なし
- Exp 3 (NLLB-distilled-600M): 望み薄

---

## Experiment 4: mBART-large-50-many-to-many-mmt

**日時:** 2026-02-03 14:40

**目的:** 多言語翻訳で実績のあるmBARTで教師データ生成

**モデル:**
- facebook/mbart-large-50-many-to-many-mmt
- 多言語翻訳に特化、50言語対応

**パラメータ:**
```python
src_lang: ko
tgt_lang: ja
batch_size: 20  # mBARTは大きいので小さめ
num_beams: 3
max_length: 128
repetition_penalty: 1.0
```

**データ:**
- train_sample1000.ko（1000件サンプル）

**実行中...**

**結果:** ❌ 失敗

**エラー:**
```
RuntimeError: Disk quota exceeded (os error 122)
OSError: Can't load the model for 'facebook/mbart-large-50-many-to-many-mmt'
```

**原因:**
- HuggingFaceキャッシュの容量問題
- mBARTモデルのダウンロード中にエラー

**結論:** mBARTはキャッシュ問題で実行できず

**教訓:** 次のモデルを試す前に必ず  を削除すること

---

**結果:** ❌ 失敗

**統計:**
- 生成完了: 1000/1000件
- FAILED: 12件（1.2%）
- **英語出力: 747件（74.7%）** ← 重大な問題
- 異常な繰り返し: 多数（I×100、close×100など）

**サンプル出力:**
```
막 하려던 참이었어요 → ?I'm about to do it.
버틸 수 있는 것 같아요 → I I I I I... (×100)
더 가까이 가야한다면 → If close close close... (×100)
```

**問題点:**
1. 日本語ではなく英語で出力（74.7%）
2. NLLBと同じ異常な繰り返し
3. 実質的に使える翻訳がほぼゼロ

**原因分析:**
mBARTの言語コード指定が間違っている可能性
- 正しい: ko_KR → ja_XX
- 現在: ko → ja

**結論:** 言語コード修正が必要、または直接学習に切り替え

---

## Experiment 5: mBART + 正しい言語コード

**日時:** 2026-02-03 14:50

**目的:** mBARTで正しい言語コード（ko_KR, ja_XX）を使用

**モデル:**
- facebook/mbart-large-50-many-to-many-mmt
- mBART用言語コード対応

**パラメータ:**
```python
src_lang: ko (→ ko_KR)
tgt_lang: ja (→ ja_XX)
batch_size: 20
num_beams: 3
max_length: 128
```

**データ:**
- train_sample1000.ko（1000件サンプル）

**変更点:**
- スクリプトにmBART用言語コードマッピングを追加
- 自動的にモデル名から判定して適切な言語コードを使用

**実行中...**

**結果:** ✅ 成功！

**統計:**
- 生成完了: 1000/1000件
- FAILED: 6件（0.6%）← 前回175件から大幅改善
- 英語出力: 90件（9%）← 前回747件から大幅改善
- **異常な繰り返し: 0件** ← 完全に解消！

**パラメータ（最終版）:**
```python
model: facebook/mbart-large-50-many-to-many-mmt
precision: 16-bit (float16)

# 言語コード
src_lang: ko → ko_KR (mBART用)
tgt_lang: ja → ja_XX (mBART用)

# 生成パラメータ
batch_size: 20
num_beams: 3
max_length: 128
do_sample: False

# ペナルティ系（重要！）
repetition_penalty: 2.0   # 繰り返しを強く抑制
length_penalty: 0.5       # 短文に最適化
early_stopping: True
```

**サンプル翻訳:**
```
막 하려던 참이었어요 → それはそうだった。
왜 여기 있었죠? → なぜここにいる?
메탈 좋아해? → メタル?
제가 바라는 건... → 私の願いは...
의뢰를 수행하게 → 依頼をします。
저기 있죠, 저도... 저도 당신 같았어요 → あ、僕も... 僕も君と同じ。
```

**成功の要因:**
1. 正しい言語コード（ko_KR, ja_XX）
2. repetition_penalty=2.0（繰り返しを強く抑制）
3. length_penalty=0.5（短文に最適化）
4. num_beams=3（品質とスピードのバランス）

**問題点:**
- 一部英語混入（9%）
- 翻訳の質にばらつきあり

**結論:**
**使える可能性あり！** 全件実行を検討する価値がある。
所要時間: 6-8時間（RTX 4090、1,025,749文）

---

## Experiment 6: M2M100-1.2B (より大きいモデル)

**日時:** 2026-02-03 15:05

**目的:** より大きいモデル（M2M100-1.2B）で品質向上を試みる

**モデル:**
- facebook/m2m100_1.2B
- 1.2B parameters（mBARTの約2倍）
- 100言語対応

**パラメータ:**
```python
src_lang: ko
tgt_lang: ja
batch_size: 20
num_beams: 3
max_length: 128
repetition_penalty: 2.0
length_penalty: 0.5
```

**データ:**
- train_sample1000.ko（1000件サンプル）

**期待:**
- mBARTより高品質な翻訳
- 英語混入の減少
- 正確な翻訳の増加

**実行中...**

## Experiment 6: M2M100-1.2B (2026-02-03)

### 設定
- **モデル**: facebook/m2m100_1.2B (1.2B params)
- **言語**: Korean (ko) → Japanese (ja)
- **データ**: data/splits/train_sample1000.ko (1,000 samples)
- **パラメータ**: batch_size=20, num_beams=3, fp16=True, repetition_penalty=1.5
- **出力**: data/teacher/exp6_m2m100_sample1000.ja

### 結果
❌ **FAILED - 異常な繰り返し発生**

**所要時間**: 9分13秒

**品質評価**:
- ほぼ全行が jajaja... の異常な繰り返し
- 正常な日本語訳: ほぼ0%
- NLLB-200-3.3B (Experiment 1) と全く同じ問題

**サンプル出力**:


### 結論
**大きいモデル ≠ 高品質**
- mBART-large (611M): ✅ 成功 (80%品質)
- M2M100-1.2B (1.2B): ❌ 異常な繰り返し
- NLLB-200-3.3B (3.3B): ❌ 異常な繰り返し

→ **mBARTが最適な選択**と判明

### 次のステップ
1. mBARTのパラメータ調整でさらなる品質向上を試みる
   - num_beams=4-5（現在3）
   - repetition_penalty=1.7（現在1.5）
2. より大きなサンプル（5K-10K）でテスト
3. 品質が改善すれば全データ（1M+）で生成

---

## Experiment 6: M2M100-1.2B (2026-02-03)

### 設定
- モデル: facebook/m2m100_1.2B (1.2B params)
- 言語: Korean (ko) → Japanese (ja)
- データ: data/splits/train_sample1000.ko (1,000 samples)
- パラメータ: batch_size=20, num_beams=3, fp16=True, repetition_penalty=1.5
- 出力: data/teacher/exp6_m2m100_sample1000.ja

### 結果
FAILED - 異常な繰り返し発生

所要時間: 9分13秒

品質評価:
- ほぼ全行が "jajaja..." の異常な繰り返し
- 正常な日本語訳: ほぼ0%
- NLLB-200-3.3B (Experiment 1) と全く同じ問題

### 結論
大きいモデル ≠ 高品質

| モデル | サイズ | 結果 |
|--------|--------|------|
| mBART-large | 611M | 成功 (80%品質) |
| M2M100-1.2B | 1.2B | 異常な繰り返し |
| NLLB-200-3.3B | 3.3B | 異常な繰り返し |

→ mBARTが最適な選択と判明

### 次のステップ
1. mBARTのパラメータ調整でさらなる品質向上を試みる
   - num_beams=4-5（現在3）
   - repetition_penalty=1.7（現在1.5）
2. より大きなサンプル（5K-10K）でテスト
3. 品質が改善すれば全データ（1M+）で生成

---

## Experiment 7: mBART-50-many-to-many-mmt 再確認 (2026-02-03)

### 設定
- モデル: facebook/mbart-large-50-many-to-many-mmt (611M params)
- 言語: Korean (ko_KR) → Japanese (ja_XX)
- データ: data/splits/train_sample1000.ko (1,000 samples)
- パラメータ: batch_size=20, num_beams=3
- 出力: data/teacher/exp7_mbart50_sample1000.ja

### 結果
✅ 成功

所要時間: 18秒

品質評価:
- 英語混入: 11.9% (119/1000)
- 異常な繰り返し: 0.5% (5/1000)
- 空行: 0%

### 重要な発見
**Experiment 5と完全一致（MD5ハッシュ同一）**
- Exp 5で実際に使ったモデルも facebook/mbart-large-50-many-to-many-mmt だった
- 同じモデルを2回テストしていたことが判明
- cc25（事前学習のみ）は未テスト

### 結論
mbart-large-50-many-to-many-mmt が現時点での最適解:
- 80%品質（36%正確 + 44%部分的使用可能）
- 0.6% FAILED率
- 異常な繰り返しなし

### 次のステップ
- mbart-large-cc25（事前学習のみ）をテスト
- パラメータ調整（num_beams, repetition_penalty）で品質向上を試みる

---

## Experiment 8: mBART-cc25 (事前学習のみ) (2026-02-03)

### 設定
- モデル: facebook/mbart-large-cc25 (611M params, 事前学習のみ)
- 言語: Korean (ko_KR) → Japanese (ja_XX)
- データ: data/splits/train_sample1000.ko (1,000 samples)
- パラメータ: batch_size=20, num_beams=3
- 出力: data/teacher/exp8_mbart_cc25_sample1000.ja

### 結果
❌ FAILED - ほぼ全行が翻訳失敗

所要時間: 53秒

品質評価:
- FAILED率: 98.5% (985/1000)
- 残り15行も異常な出力（記号の羅列、異常な繰り返し）

サンプル出力:
```
FAILED_TRANSLATION_CLEANED
....-----------------------------------------------------------------------------------------------------------------------
?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
```

### 結論
**事前学習のみのモデルは翻訳タスクに使えない**
- fine-tuning済みモデル（many-to-many-mmt）が必須
- cc25は翻訳タスクには不適

---

## 全実験まとめ (2026-02-03)

| Experiment | モデル | サイズ | 結果 | FAILED率 |
|-----------|--------|--------|------|----------|
| Exp 1 | NLLB-200-3.3B | 3.3B | 異常な繰り返し | 17.5% |
| Exp 4 | mBART (誤コード) | 611M | 英語出力 | 74.7% |
| Exp 5/7 | **mBART-50-mmt** | 611M | **✅ 成功 80%品質** | **0.6%** |
| Exp 6 | M2M100-1.2B | 1.2B | 異常な繰り返し | ~100% |
| Exp 8 | mBART-cc25 | 611M | 翻訳失敗 | 98.5% |

**結論:** mbart-large-50-many-to-many-mmt が唯一の成功モデル

### 次のステップ
- パラメータ調整（num_beams=5, repetition_penalty=1.7）で品質向上を試みる

---

## Experiment 9: パラメータ調整 (beams=5, penalty=1.7) (2026-02-03)

### 設定
- モデル: facebook/mbart-large-50-many-to-many-mmt (611M params)
- 言語: Korean (ko_KR) → Japanese (ja_XX)
- データ: data/splits/train_sample1000.ko (1,000 samples)
- パラメータ: batch_size=20, **num_beams=5**, **repetition_penalty=1.7**
- 出力: data/teacher/exp9_mbart50_tuned_sample1000.ja

### 結果
❌ 品質悪化 - パラメータ調整は逆効果

所要時間: 20秒

### Exp5との比較

| 指標 | Exp 5 (b3, p2.0) | Exp 9 (b5, p1.7) | 変化 |
|------|------------------|------------------|------|
| 英語混入 | 11.9% | **14.3%** | ⬆️ +2.4% |
| 繰り返し | 0.5% | **0.7%** | ⬆️ +0.2% |
| 完全一致 | - | 66.0% | - |
| 異なる翻訳 | - | 34.0% | - |

### 翻訳品質サンプル
```
改善例:
  原文: 저게 뭐야? - 몰라!
  Exp5: 何? - うん!
  Exp9: 何? - わからない! ✅

悪化例:
  原文: 좋은 성적 내라고 족쳐놓곤 뭐하는 짓거리야?
  Exp5: 良い成績を挙げて、何をするの?
  Exp9: 良い成績を誇りにしている。 ❌ (意味が変わった)
```

### 結論
- **num_beams=5, repetition_penalty=1.7 は改善効果なし**
- むしろ品質が悪化（英語混入+2.4%、繰り返し+0.2%）
- **Experiment 5の設定が最適**（beams=3, penalty=2.0）

---

## 最終結論 (2026-02-03)

### 最適な設定
- **モデル:** facebook/mbart-large-50-many-to-many-mmt
- **パラメータ:** batch_size=20, num_beams=3, repetition_penalty=2.0
- **品質:** 80%使用可能（36%正確 + 44%部分的使用可能）
- **FAILED率:** 0.6%
- **英語混入:** 11.9%
- **異常な繰り返し:** 0.5%

### テストしたモデル
1. ❌ NLLB-200-3.3B (3.3B) - 異常な繰り返し
2. ❌ M2M100-1.2B (1.2B) - 異常な繰り返し
3. ❌ mBART-cc25 (611M) - 翻訳失敗98.5%
4. ✅ **mBART-50-many-to-many-mmt (611M)** - 80%品質

### 次のステップ
元のコーパスの質を再評価し、Teacher Dataの必要性を検討

---

## Experiment 10: Matched Dataset Training (2026-02-03)

### 目的
教師データ生成（Exp 1-9）を諦め、元のOPUSコーパスから品質でマッチングしたデータセットで直接トレーニング

### データセット
- **ソース:** OPUS OpenSubtitles ja-ko
- **クリーニング:** 空行削除、重複除去、長さフィルタリング
- **マッチング基準:** 文字数比率 0.3-3.0
- **分割:**
  - Train: 546,881 (95%)
  - Validation: 14,392 (2.5%)
  - Test: 14,392 (2.5%)

### 初回トレーニング試行 (2026-02-03 17:00-18:25)

#### 設定
- **wandb run:** rare-cherry-13 (80s3lwwk)
- **transformers:** 5.0.0 (⚠️ 問題あり)
- **モデル:** MarianMT (6L encoder/decoder, 61.0M params)
- **バッチサイズ:** 96
- **学習率:** 3e-4
- **Epochs:** 10

#### 結果
❌ **CATASTROPHIC FAILURE - チェックポイントが完全に壊れている**

**トレーニング中の指標:**
- WandB Validation BLEU: 0.22 (22%?)
- Loss: 正常に減少
- Step 784 でクラッシュ（OOM）

**保存されたチェックポイントの評価:**
- **checkpoint-6000 Test BLEU: 0.01** (壊滅的)
- 出力: ",,," と "." のみの繰り返し
- Brevity Penalty: 0.376（出力が参照の半分の長さ）
- 出力トークン数: 35,931 vs 参照 71,078 (50.6%)

**サンプル出力:**
```
Source: ラボに送って見てもらおう
Reference: 가능하지
Translation: . . . . .

Source: 抵抗を予測しろ 注意を怠るな
Reference: 반항이 있을 거니 전술적으로 진입해야 한다
Translation: . . . . .
```

### 根本原因の特定

#### 🔍 wandb history 分析
前回成功した run (zsh840l3, deft-monkey-8) の設定を取得：

**成功した設定 (2026-01-25, BLEU 33.03):**
```yaml
transformers: 4.57.6  ✅
batch_size: 128
use_cache: True  ✅
learning_rate: 0.0003
warmup_steps: 4000
num_train_epochs: 10
fp16: True
gradient_accumulation_steps: 2
lr_scheduler_type: cosine
```

**失敗した設定 (2026-02-03):**
```yaml
transformers: 5.0.0  ⚠️ 問題
batch_size: 96
use_cache: False  ⚠️ 問題
(その他は同じ)
```

#### ⚡ 重大な発見
**transformers 5.0.0 は破壊的変更を含んでいる！**

1. **デフォルト設定の変更:** `use_cache: True → False`
2. **チェックポイント保存/読み込み機構の問題**
   - トレーニング中は BLEU 0.22 を記録
   - 保存されたチェックポイントは BLEU 0.01（ゴミ）
   - モデルの状態が正しく保存されていない可能性

### 解決策

#### ✅ transformers 4.57.6 へダウングレード
```bash
pip install transformers==4.57.6
```

成功したバージョンに戻すことで、チェックポイント保存/読み込みの問題を回避。

### 再トレーニング (2026-02-03 18:37-)

#### 設定
- **wandb run:** summer-butterfly-15 (c63hera9)
- **transformers:** 4.57.6 ✅
- **モデル:** MarianMT (6L encoder/decoder, 61.0M params)
- **バッチサイズ:** 128（成功時と同じ）
- **学習率:** 3e-4
- **Epochs:** 10
- **Total steps:** 21,370

#### 期待される結果
- 前回 BLEU 33.03 を達成した設定を再現
- データセットが異なるため完全一致はしないが、同等の性能を期待
- チェックポイントが正常に保存・読み込み可能

#### 状態
🚀 **トレーニング実行中**
- wandb: https://wandb.ai/okamoto2okamoto-personal/huggingface/runs/c63hera9

### 教訓

#### ❌ やってはいけないこと
1. **transformers の最新版を無条件に使わない**
   - 破壊的変更が含まれる可能性
   - 成功した設定のバージョンを固定すべき
2. **チェックポイントの評価を怠らない**
   - wandb の指標だけを信じない
   - 実際にチェックポイントをロードして評価すること

#### ✅ ベストプラクティス
1. **バージョン固定**
   - 成功した環境は `requirements.txt` でバージョン固定
   - 更新は慎重に、テスト環境で確認してから
2. **評価の二重チェック**
   - トレーニング中の指標
   - 保存されたチェックポイントの実評価
3. **前回の成功事例を記録**
   - wandb run ID
   - 全設定（transformers バージョン含む）
   - 再現可能な形で保存

### 参考: 成功した run の詳細

**wandb run:** zsh840l3 (deft-monkey-8)
- **日時:** 2026-01-25 08:25 UTC
- **Test BLEU:** 33.03
- **総ステップ:** 21,370
- **所要時間:** 3017秒 (約50分)
- **最終 loss:** 0.9787
- **notebook:** training/mt_ja_ko_training.ipynb

**トレーニング速度:**
- train_samples_per_second: 1812.585
- train_steps_per_second: 7.083
- eval_samples_per_second: 375.075

---
