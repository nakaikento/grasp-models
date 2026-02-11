# LLM翻訳品質検証 実験設計

## 目的
Qwen3-32B / DeepSeek-R1で直訳問題を克服できるかを定量検証し、
大規模合成データ生成のプロンプト戦略を決定する。

## ステータス
- [x] 実験設計
- [x] FLORES-200から1012文の日韓並列コーパス抽出
- [x] 12条件対応の翻訳スクリプト
- [x] chrF++/BLEU/COMET/COMET-QE対応の評価スクリプト
- [x] エラー分析スクリプト
- [ ] LLM翻訳実行（RunPod or API待ち）
- [ ] 評価実行
- [ ] エラー分析

## 実験条件 (12条件)

### モデル (3種)
| ID | Model | Size | Provider |
|----|-------|------|----------|
| M1 | Qwen3-32B | 32B | vLLM/OpenRouter |
| M2 | Qwen3-235B-A22B | 235B (22B active) | OpenRouter |
| M3 | DeepSeek-R1-Distill-32B | 32B | vLLM/OpenRouter |

### プロンプト戦略 (4種)
| ID | Strategy | Description |
|----|----------|-------------|
| P1 | zero-shot | シンプルな翻訳指示のみ |
| P2 | few-shot | 3例の高品質翻訳例を含む |
| P3 | thinking | Thinking Mode有効化（文化的コンテキスト考慮） |
| P4 | natural | 意訳指示（「自然な日本語に」「直訳を避けて」） |

### 固定パラメータ
- Temperature: 0.3
- Max tokens: 256
- n: 1 (single output)

### 条件マトリクス
```
       P1    P2    P3    P4
M1    M1P1  M1P2  M1P3  M1P4
M2    M2P1  M2P2  M2P3  M2P4
M3    M3P1  M3P2  M3P3  M3P4
```

## 評価指標

### 参照あり (Reference-based)
| Metric | Target | Notes |
|--------|--------|-------|
| **chrF++** | > 50 (良好), > 60 (優秀) | PRIMARY - WMT2023推奨 |
| **BLEU** | > 30 (良好), > 40 (優秀) | BASELINE - 業界標準 |
| **COMET** | > 0.80 | SEMANTIC - 人間評価相関 r=0.84 |

### 参照なし (Reference-free)
| Metric | Target | Notes |
|--------|--------|-------|
| **COMET-QE** | > 0.70 | 大規模生成時のフィルタリング閾値決定用 |

## データセット

### 要件
- 日本語原文 + 韓国語人手翻訳のペア
- 1000文（層化サンプリング）
- ドメイン: 会話文（アニメ/ドラマ風）

### ソース候補
1. **AI Hub 日韓並列コーパス** (推奨)
   - 高品質な人手翻訳
   - ダウンロード要（アカウント必要）
   
2. **OPUS CCAligned** (代替)
   - Web crawlベース
   - 品質はAI Hubより劣る

3. **FLORES-200** (補助)
   - 200言語並列
   - 約1000文、高品質だがドメインが限定的

## パイプライン

```
1. extract_samples.py
   └─ AI Hubから(ja_src, ko_ref)ペア1000件抽出
   
2. translate_with_llm.py
   └─ 12条件で ja → ko_hyp を生成
   └─ 出力: translations/{model}-{strategy}.jsonl
   
3. compare_translations.py
   └─ 条件別に chrF++/BLEU/COMET/COMET-QE を算出
   └─ paired bootstrap で条件間の有意差検定
   └─ 出力: results/metrics.json
   
4. analyze_errors.py
   └─ 最良条件 vs 最悪条件 で50文サンプリング
   └─ エラータイプ分類:
      - 直訳 (literal): 文法的に正しいが不自然
      - 誤訳 (mistranslation): 意味が異なる
      - 不自然 (unnatural): 表現がおかしい
      - 情報欠落 (omission): 情報が落ちている
   └─ 出力: results/error_analysis.json
```

## 推定リソース

### 推論
- 1000文 × 12条件 = 12,000推論
- Qwen3-32B @ vLLM: ~2000 tokens/s → 約1-2時間
- API経由: ~$10-15

### 評価
- COMET: GPU必要、約10分
- COMET-QE: GPU必要、約10分
- chrF++/BLEU: CPU、数秒

## 成功基準

### Phase 1: パイプライン検証
- [ ] 100文でend-to-end動作確認
- [ ] 全指標が計算可能

### Phase 2: 本番評価
- [ ] 最良条件で chrF++ > 50
- [ ] COMET-QE と COMET の相関確認
- [ ] エラー分析で直訳率が低い条件を特定

### Phase 3: 意思決定
- [ ] 大規模生成に使うモデル/プロンプトを決定
- [ ] COMET-QEフィルタリング閾値を決定
- [ ] vLLMパラメータを最終決定

## ファイル構成

```
evaluation/
├── EXPERIMENT_DESIGN.md    # この設計書
├── data/
│   ├── ja_source.txt       # 日本語原文
│   └── ko_reference.txt    # 韓国語人手翻訳
├── translations/
│   ├── qwen3-32b-zero_shot.jsonl
│   ├── qwen3-32b-few_shot.jsonl
│   └── ...
├── results/
│   ├── metrics.json
│   └── error_analysis.json
└── scripts/
    ├── extract_samples.py
    ├── translate_with_llm.py
    ├── compare_translations.py
    └── analyze_errors.py
```
