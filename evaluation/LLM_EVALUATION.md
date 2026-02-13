# LLM翻訳品質検証パイプライン

## 目的
Qwen3-32B / DeepSeek-R1で、小規模モデル（Qwen2.5-7B等）の直訳問題を克服できるかを定量検証する。

## パイプライン概要

```
1. サンプル抽出 (extract_samples.py)
   └─ OpenSubtitlesから1000文を層化抽出
   
2. LLM翻訳 (translate_with_llm.py)
   └─ 複数モデル × 複数プロンプト戦略
   
3. 評価・比較 (compare_translations.py)
   └─ chrF++ / BLEU / COMET で比較
```

## ステータス
- [x] サンプル抽出スクリプト作成
- [x] LLM翻訳スクリプト作成
- [x] 評価比較スクリプト作成
- [x] 1000サンプル抽出完了
- [ ] LLM翻訳実行 (APIキー必要)
- [ ] 評価実行
- [ ] 結果分析

## サンプル統計
```
サンプル数: 1000
韓国語平均長: 37.5文字
韓国語最小/最大: 10/94文字
日本語平均長: 22.0文字
```

## 検証モデル

| モデル | プロバイダー | 推定コスト (1000文) |
|--------|-------------|---------------------|
| Qwen3-32B | OpenRouter/Together | ~$0.50 |
| Qwen3-235B-A22B | OpenRouter | ~$2.00 |
| DeepSeek-R1 | OpenRouter | ~$1.50 |

## プロンプト戦略

1. **baseline**: シンプルな翻訳指示
2. **literal**: 直訳指示
3. **natural**: 自然な意訳指示
4. **thinking**: Thinking Mode（文化的コンテキスト考慮）
5. **few_shot**: Few-shot例付き

## 実行方法

### 1. サンプル抽出（完了済み）
```bash
python3 scripts/extract_samples.py --n-samples 1000
```

### 2. LLM翻訳
```bash
# OpenRouter経由
export OPENROUTER_API_KEY="your-key"

# Qwen3-32B + natural戦略
python3 scripts/translate_with_llm.py \
  --input samples/source_ko.txt \
  --output translations/qwen3-32b-natural.txt \
  --provider openrouter \
  --model "qwen/qwen3-32b" \
  --strategy natural

# DeepSeek-R1
python3 scripts/translate_with_llm.py \
  --input samples/source_ko.txt \
  --output translations/deepseek-r1-natural.txt \
  --provider openrouter \
  --model "deepseek/deepseek-r1" \
  --strategy natural
```

### 3. 評価
```bash
python3 scripts/compare_translations.py \
  --source samples/source_ko.txt \
  --reference samples/reference_ja.txt \
  --translations translations/*.txt \
  --output results/comparison.json
```

## 期待される結果

### 成功基準
- **chrF++ > 50**: 実用品質
- **chrF++ > 60**: 高品質
- モデルサイズによる品質差が明確に測定できること

### 検証ポイント
1. 32B以上で直訳問題が解消されるか
2. Thinking Modeの効果
3. Few-shotの効果
4. プロンプト戦略間の差

## 次のステップ

品質が十分（chrF++ > 50）であれば：
1. vLLMでローカル推論セットアップ
2. 大規模データ生成パイプライン構築
3. 蒸留用データセット作成

## ファイル構成
```
evaluation/
├── LLM_EVALUATION.md     # このファイル
├── samples/
│   ├── source_ko.txt     # 韓国語ソース (1000行)
│   ├── reference_ja.txt  # 日本語リファレンス (1000行)
│   └── stats.txt         # 統計情報
├── translations/         # LLM翻訳結果
├── results/              # 評価結果
└── scripts/
    ├── extract_samples.py
    ├── translate_with_llm.py
    └── compare_translations.py
```
