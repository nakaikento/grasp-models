# 設計ドキュメント: mt-ja-ko

OPUS-MT-trainおよびHelsinki-NLPの知見を参考にした設計メモ。

---

## 1. モデルアーキテクチャ（OPUS-MT準拠）

Helsinki-NLPのMarianMTモデルは以下の構成：

```python
from transformers import MarianConfig

config = MarianConfig(
    # アーキテクチャ（OPUS-MT標準）
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    d_model=512,
    encoder_ffn_dim=2048,
    decoder_ffn_dim=2048,
    
    # OPUS-MT特有の設定
    static_position_embeddings=True,  # 静的sinusoidal位置埋め込み
    normalize_embedding=False,         # layernorm_embeddingなし
    add_bias_logits=True,              # final_logits_bias追加
    
    # トークナイザー設定
    vocab_size=32000,  # SentencePieceで作成
    max_position_embeddings=512,
    pad_token_id=0,
    eos_token_id=0,
    decoder_start_token_id=0,  # pad_token_idで開始（BARTの<s>とは異なる）
    
    # ドロップアウト
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.0,
    activation_function="swish",
)
```

### サイズ見積もり

| 設定 | パラメータ数 | ファイルサイズ |
|------|-------------|---------------|
| 6層, d_model=512 | ~74M | ~300MB (fp32) / ~150MB (fp16) |
| 6層, d_model=256 | ~20M | ~80MB (fp32) |

目標の100-300MBに収まる。

---

## 2. SentencePiece設定（OPUS-MT準拠）

OPUS-MTの標準設定を参考に：

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='data/cleaned/all.txt',  # 日韓両方を結合
    model_prefix='spm',
    
    # 語彙サイズ
    vocab_size=32000,  # OPUS-MT標準
    
    # モデルタイプ
    model_type='unigram',  # BPEではなくunigram（OPUS-MT標準）
    
    # 文字カバレッジ（日韓は文字種が多いので高めに）
    character_coverage=0.9995,
    
    # 特殊トークン（MarianMT互換）
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    
    # その他
    input_sentence_size=5000000,
    shuffle_input_sentence=True,
    byte_fallback=True,  # 未知文字対策
)
```

**注意**: OPUS-MTは言語ごとに別々のSPMを使う場合もあるが、日韓翻訳では共通vocabで問題ない。

---

## 3. 学習設定

### OPUS-MTの学習条件（参考値）

| 項目 | OPUS-MT設定 |
|------|------------|
| 最大学習時間 | 72時間 |
| GPU数 | 1〜4 |
| バッチサイズ | 動的（トークン数ベース） |
| Early stopping | validation perplexity |
| Validation data | Tatoeba等から自動選択 |

### 推奨設定（HuggingFace Trainer用）

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./models/ja-ko",
    
    # バッチ設定
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=4,  # 実効バッチ: 32*4=128
    
    # 学習率
    learning_rate=3e-4,
    lr_scheduler_type="inverse_sqrt",
    warmup_steps=4000,
    
    # エポック/ステップ
    num_train_epochs=10,
    # または max_steps=100000,
    
    # 評価・保存
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    
    # Early stopping
    # EarlyStoppingCallback(early_stopping_patience=5)
    
    # その他
    fp16=True,
    predict_with_generate=True,
    generation_max_length=128,
    logging_steps=100,
    report_to="wandb",  # または "tensorboard"
)
```

---

## 4. Knowledge Distillation設計

OPUS-MT-trainはKDをサポートしていないが、以下の戦略を採用：

### Phase 1: 教師データ生成

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# 教師モデル（M2M100 or NLLB-200）
teacher_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
teacher_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

def generate_teacher_translations(ja_texts, batch_size=32):
    """日本語テキストから教師翻訳を生成"""
    teacher_tokenizer.src_lang = "ja"
    translations = []
    
    for i in range(0, len(ja_texts), batch_size):
        batch = ja_texts[i:i+batch_size]
        inputs = teacher_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        
        generated = teacher_model.generate(
            **inputs,
            forced_bos_token_id=teacher_tokenizer.get_lang_id("ko"),
            max_length=128,
            num_beams=5,
        )
        
        decoded = teacher_tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(decoded)
    
    return translations
```

### Phase 2: 生徒モデル学習

```python
# 学習データ
# - 入力: 日本語（OPUS）
# - ターゲット: 韓国語（教師モデル出力）

train_dataset = Dataset.from_dict({
    "ja": ja_texts,
    "ko": teacher_translations,  # M2M100の出力
})
```

### 蒸留ロス（オプション）

標準的なクロスエントロピーでも十分だが、ソフトラベル蒸留も可能：

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=2.0):
    """
    ハード/ソフト ラベルの組み合わせ
    """
    # ハードラベルロス（通常のCE）
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # ソフトラベルロス（KL divergence）
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

---

## 5. Back-translation フィルタ（OPUS-MT参考）

OPUS-MT-trainのBack-translationは主にデータ拡張用だが、フィルタリングにも応用可能：

```python
def backtranslation_filter(ja_texts, ko_texts, threshold=0.5):
    """
    1. ko → ja に逆翻訳
    2. 元のjaとの類似度を計算
    3. 閾値以下を除去
    """
    # 逆翻訳モデル（ko→ja）が必要
    # 存在しない場合はLaBSE等の埋め込み類似度で代用
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    ja_emb = model.encode(ja_texts)
    ko_emb = model.encode(ko_texts)
    
    # コサイン類似度
    similarities = (ja_emb * ko_emb).sum(axis=1) / (
        np.linalg.norm(ja_emb, axis=1) * np.linalg.norm(ko_emb, axis=1)
    )
    
    # フィルタ
    mask = similarities >= threshold
    return [ja for ja, m in zip(ja_texts, mask) if m], \
           [ko for ko, m in zip(ko_texts, mask) if m]
```

---

## 6. 評価（OPUS-MT準拠）

OPUS-MTは**sacrebleu**を使用：

```python
import sacrebleu

def evaluate_bleu(predictions, references):
    """
    sacrebleuでBLEUスコアを計算
    """
    # referencesは2重リストにする
    refs = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions, refs)
    return bleu.score

# HuggingFace Trainer用
from datasets import load_metric

def compute_metrics(eval_preds):
    metric = load_metric("sacrebleu")
    preds, labels = eval_preds
    
    # デコード
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # BLEU計算
    result = metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )
    return {"bleu": result["score"]}
```

---

## 7. データ分割

OPUS-MTの方式を参考に：

```
全データ: 1,035,749行
    ├── train: 1,025,000行 (99%)
    ├── val:       5,000行 (0.5%)
    └── test:      5,749行 (0.5%)
```

```python
from sklearn.model_selection import train_test_split

# 最初にtest分割
train_val, test = train_test_split(data, test_size=5749, random_state=42)

# trainとvalを分割
train, val = train_test_split(train_val, test_size=5000, random_state=42)
```

---

## 8. ディレクトリ構成（更新版）

```
mt-ja-ko/
├── data/
│   ├── raw/                    # OPUSデータ
│   ├── cleaned/                # クレンジング済み
│   ├── tokenized/              # SentencePieceモデル
│   └── splits/                 # train/val/test分割
│       ├── train.ja
│       ├── train.ko
│       ├── val.ja
│       ├── val.ko
│       ├── test.ja
│       └── test.ko
├── scripts/
│   ├── clean.py
│   ├── train_tokenizer.py
│   ├── split_data.py           # 新規
│   └── analyze_data.py
├── training/
│   ├── config.py               # モデル設定
│   ├── train.py                # 学習スクリプト
│   ├── distill.py              # 蒸留スクリプト
│   └── evaluate.py             # 評価スクリプト
├── export/
│   └── to_onnx.py
└── models/
```

---

## 9. 参考文献

- [OPUS-MT-train](https://github.com/Helsinki-NLP/OPUS-MT-train)
- [MarianMT HuggingFace](https://huggingface.co/docs/transformers/model_doc/marian)
- [Democratizing NMT with OPUS-MT (Tiedemann et al., 2023)](https://link.springer.com/article/10.1007/s10579-023-09704-w)
