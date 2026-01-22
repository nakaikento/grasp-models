"""
モデル・学習設定

OPUS-MT準拠のMarianMT設定
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """MarianMT モデル設定（OPUS-MT準拠）"""
    
    # アーキテクチャ
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    d_model: int = 512
    encoder_ffn_dim: int = 2048
    decoder_ffn_dim: int = 2048
    
    # OPUS-MT特有
    static_position_embeddings: bool = True
    normalize_embedding: bool = False
    add_bias_logits: bool = True
    
    # トークナイザー
    vocab_size: int = 32000
    max_position_embeddings: int = 512
    max_length: int = 128
    
    # 特殊トークン
    pad_token_id: int = 0
    eos_token_id: int = 3
    decoder_start_token_id: int = 0
    
    # ドロップアウト
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    activation_function: str = "swish"


@dataclass
class TrainingConfig:
    """学習設定"""
    
    # パス
    data_dir: Path = Path("data/splits")
    tokenizer_path: Path = Path("data/tokenized/spm.model")
    output_dir: Path = Path("models/ja-ko")
    
    # バッチ設定
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 4  # 実効バッチ: 32*4=128
    
    # 学習率
    learning_rate: float = 3e-4
    lr_scheduler_type: str = "inverse_sqrt"
    warmup_steps: int = 4000
    weight_decay: float = 0.01
    
    # エポック/ステップ
    num_train_epochs: int = 10
    max_steps: int = -1  # -1 = エポックベース
    
    # 評価・保存
    eval_strategy: str = "steps"
    eval_steps: int = 2000
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "bleu"
    greater_is_better: bool = True
    
    # Early stopping
    early_stopping_patience: int = 5
    
    # その他
    fp16: bool = True
    bf16: bool = False  # A100ならTrue推奨
    dataloader_num_workers: int = 4
    logging_steps: int = 100
    report_to: str = "wandb"  # or "tensorboard"
    
    # 生成設定（評価用）
    generation_max_length: int = 128
    generation_num_beams: int = 4


@dataclass
class DistillationConfig:
    """Knowledge Distillation設定"""
    
    # 教師モデル
    teacher_model_name: str = "facebook/nllb-200-1.3B"  # or "facebook/m2m100_1.2B"
    teacher_src_lang: str = "jpn_Jpan"  # NLLB形式
    teacher_tgt_lang: str = "kor_Hang"  # NLLB形式
    
    # M2M100の場合
    # teacher_model_name: str = "facebook/m2m100_1.2B"
    # teacher_src_lang: str = "ja"
    # teacher_tgt_lang: str = "ko"
    
    # 教師データ生成
    teacher_batch_size: int = 16
    teacher_max_length: int = 128
    teacher_num_beams: int = 5
    
    # 蒸留設定
    use_soft_labels: bool = False  # True = KLロスも使用
    temperature: float = 2.0
    alpha: float = 0.5  # soft/hard ラベルの比率
    
    # 出力
    teacher_output_dir: Path = Path("data/teacher")


@dataclass 
class AWSConfig:
    """AWS環境設定"""
    
    # インスタンス
    instance_type: str = "g4dn.xlarge"  # T4 GPU x1
    # instance_type: str = "g5.xlarge"  # A10G GPU x1（高速）
    # instance_type: str = "p3.2xlarge"  # V100 GPU x1（高性能）
    
    # S3
    s3_bucket: str = "your-bucket-name"
    s3_data_prefix: str = "mt-ja-ko/data"
    s3_model_prefix: str = "mt-ja-ko/models"
    
    # Spot Instance（コスト削減）
    use_spot: bool = True
    spot_max_price: float = 0.5  # $/hour


# デフォルト設定のエクスポート
def get_model_config() -> ModelConfig:
    return ModelConfig()

def get_training_config() -> TrainingConfig:
    return TrainingConfig()

def get_distillation_config() -> DistillationConfig:
    return DistillationConfig()
