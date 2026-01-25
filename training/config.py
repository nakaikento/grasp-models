#!/usr/bin/env python3
"""
MarianMT 学習設定

RunPod RTX 4090 (24GB VRAM, 62GB RAM, 16 vCPU) 用に最適化
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class ModelConfig:
    """MarianMTモデル設定"""
    
    # アーキテクチャ（Transformer-Base相当）
    encoder_layers: int = 6
    decoder_layers: int = 6
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    d_model: int = 512
    encoder_ffn_dim: int = 2048
    decoder_ffn_dim: int = 2048
    
    # 位置エンコーディング
    max_position_embeddings: int = 512
    static_position_embeddings: bool = True
    
    # 正則化
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    
    # 活性化関数
    activation_function: str = "swish"
    
    # 特殊トークン
    pad_token_id: int = 0
    eos_token_id: int = 3
    decoder_start_token_id: int = 0
    
    # シーケンス長
    max_length: int = 128


@dataclass
class TrainingConfig:
    """学習設定（RunPod RTX 4090最適化）"""
    
    # パス
    output_dir: Path = Path("models/ja-ko")
    tokenizer_path: Path = Path("data/tokenized/spm.model")
    data_dir: Path = Path("data/splits")
    
    # バッチサイズ（RTX 4090 24GB VRAM）
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 2  # 実効バッチサイズ = 64 x 2 = 128
    
    # 学習率
    learning_rate: float = 3e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 4000
    weight_decay: float = 0.01
    
    # エポック
    num_train_epochs: int = 10
    
    # 評価・保存
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "bleu"
    greater_is_better: bool = True
    
    # 生成（評価用）
    generation_max_length: int = 128
    generation_num_beams: int = 4
    
    # 高速化
    fp16: bool = True  # RTX 4090で効果的
    dataloader_num_workers: int = 8  # 16 vCPU
    
    # ロギング
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    
    # Early Stopping
    early_stopping_patience: int = 3


@dataclass
class DistillationConfig:
    """Knowledge Distillation設定"""
    
    # 教師モデル
    teacher_model: str = "facebook/nllb-200-distilled-600M"
    
    # 言語設定
    src_lang: str = "jpn_Jpan"  # NLLB形式
    tgt_lang: str = "kor_Hang"
    
    # 生成設定
    teacher_batch_size: int = 32
    teacher_max_length: int = 128
    teacher_num_beams: int = 5
    
    # 出力
    teacher_output_dir: Path = Path("data/teacher")


# 小規模モデル設定（テスト用）
@dataclass
class SmallModelConfig(ModelConfig):
    """軽量モデル設定（デバッグ・テスト用）"""
    encoder_layers: int = 3
    decoder_layers: int = 3
    d_model: int = 256
    encoder_ffn_dim: int = 1024
    decoder_ffn_dim: int = 1024


# Colab用設定
@dataclass  
class ColabTrainingConfig(TrainingConfig):
    """Google Colab (T4 16GB) 用設定"""
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    dataloader_num_workers: int = 2
    save_steps: int = 2000
    eval_steps: int = 2000