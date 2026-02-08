#!/usr/bin/env python3
"""
ビームサーチのテスト用スクリプト

ONNXモデルを直接ロードして翻訳をテスト
"""

import onnxruntime as ort
import numpy as np
import sentencepiece as spm
from pathlib import Path

# モデルパス
MODEL_DIR = Path("models/ko-ja-onnx-int8")
ENCODER_PATH = MODEL_DIR / "encoder_model_quantized.onnx"
DECODER_PATH = MODEL_DIR / "decoder_model_quantized.onnx"
SPM_PATH = MODEL_DIR / "spm.model"

# 定数
PAD_ID = 0
EOS_ID = 3
MAX_LENGTH = 128
NUM_BEAMS = 4

print("=" * 60)
print("ONNX Beam Search Test")
print("=" * 60)

# トークナイザーロード
print(f"\n1. Loading tokenizer from {SPM_PATH}...")
sp = spm.SentencePieceProcessor()
sp.Load(str(SPM_PATH))

print(f"   Vocab size: {sp.GetPieceSize()}")
print(f"   PAD ID: {sp.pad_id()}")
print(f"   EOS ID: {sp.eos_id()}")
print(f"   BOS ID: {sp.bos_id()}")
print(f"   UNK ID: {sp.unk_id()}")

# ONNX Sessions
print(f"\n2. Loading ONNX models...")
encoder_session = ort.InferenceSession(str(ENCODER_PATH))
decoder_session = ort.InferenceSession(str(DECODER_PATH))
print(f"   Encoder loaded: {ENCODER_PATH.name}")
print(f"   Decoder loaded: {DECODER_PATH.name}")

# Encoder入力確認
print(f"\n3. Encoder inputs:")
for inp in encoder_session.get_inputs():
    print(f"   - {inp.name}: {inp.shape} ({inp.type})")

# Decoder入力確認
print(f"\n4. Decoder inputs:")
for inp in decoder_session.get_inputs():
    print(f"   - {inp.name}: {inp.shape} ({inp.type})")

# テスト翻訳
def translate(text, decoder_start_token=None):
    if decoder_start_token is None:
        decoder_start_token = PAD_ID
    
    print(f"\n{'=' * 60}")
    print(f"Translating: {text} (decoder_start_token={decoder_start_token})")
    print(f"{'=' * 60}")
    
    # エンコード
    input_ids = sp.EncodeAsIds(text) + [sp.eos_id()]
    print(f"Input tokens: {input_ids}")
    
    # Encoder
    encoder_inputs = {
        "input_ids": np.array([input_ids], dtype=np.int64),
        "attention_mask": np.ones((1, len(input_ids)), dtype=np.int64)
    }
    
    encoder_outputs = encoder_session.run(None, encoder_inputs)
    encoder_hidden_states = encoder_outputs[0]
    print(f"Encoder output shape: {encoder_hidden_states.shape}")
    
    # Greedy decoding (まずシンプルに)
    output_ids = [decoder_start_token]
    
    for step in range(MAX_LENGTH):
        decoder_inputs = {
            "input_ids": np.array([output_ids], dtype=np.int64),
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": np.ones((1, len(input_ids)), dtype=np.int64)
        }
        
        decoder_outputs = decoder_session.run(None, decoder_inputs)
        logits = decoder_outputs[0]  # [1, seq_len, vocab_size]
        
        # 最後のトークンのlogitsを取得
        last_logits = logits[0, -1, :]
        next_token = int(np.argmax(last_logits))
        
        if step < 5:
            print(f"Step {step}: token={next_token}, max_logit={last_logits[next_token]:.4f}")
        
        if next_token == sp.eos_id():
            print(f"Step {step}: EOS detected")
            break
        
        output_ids.append(next_token)
        
        if len(output_ids) > MAX_LENGTH:
            break
    
    # デコード
    result_ids = output_ids[1:]  # PAD_IDを除く
    if sp.eos_id() in result_ids:
        result_ids = result_ids[:result_ids.index(sp.eos_id())]
    
    result = sp.DecodeIds(result_ids)
    
    print(f"\nResult tokens: {result_ids}")
    print(f"Translation: {result}")
    
    return result


# テスト実行
test_texts = [
    "안녕하세요",
    "날 믿어",
    "거짓말하지 마",
    "저는 일본 사람이에요"
]

print(f"\n{'=' * 60}")
print("TEST 1: decoder_start_token = PAD (0)")
print(f"{'=' * 60}")

for text in test_texts:
    translate(text, decoder_start_token=PAD_ID)

print(f"\n{'=' * 60}")
print("TEST 2: decoder_start_token = EOS (3)")
print(f"{'=' * 60}")

for text in test_texts:
    translate(text, decoder_start_token=EOS_ID)

print(f"\n{'=' * 60}")
print("TEST 3: decoder_start_token = BOS (2)")
print(f"{'=' * 60}")

for text in test_texts:
    translate(text, decoder_start_token=sp.bos_id())

print(f"\n{'=' * 60}")
print("Test completed!")
print(f"{'=' * 60}")
