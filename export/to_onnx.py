#!/usr/bin/env python3
"""
ONNX エクスポートスクリプト

学習済みMarianMTモデルをONNX形式に変換

Usage:
    python export/to_onnx.py
    python export/to_onnx.py --model-dir models/ja-ko --output models/ja-ko/model.onnx

Output:
    model.onnx (エンコーダー + デコーダー統合)
    または
    encoder.onnx + decoder.onnx (分離)
"""

import torch
from pathlib import Path
import argparse
from transformers import MarianMTModel, MarianConfig
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_encoder(model, output_path: Path, opset_version: int = 14):
    """エンコーダーをエクスポート"""
    print("エンコーダーをエクスポート中...")
    
    encoder = model.get_encoder()
    
    # ダミー入力
    dummy_input = torch.randint(0, 1000, (1, 32))
    attention_mask = torch.ones_like(dummy_input)
    
    torch.onnx.export(
        encoder,
        (dummy_input, attention_mask),
        str(output_path),
        input_names=['input_ids', 'attention_mask'],
        output_names=['encoder_hidden_states'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'encoder_hidden_states': {0: 'batch', 1: 'sequence'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"  保存: {output_path}")


def export_decoder(model, output_path: Path, opset_version: int = 14):
    """デコーダーをエクスポート（with encoder hidden states）"""
    print("デコーダーをエクスポート中...")
    
    # ダミー入力
    batch_size = 1
    src_len = 32
    tgt_len = 1
    d_model = model.config.d_model
    
    decoder_input_ids = torch.randint(0, 1000, (batch_size, tgt_len))
    encoder_hidden_states = torch.randn(batch_size, src_len, d_model)
    encoder_attention_mask = torch.ones(batch_size, src_len)
    
    # デコーダーを取得
    decoder = model.model.decoder
    
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.embed_tokens = model.model.decoder.embed_tokens
            self.decoder = model.model.decoder
            self.lm_head = model.lm_head
            
        def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            logits = self.lm_head(decoder_outputs.last_hidden_state)
            return logits
    
    wrapper = DecoderWrapper(model)
    wrapper.eval()
    
    torch.onnx.export(
        wrapper,
        (decoder_input_ids, encoder_hidden_states, encoder_attention_mask),
        str(output_path),
        input_names=['decoder_input_ids', 'encoder_hidden_states', 'encoder_attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'decoder_input_ids': {0: 'batch', 1: 'target_sequence'},
            'encoder_hidden_states': {0: 'batch', 1: 'source_sequence'},
            'encoder_attention_mask': {0: 'batch', 1: 'source_sequence'},
            'logits': {0: 'batch', 1: 'target_sequence'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"  保存: {output_path}")


def quantize_model(input_path: Path, output_path: Path):
    """INT8量子化"""
    print(f"量子化中: {input_path} -> {output_path}")
    
    quantize_dynamic(
        str(input_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )
    
    # サイズ比較
    original_size = input_path.stat().st_size / 1024 / 1024
    quantized_size = output_path.stat().st_size / 1024 / 1024
    print(f"  元サイズ: {original_size:.1f} MB")
    print(f"  量子化後: {quantized_size:.1f} MB ({quantized_size/original_size*100:.0f}%)")


def verify_onnx(model_path: Path):
    """ONNXモデルを検証"""
    print(f"検証中: {model_path}")
    
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    
    print("  ✓ 検証OK")


def main():
    parser = argparse.ArgumentParser(description="ONNXエクスポート")
    parser.add_argument("--model-dir", type=str, default="models/ja-ko")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quantize", action="store_true", help="INT8量子化を適用")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()
    
    print("=" * 50)
    print("ONNX エクスポート")
    print("=" * 50)
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # モデルロード
    print(f"\nモデルをロード: {model_dir}")
    
    # config.jsonがある場合はHuggingFace形式
    if (model_dir / "config.json").exists():
        model = MarianMTModel.from_pretrained(model_dir)
    else:
        # PyTorchチェックポイントの場合
        raise NotImplementedError("PyTorchチェックポイントからのロードは未実装")
    
    model.eval()
    
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # エクスポート
    encoder_path = output_dir / "encoder.onnx"
    decoder_path = output_dir / "decoder.onnx"
    
    export_encoder(model, encoder_path, args.opset)
    export_decoder(model, decoder_path, args.opset)
    
    # 検証
    verify_onnx(encoder_path)
    verify_onnx(decoder_path)
    
    # 量子化（オプション）
    if args.quantize:
        print("\n量子化...")
        quantize_model(encoder_path, output_dir / "encoder_int8.onnx")
        quantize_model(decoder_path, output_dir / "decoder_int8.onnx")
    
    # サイズ表示
    print("\n出力ファイル:")
    for f in output_dir.glob("*.onnx"):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")
    
    print("\n完了！")


if __name__ == "__main__":
    main()
