#!/usr/bin/env python3
"""
ONNX量子化スクリプト

ONNXモデルをINT8量子化してサイズを削減

Usage:
    python scripts/quantize_onnx.py --model-dir models/ko-ja-onnx

Output:
    models/ko-ja-onnx-quantized/
      encoder_model_quantized.onnx
      decoder_model_quantized.onnx
      decoder_with_past_model_quantized.onnx
"""

import argparse
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_model(input_path: Path, output_path: Path):
    """単一のONNXモデルを量子化"""
    
    print(f"  量子化中: {input_path.name}")
    
    quantize_dynamic(
        str(input_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )
    
    # サイズ比較
    original_size = input_path.stat().st_size / 1024 / 1024
    quantized_size = output_path.stat().st_size / 1024 / 1024
    ratio = quantized_size / original_size * 100
    
    print(f"    元: {original_size:.1f} MB → 量子化後: {quantized_size:.1f} MB ({ratio:.0f}%)")


def quantize_all(model_dir: Path, output_dir: Path = None):
    """ディレクトリ内のすべてのONNXモデルを量子化"""
    
    print("=" * 50)
    print(f"ONNX量子化: {model_dir}")
    print("=" * 50)
    
    # 出力ディレクトリ
    if output_dir is None:
        if model_dir.name.endswith("-onnx"):
            output_dir = model_dir.parent / f"{model_dir.name}-quantized"
        else:
            output_dir = model_dir.parent / f"{model_dir.name}-quantized"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ONNXファイルを検索
    onnx_files = list(model_dir.glob("*.onnx"))
    
    if not onnx_files:
        print(f"❌ ONNXファイルが見つかりません: {model_dir}")
        return 1
    
    print(f"\n{len(onnx_files)} 個のONNXファイルを量子化...")
    print()
    
    total_original = 0
    total_quantized = 0
    
    for onnx_file in sorted(onnx_files):
        # 既に量子化済みのファイルはスキップ
        if "quantized" in onnx_file.name:
            continue
        
        output_file = output_dir / onnx_file.name.replace(".onnx", "_quantized.onnx")
        
        quantize_model(onnx_file, output_file)
        
        total_original += onnx_file.stat().st_size / 1024 / 1024
        total_quantized += output_file.stat().st_size / 1024 / 1024
    
    # その他のファイル（spm.model等）をコピー
    print(f"\nその他のファイルをコピー...")
    for f in model_dir.iterdir():
        if f.is_file() and not f.name.endswith(".onnx"):
            dest = output_dir / f.name
            import shutil
            shutil.copy(f, dest)
            print(f"  ✓ {f.name}")
    
    # 結果表示
    ratio = total_quantized / total_original * 100 if total_original > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"✅ 量子化完了！")
    print(f"{'='*50}")
    print(f"出力: {output_dir}")
    print(f"\n合計サイズ:")
    print(f"  元: {total_original:.1f} MB")
    print(f"  量子化後: {total_quantized:.1f} MB ({ratio:.0f}%)")
    print(f"  削減: {total_original - total_quantized:.1f} MB")
    
    # 次のステップ
    print(f"\n次のステップ:")
    print(f"  ZIP作成: cd {output_dir} && zip -r ../{output_dir.name}.zip *_quantized.onnx spm.model")
    print(f"  GitHub Release: gh release create v2.0.0 {output_dir.parent / (output_dir.name + '.zip')}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="ONNX量子化")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="ONNXモデルディレクトリ")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="出力ディレクトリ（デフォルト: {model-dir}-quantized）")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # 存在確認
    if not model_dir.exists():
        print(f"❌ ディレクトリが見つかりません: {model_dir}")
        return 1
    
    # 量子化
    try:
        return quantize_all(model_dir, output_dir)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
