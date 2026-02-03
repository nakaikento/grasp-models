#!/usr/bin/env python3
"""
ONNX変換スクリプト（Optimum使用）

学習済みMarianMTモデルをONNX形式に変換（Android用）

Usage:
    # 韓日翻訳モデル
    python scripts/convert_to_onnx.py --model-dir models/ko-ja
    
    # 日韓翻訳モデル
    python scripts/convert_to_onnx.py --model-dir models/ja-ko

Output:
    models/ko-ja-onnx/
      encoder_model.onnx
      decoder_model.onnx
      decoder_with_past_model.onnx
      spm.model
"""

import argparse
import shutil
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM


def convert_to_onnx(model_dir: Path, output_dir: Path = None):
    """モデルをONNXに変換"""
    
    print("=" * 50)
    print(f"ONNX変換: {model_dir}")
    print("=" * 50)
    
    # 出力ディレクトリ
    if output_dir is None:
        output_dir = model_dir.parent / f"{model_dir.name}-onnx"
    
    output_dir = Path(output_dir)
    
    # 既存のONNXディレクトリがある場合は削除
    if output_dir.exists():
        print(f"⚠️  既存のONNXディレクトリを削除: {output_dir}")
        shutil.rmtree(output_dir)
    
    # 変換
    print(f"\n変換中...")
    print(f"  入力: {model_dir}")
    print(f"  出力: {output_dir}")
    
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(
        str(model_dir),
        export=True
    )
    
    # 保存
    print(f"\n保存中...")
    ort_model.save_pretrained(str(output_dir))
    
    # トークナイザーをコピー
    spm_src = model_dir / "spm.model"
    if spm_src.exists():
        spm_dst = output_dir / "spm.model"
        shutil.copy(spm_src, spm_dst)
        print(f"✓ トークナイザーをコピー: {spm_dst.name}")
    else:
        print(f"⚠️  トークナイザーが見つかりません: {spm_src}")
    
    # 結果表示
    print(f"\n{'='*50}")
    print(f"✅ ONNX変換完了！")
    print(f"{'='*50}")
    print(f"出力: {output_dir}")
    print(f"\nファイル一覧:")
    
    total_size = 0
    for f in sorted(output_dir.glob("*.onnx")):
        size_mb = f.stat().st_size / 1024 / 1024
        total_size += size_mb
        print(f"  {f.name}: {size_mb:.1f} MB")
    
    if (output_dir / "spm.model").exists():
        spm_size = (output_dir / "spm.model").stat().st_size / 1024
        print(f"  spm.model: {spm_size:.1f} KB")
    
    print(f"\n合計サイズ: {total_size:.1f} MB")
    
    # 次のステップ
    print(f"\n次のステップ:")
    print(f"  量子化: python scripts/quantize_onnx.py --model-dir {output_dir}")
    print(f"  ZIP作成: cd {output_dir} && zip -r ../{output_dir.name}.zip *.onnx spm.model")


def main():
    parser = argparse.ArgumentParser(description="ONNX変換（Optimum使用）")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="学習済みモデルディレクトリ（PyTorch）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="出力ディレクトリ（デフォルト: {model-dir}-onnx）")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # 存在確認
    if not model_dir.exists():
        print(f"❌ モデルディレクトリが見つかりません: {model_dir}")
        return 1
    
    if not (model_dir / "config.json").exists():
        print(f"❌ config.jsonが見つかりません: {model_dir / 'config.json'}")
        return 1
    
    # 変換
    try:
        convert_to_onnx(model_dir, output_dir)
        return 0
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
