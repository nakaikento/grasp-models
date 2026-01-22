#!/usr/bin/env python3
"""
教師データ生成スクリプト（途中保存対応版）

M2M100 or NLLB-200を使用して、日本語から高品質な韓国語翻訳を生成

Usage:
    python training/generate_teacher_data.py
    python training/generate_teacher_data.py --resume  # 途中から再開

Input:  data/splits/train.ja
Output: data/teacher/train.ko (教師翻訳)
"""

import torch
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
import argparse

# 設定をインポート
import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.config import DistillationConfig


@dataclass
class GenerationArgs:
    input_file: Path = Path("data/splits/train.ja")
    output_file: Path = Path("data/teacher/train.ko")
    model_name: str = "facebook/nllb-200-1.3B"
    batch_size: int = 16
    max_length: int = 128
    num_beams: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume_from: int = 0  # 途中から再開する場合


def load_model(model_name: str, device: str):
    """教師モデルをロード"""
    print(f"モデルをロード中: {model_name}")
    
    # accelerateを使わずにシンプルにロード
    import os
    os.environ["ACCELERATE_USE_SAFETENSORS"] = "true"
    
    if "nllb" in model_name.lower():
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # シンプルにロード（device_mapなし）
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        if device == "cuda":
            model = model.half()  # GPU時はfp16
        model = model.to(device)
        
        src_lang = "jpn_Jpan"
        tgt_lang = "kor_Hang"
        
    elif "m2m100" in model_name.lower():
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        
        model = M2M100ForConditionalGeneration.from_pretrained(
            model_name,
            use_safetensors=True,
            low_cpu_mem_usage=False,
        )
        if device == "cuda":
            model = model.half()
        model = model.to(device)
        
        tokenizer.src_lang = "ja"
        src_lang = "ja"
        tgt_lang = "ko"
    else:
        raise ValueError(f"未対応のモデル: {model_name}")
    
    model.eval()
    
    return model, tokenizer, src_lang, tgt_lang


def count_existing_lines(output_path: Path) -> int:
    """既存の出力ファイルの行数をカウント"""
    if not output_path.exists():
        return 0
    with open(output_path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def generate_translations(
    model,
    tokenizer,
    texts: list,
    tgt_lang: str,
    batch_size: int,
    max_length: int,
    num_beams: int,
    device: str,
    model_name: str,
    output_path: Path,
    save_every: int = 100,  # 100バッチごとに保存
    start_idx: int = 0,
):
    """バッチ処理で翻訳を生成（途中保存対応）"""
    translations = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # 出力ファイルを追記モードで開く準備
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for batch_idx, i in enumerate(tqdm(range(0, len(texts), batch_size), desc="翻訳生成中")):
        batch = texts[i:i + batch_size]
        
        # トークナイズ
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # 生成
        with torch.no_grad():
            if "nllb" in model_name.lower():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )
            else:  # M2M100
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )
        
        # デコード
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(decoded)
        
        # メモリ解放
        del inputs, generated
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # 定期保存
        if (batch_idx + 1) % save_every == 0:
            # 追記モードで保存
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(translations) + '\n')
            
            total_saved = start_idx + (batch_idx + 1) * batch_size
            print(f"\n💾 途中保存: {total_saved:,}行 ({(batch_idx + 1) / total_batches * 100:.1f}%)")
            
            # メモリ解放
            translations = []
    
    # 残りを保存
    if translations:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(translations) + '\n')
    
    return len(texts)


def main():
    parser = argparse.ArgumentParser(description="教師データ生成")
    parser.add_argument("--input", type=str, default="data/splits/train.ja")
    parser.add_argument("--output", type=str, default="data/teacher/train.ko")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-1.3B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="途中から再開")
    parser.add_argument("--save-every", type=int, default=100, help="何バッチごとに保存するか")
    args = parser.parse_args()
    
    print("=" * 50)
    print("教師データ生成")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")
    
    # モデルロード
    model, tokenizer, src_lang, tgt_lang = load_model(args.model, device)
    print(f"言語ペア: {src_lang} → {tgt_lang}")
    
    # 入力読み込み
    input_path = Path(args.input)
    output_path = Path(args.output)
    print(f"\n入力ファイル: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        ja_texts = [line.strip() for line in f]
    
    total_lines = len(ja_texts)
    print(f"入力行数: {total_lines:,}")
    
    # 途中から再開
    start_idx = 0
    if args.resume:
        start_idx = count_existing_lines(output_path)
        if start_idx > 0:
            print(f"\n🔄 再開モード: 既存 {start_idx:,}行 を検出")
            print(f"   行 {start_idx + 1} から再開します")
            ja_texts = ja_texts[start_idx:]
        else:
            print("\n既存ファイルなし。最初から開始します。")
    else:
        # 新規開始の場合は既存ファイルをクリア
        if output_path.exists():
            output_path.unlink()
    
    if not ja_texts:
        print("\n✅ すべて完了済みです！")
        return
    
    # 翻訳生成
    print(f"\n翻訳生成開始...")
    print(f"  バッチサイズ: {args.batch_size}")
    print(f"  ビーム数: {args.num_beams}")
    print(f"  保存間隔: {args.save_every}バッチごと")
    print(f"  残り: {len(ja_texts):,}行")
    
    num_generated = generate_translations(
        model=model,
        tokenizer=tokenizer,
        texts=ja_texts,
        tgt_lang=tgt_lang,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_beams=args.num_beams,
        device=device,
        model_name=args.model,
        output_path=output_path,
        save_every=args.save_every,
        start_idx=start_idx,
    )
    
    # 最終確認
    final_count = count_existing_lines(output_path)
    print(f"\n✅ 保存完了: {output_path}")
    print(f"   総行数: {final_count:,} / {total_lines:,}")
    
    if final_count >= total_lines:
        print("\n🎉 すべての翻訳が完了しました！")
    else:
        print(f"\n⚠️  残り {total_lines - final_count:,}行")
        print("   --resume オプションで再開できます")


if __name__ == "__main__":
    main()