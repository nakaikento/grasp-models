import torch
import re
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# NLLB言語コードマッピング
LANG_CODES = {
    "ja": "jpn_Jpan",
    "ko": "kor_Hang"
}

# 言語検出用パターン
JP_PATTERN = re.compile(r'[ぁ-んァ-ヶ一-龠]')
KO_PATTERN = re.compile(r'[가-힣]')

def contains_language(text, lang_code):
    """指定された言語が含まれているかチェック"""
    if lang_code == "ja":
        return bool(JP_PATTERN.search(text))
    elif lang_code == "ko":
        return bool(KO_PATTERN.search(text))
    return False

def clean_input(text):
    """モデルが混乱しやすい記号を一時的に除去"""
    t = text.replace('・・', '')
    t = t.replace('・', '')
    t = re.sub(r'^[-ー－]\s*', '', t)  # 文頭のハイフン等を除去
    return t.strip()

def main():
    parser = argparse.ArgumentParser(description="NLLB-200を使用した教師データ生成（汎用版）")
    
    # 言語設定
    parser.add_argument("--src_lang", type=str, required=True, 
                        choices=["ja", "ko"],
                        help="ソース言語 (ja: 日本語, ko: 韓国語)")
    parser.add_argument("--tgt_lang", type=str, required=True,
                        choices=["ja", "ko"],
                        help="ターゲット言語 (ja: 日本語, ko: 韓国語)")
    
    # ファイルパス
    parser.add_argument("--src_file", type=str, required=True,
                        help="入力ファイルパス (例: data/raw/OpenSubtitles.ja-ko.ja)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="出力ファイルパス (例: data/teacher/train.ko)")
    
    # モデル設定
    parser.add_argument("--model_name", type=str, 
                        default="facebook/nllb-200-3.3b",
                        help="NLLBモデル名")
    parser.add_argument("--batch_size", type=int, default=40,
                        help="バッチサイズ")
    parser.add_argument("--num_beams", type=int, default=3,
                        help="ビームサーチの幅")
    parser.add_argument("--max_length", type=int, default=128,
                        help="最大トークン長")
    parser.add_argument("--sample_interval", type=int, default=5000,
                        help="進捗表示の間隔")
    
    args = parser.parse_args()
    
    # 言語コード取得
    src_lang_code = LANG_CODES[args.src_lang]
    tgt_lang_code = LANG_CODES[args.tgt_lang]
    
    print(f"🚀 設定:")
    print(f"  ソース言語: {args.src_lang} ({src_lang_code})")
    print(f"  ターゲット言語: {args.tgt_lang} ({tgt_lang_code})")
    print(f"  入力ファイル: {args.src_file}")
    print(f"  出力ファイル: {args.output_file}")
    print(f"  モデル: {args.model_name}")
    print(f"  バッチサイズ: {args.batch_size}, ビーム: {args.num_beams}")
    print()
    
    # モデルロード
    print(f"🚀 超高品質モード(16-bit + Beam{args.num_beams} + Cleaning)でロード中...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
    
    # 入力データ読み込み
    print(f"📖 入力ファイル読み込み中: {args.src_file}")
    if not os.path.exists(args.src_file):
        raise FileNotFoundError(f"入力ファイルが見つかりません: {args.src_file}")
    
    with open(args.src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]
    
    print(f"✅ {len(src_lines):,}行読み込み完了")
    
    # 再開処理
    start_idx = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            start_idx = sum(1 for _ in f)
        print(f"🔄 {start_idx:,}行目から再開します...")
    else:
        # 出力ディレクトリを作成
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 翻訳実行
    print(f"🔥 翻訳開始...")
    with open(args.output_file, 'a', encoding='utf-8') as f:
        for i in tqdm(range(start_idx, len(src_lines), args.batch_size), 
                      initial=start_idx//args.batch_size,
                      desc="翻訳中"):
            batch_raw = src_lines[i : i + args.batch_size]
            
            # 入力文をクリーニング
            batch_cleaned = [clean_input(line) if len(line) > 1 else line 
                             for line in batch_raw]
            
            # 空行対策
            batch_cleaned = [c if c else "。" for c in batch_cleaned]
            
            # ターゲット言語だけでなく、ソース言語(src_lang)も明示的に指定します
            tokenizer.src_lang = src_lang_code
            inputs = tokenizer(batch_cleaned, return_tensors="pt", 
                               padding=True, truncation=True, 
                               max_length=args.max_length).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            final_results = []
            for res in results:
                clean_res = res.replace("\n", " ").strip()
                
                # ソース言語が残っている、または極端に短い場合は失敗とマーク
                if contains_language(clean_res, args.src_lang) or len(clean_res) < 1:
                    final_results.append("FAILED_TRANSLATION_CLEANED")
                else:
                    final_results.append(clean_res)
            
            # サンプル表示
            if i % args.sample_interval < args.batch_size:
                print(f"\n--- [進捗チェック: {i:,}行目] ---")
                print(f"原文 ({args.src_lang}): {batch_raw[0]}")
                print(f"洗浄後: {batch_cleaned[0]}")
                print(f"翻訳 ({args.tgt_lang}): {final_results[0]}")
                print("-" * 50)
            
            # 結果を書き込み
            for res in final_results:
                f.write(res + "\n")
            f.flush()
            
            # メモリクリア
            del inputs, outputs
            torch.cuda.empty_cache()
    
    print(f"\n✅ 翻訳完了: {args.output_file}")

if __name__ == "__main__":
    main()
