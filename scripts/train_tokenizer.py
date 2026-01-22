#!/usr/bin/env python3
"""
SentencePiece 語彙辞書作成スクリプト

Usage:
    pip install sentencepiece
    python scripts/train_tokenizer.py

Input:  data/cleaned/cleaned.{ja,ko}
Output: data/tokenized/spm.{model,vocab}
"""

import sentencepiece as spm
from pathlib import Path
from dataclasses import dataclass
import tempfile

# =============================================================================
# 設定
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class TokenizerConfig:
    # 入力
    ja_input: Path = PROJECT_ROOT / "data/cleaned/cleaned.ja"
    ko_input: Path = PROJECT_ROOT / "data/cleaned/cleaned.ko"
    
    # 出力
    output_dir: Path = PROJECT_ROOT / "data/tokenized"
    model_prefix: str = "spm"
    
    # SentencePiece設定
    vocab_size: int = 32000
    model_type: str = "unigram"  # unigram, bpe, char, word
    character_coverage: float = 0.9995
    
    # 特殊トークン
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    
    # 学習データのサンプリング（大規模データ用）
    input_sentence_size: int = 5000000  # 最大500万文
    shuffle_input_sentence: bool = True


config = TokenizerConfig()

# =============================================================================
# メイン処理
# =============================================================================

def main():
    print("=" * 50)
    print("SentencePiece 語彙辞書作成")
    print("=" * 50)
    
    # 入力確認
    if not config.ja_input.exists():
        print(f"エラー: {config.ja_input} が見つかりません")
        print("先に scripts/clean.py を実行してください")
        return
    
    # 出力ディレクトリ作成
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 日韓を結合した一時ファイル作成
    print("\nデータを準備中...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        temp_path = f.name
        
        # 日本語
        with open(config.ja_input, 'r', encoding='utf-8') as ja_f:
            for line in ja_f:
                f.write(line)
        
        # 韓国語
        with open(config.ko_input, 'r', encoding='utf-8') as ko_f:
            for line in ko_f:
                f.write(line)
    
    print(f"一時ファイル作成: {temp_path}")
    
    # SentencePiece学習
    model_path = config.output_dir / config.model_prefix
    
    print(f"\nSentencePiece学習開始...")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  model_type: {config.model_type}")
    print(f"  character_coverage: {config.character_coverage}")
    
    spm.SentencePieceTrainer.train(
        input=temp_path,
        model_prefix=str(model_path),
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        character_coverage=config.character_coverage,
        pad_id=config.pad_id,
        unk_id=config.unk_id,
        bos_id=config.bos_id,
        eos_id=config.eos_id,
        input_sentence_size=config.input_sentence_size,
        shuffle_input_sentence=config.shuffle_input_sentence,
    )
    
    # 一時ファイル削除
    Path(temp_path).unlink()
    
    print(f"\n完了！")
    print(f"  モデル: {model_path}.model")
    print(f"  語彙:   {model_path}.vocab")
    
    # テスト
    print("\n" + "=" * 50)
    print("トークナイズテスト")
    print("=" * 50)
    
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_path}.model")
    
    test_sentences = [
        "昔の人は言っていた",
        "二兎を追う者は一兎をも得ず",
        "옛 격언에 두 가지 일을 동시에 하려고도 하지 말고",
        "彼は町の映画館に勤める映写技師であり",
        "그는 소도시의 한 극장에서 영사 기사로 일하면서",
    ]
    
    for sent in test_sentences:
        tokens = sp.encode_as_pieces(sent)
        ids = sp.encode_as_ids(sent)
        print(f"\n原文: {sent}")
        print(f"トークン: {tokens}")
        print(f"ID数: {len(ids)}")


if __name__ == "__main__":
    main()