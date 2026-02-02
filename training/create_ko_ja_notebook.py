#!/usr/bin/env python3
"""
mt_ja_ko_training.ipynb を ko_ja_training.ipynb に変換するスクリプト
"""
import json
import re

# 既存のNotebookを読み込み
with open('mt_ja_ko_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 変更リスト
replacements = [
    # タイトル・説明
    ('🇯🇵→🇰🇷', '🇰🇷→🇯🇵'),
    ('日本語→韓国語', '韓国語→日本語'),
    ('Japanese→Korean', 'Korean→Japanese'),
    
    # リポジトリ名は維持（変更前にマーク）
    ('mt-ja-ko', 'KEEP_REPO_NAME'),
    
    ('ja-ko', 'ko-ja'),
    ('ja→ko', 'ko→ja'),
    
    # リポジトリ名を戻す
    ('KEEP_REPO_NAME', 'mt-ja-ko'),
    
    # ファイルパス（データ）
    ('train.ja', 'TEMP_TRAIN_JA'),
    ('train.ko', 'train.ja'),
    ('TEMP_TRAIN_JA', 'train.ko'),
    ('val.ja', 'TEMP_VAL_JA'),
    ('val.ko', 'val.ja'),
    ('TEMP_VAL_JA', 'val.ko'),
    ('test.ja', 'TEMP_TEST_JA'),
    ('test.ko', 'test.ja'),
    ('TEMP_TEST_JA', 'test.ko'),
    
    # NLLB言語コード
    ('jpn_Jpan', 'TEMP_JPN'),
    ('kor_Hang', 'jpn_Jpan'),
    ('TEMP_JPN', 'kor_Hang'),
    
    # モデル出力ディレクトリ
    ('models/ja-ko', 'models/ko-ja'),
    
    # その他の言及
    ('韓国語翻訳', '日本語翻訳'),
]

def apply_replacements(text):
    """テキストに置換を適用"""
    if not isinstance(text, str):
        return text
    
    result = text
    for old, new in replacements:
        result = result.replace(old, new)
    return result

# 各セルに置換を適用
for cell in nb['cells']:
    if 'source' in cell:
        if isinstance(cell['source'], list):
            cell['source'] = [apply_replacements(line) for line in cell['source']]
        else:
            cell['source'] = apply_replacements(cell['source'])
    
    if 'outputs' in cell:
        # 出力もクリア（古い実行結果を削除）
        cell['outputs'] = []
    
    if 'execution_count' in cell:
        cell['execution_count'] = None

# 新しいNotebookとして保存
with open('ko_ja_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ ko_ja_training.ipynb 作成完了")
print("変更内容:")
print("  - タイトル: 🇯🇵→🇰🇷 → 🇰🇷→🇯🇵")
print("  - データパス: train.ja ↔ train.ko を入れ替え")
print("  - NLLB言語: jpn_Jpan ↔ kor_Hang を入れ替え")
print("  - 出力: models/ja-ko → models/ko-ja")
