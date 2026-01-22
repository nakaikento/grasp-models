#!/bin/bash
# プロジェクト構成のセットアップ
# mt-ja-ko/ ディレクトリで実行

# ディレクトリ作成
mkdir -p data/raw
mkdir -p data/cleaned
mkdir -p data/tokenized
mkdir -p scripts
mkdir -p training
mkdir -p export
mkdir -p models

echo "ディレクトリ構成:"
echo "mt-ja-ko/"
echo "├── data/"
echo "│   ├── raw/        ← OPUSデータをここへ"
echo "│   ├── cleaned/"
echo "│   └── tokenized/"
echo "├── scripts/"
echo "├── training/"
echo "├── export/"
echo "├── models/"
echo "└── README.md"
echo ""
echo "次のステップ:"
echo "  1. OPUSファイルを data/raw/ に移動"
echo "     mv raw/OpenSubtitles.ja-ko.* data/raw/"
echo ""
echo "  2. クレンジング実行"
echo "     python scripts/clean.py"
