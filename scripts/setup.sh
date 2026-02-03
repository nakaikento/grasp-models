#!/bin/bash
# セットアップスクリプト（RunPod用）

set -e

echo "=================================================="
echo "🚀 mt-ja-ko セットアップ"
echo "=================================================="

# GPU確認
echo ""
echo "📊 GPU確認..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "⚠️ GPU not found"

# ディレクトリ確認
echo ""
echo "📁 作業ディレクトリ: $(pwd)"

# HuggingFaceキャッシュ設定（RunPod用）
if [ -d "/workspace" ]; then
    echo ""
    echo "🔧 HuggingFaceキャッシュを /workspace に設定..."
    mkdir -p /workspace/huggingface_cache
    export HF_HOME=/workspace/huggingface_cache
    export HUGGINGFACE_HUB_CACHE=/workspace/huggingface_cache
    echo "  HF_HOME=$HF_HOME"
fi

# リポジトリ最新化
echo ""
echo "🔄 Gitリポジトリを最新化..."
git pull

# 依存関係インストール
echo ""
echo "📦 依存関係をインストール..."
pip install -q -r requirements.txt

# データ確認
echo ""
echo "✅ データ確認..."
if [ -d "data/splits" ]; then
    echo "  ✓ data/splits/ 存在"
    wc -l data/splits/*.{ja,ko} 2>/dev/null || echo "  ⚠️ データファイルなし"
else
    echo "  ⚠️ data/splits/ が見つかりません"
fi

if [ -d "data/tokenized" ]; then
    echo "  ✓ data/tokenized/ 存在"
    ls -lh data/tokenized/spm.model 2>/dev/null || echo "  ⚠️ spm.model なし"
else
    echo "  ⚠️ data/tokenized/ が見つかりません"
fi

# ディスク容量確認
echo ""
echo "💾 ディスク容量..."
df -h . | tail -1

echo ""
echo "=================================================="
echo "✅ セットアップ完了！"
echo "=================================================="
echo ""
echo "次のステップ:"
echo "  1. 教師データ生成: python training/train_pair.py --src-lang ko --tgt-lang ja --generate-teacher"
echo "  2. 学習: python training/train_pair.py --src-lang ko --tgt-lang ja --epochs 10"
echo "  3. ONNX変換: python scripts/convert_to_onnx.py --model-dir models/ko-ja"
echo "  4. 量子化: python scripts/quantize_onnx.py --model-dir models/ko-ja-onnx"
echo ""
