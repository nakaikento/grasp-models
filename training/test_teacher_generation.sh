#!/bin/bash
# 教師データ生成テストスクリプト

set -e

echo "=========================================="
echo "教師データ生成テスト（汎用版）"
echo "=========================================="
echo

# テストケース1: 日本語 → 韓国語（従来パターン）
echo "📝 テストケース1: 日本語 → 韓国語"
echo "コマンド:"
echo "python3 generate_teacher_data.py \\"
echo "  --src_lang ja \\"
echo "  --tgt_lang ko \\"
echo "  --src_file /tmp/test_ja.txt \\"
echo "  --output_file /tmp/test_output_ja_ko.txt \\"
echo "  --model_name facebook/nllb-200-distilled-600M \\"
echo "  --batch_size 10 \\"
echo "  --num_beams 3"
echo

# テストケース2: 韓国語 → 日本語（新パターン）
echo "📝 テストケース2: 韓国語 → 日本語"
echo "コマンド:"
echo "python3 generate_teacher_data.py \\"
echo "  --src_lang ko \\"
echo "  --tgt_lang ja \\"
echo "  --src_file /tmp/test_ko.txt \\"
echo "  --output_file /tmp/test_output_ko_ja.txt \\"
echo "  --model_name facebook/nllb-200-distilled-600M \\"
echo "  --batch_size 10 \\"
echo "  --num_beams 3"
echo

echo "=========================================="
echo "RunPodでの実行例（3.3Bモデル使用）"
echo "=========================================="
echo

echo "# 日本語 → 韓国語（従来の教師データ生成）"
echo "python3 training/generate_teacher_data.py \\"
echo "  --src_lang ja \\"
echo "  --tgt_lang ko \\"
echo "  --src_file data/splits/train.ja \\"
echo "  --output_file data/teacher/train_ja_ko.ko \\"
echo "  --batch_size 40 \\"
echo "  --num_beams 3"
echo

echo "# 韓国語 → 日本語（ko-jaモデル用）"
echo "python3 training/generate_teacher_data.py \\"
echo "  --src_lang ko \\"
echo "  --tgt_lang ja \\"
echo "  --src_file data/splits/train.ko \\"
echo "  --output_file data/teacher/train_ko_ja.ja \\"
echo "  --batch_size 40 \\"
echo "  --num_beams 3"
echo

echo "=========================================="
echo "✅ テスト準備完了"
echo "=========================================="
