#!/bin/bash
# =============================================================================
# grasp-models 実行スクリプト
# 
# ローカルGPU / AWS EC2 両対応
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# -----------------------------------------------------------------------------
# 設定
# -----------------------------------------------------------------------------
WANDB_PROJECT="grasp-models"

# -----------------------------------------------------------------------------
# ヘルパー関数
# -----------------------------------------------------------------------------
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU検出:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        return 0
    else
        echo "警告: GPUが検出されませんでした（CPU実行）"
        return 1
    fi
}

activate_venv() {
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
}

# -----------------------------------------------------------------------------
# 1. セットアップ
# -----------------------------------------------------------------------------
cmd_setup() {
    echo "=== セットアップ ==="
    
    check_gpu || true
    
    # Python仮想環境
    if [ ! -d "venv" ]; then
        echo "仮想環境を作成中..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    # 依存関係
    echo "依存関係をインストール中..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo ""
    echo "セットアップ完了！"
    echo "次のステップ:"
    echo "  ./run.sh teacher  # 教師データ生成"
    echo "  ./run.sh train    # モデル学習"
}

# -----------------------------------------------------------------------------
# 2. 教師データ生成
# -----------------------------------------------------------------------------
cmd_teacher() {
    echo "=== 教師データ生成 ==="
    
    activate_venv
    check_gpu || true
    
    # 入力確認
    if [ ! -f "data/splits/train.ja" ]; then
        echo "エラー: data/splits/train.ja が見つかりません"
        echo "先にデータ準備を完了してください"
        exit 1
    fi
    
    # 既存の教師データ確認
    if [ -f "data/teacher/train.ko" ]; then
        existing_lines=$(wc -l < data/teacher/train.ko)
        total_lines=$(wc -l < data/splits/train.ja)
        echo "既存の教師データ: ${existing_lines} / ${total_lines} 行"
        
        if [ "$existing_lines" -lt "$total_lines" ]; then
            echo "途中から再開します..."
            RESUME_ARG="--resume-from $existing_lines"
        else
            echo "教師データ生成は完了しています"
            return 0
        fi
    else
        RESUME_ARG=""
    fi
    
    # 実行
    python training/generate_teacher_data.py \
        --input data/splits/train.ja \
        --output data/teacher/train.ko \
        --model facebook/nllb-200-1.3B \
        --batch-size 16 \
        --num-beams 5 \
        $RESUME_ARG
    
    echo ""
    echo "教師データ生成完了！"
    wc -l data/teacher/train.ko
}

# -----------------------------------------------------------------------------
# 3. モデル学習（Knowledge Distillation）
# -----------------------------------------------------------------------------
cmd_train() {
    echo "=== モデル学習 ==="
    
    activate_venv
    check_gpu || true
    
    # 教師データ確認
    if [ ! -f "data/teacher/train.ko" ]; then
        echo "エラー: data/teacher/train.ko が見つかりません"
        echo "先に ./run.sh teacher を実行してください"
        exit 1
    fi
    
    # Wandb設定（オプション）
    if [ -n "$WANDB_API_KEY" ]; then
        echo "Wandbにログイン中..."
        wandb login "$WANDB_API_KEY"
    fi
    
    # 学習実行
    python training/train.py \
        --output-dir models/ja-ko \
        --epochs "${EPOCHS:-10}" \
        --batch-size "${BATCH_SIZE:-32}" \
        --learning-rate "${LR:-3e-4}" \
        "$@"
    
    echo ""
    echo "学習完了！"
}

# -----------------------------------------------------------------------------
# 4. ベースライン学習（OPUS韓国語）
# -----------------------------------------------------------------------------
cmd_baseline() {
    echo "=== ベースライン学習（OPUS韓国語） ==="
    
    activate_venv
    check_gpu || true
    
    python training/train.py \
        --use-opus-target \
        --output-dir models/ja-ko-baseline \
        --epochs "${EPOCHS:-10}" \
        --batch-size "${BATCH_SIZE:-32}" \
        --learning-rate "${LR:-3e-4}" \
        "$@"
    
    echo ""
    echo "ベースライン学習完了！"
}

# -----------------------------------------------------------------------------
# 5. ONNXエクスポート
# -----------------------------------------------------------------------------
cmd_export() {
    echo "=== ONNXエクスポート ==="
    
    activate_venv
    
    MODEL_DIR="${1:-models/ja-ko}"
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo "エラー: $MODEL_DIR が見つかりません"
        exit 1
    fi
    
    python export/to_onnx.py \
        --model-dir "$MODEL_DIR" \
        --quantize
    
    echo ""
    echo "エクスポート完了！"
    ls -lh "$MODEL_DIR"/*.onnx 2>/dev/null || echo "ONNXファイルが見つかりません"
}

# -----------------------------------------------------------------------------
# 6. 評価
# -----------------------------------------------------------------------------
cmd_eval() {
    echo "=== 評価 ==="
    
    activate_venv
    
    MODEL_DIR="${1:-models/ja-ko}"
    
    python -c "
from training.train import SPMTokenizer, compute_metrics
from transformers import MarianMTModel
from datasets import Dataset
import torch
import numpy as np

# モデルとトークナイザー
model = MarianMTModel.from_pretrained('$MODEL_DIR')
tokenizer = SPMTokenizer('$MODEL_DIR/spm.model')

model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# テストデータ
with open('data/splits/test.ja') as f:
    test_ja = [l.strip() for l in f]
with open('data/splits/test.ko') as f:
    test_ko = [l.strip() for l in f]

# 翻訳
print(f'テスト: {len(test_ja)} 文')
predictions = []
batch_size = 32

for i in range(0, len(test_ja), batch_size):
    batch = test_ja[i:i+batch_size]
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4)
    predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# BLEU
import evaluate
metric = evaluate.load('sacrebleu')
result = metric.compute(predictions=predictions, references=[[r] for r in test_ko])
print(f'BLEU: {result[\"score\"]:.2f}')

# サンプル表示
print('\nサンプル:')
for i in [0, 100, 500]:
    print(f'  JA: {test_ja[i]}')
    print(f'  予測: {predictions[i]}')
    print(f'  正解: {test_ko[i]}')
    print()
"
}

# -----------------------------------------------------------------------------
# 7. データアップロード/ダウンロード（S3）
# -----------------------------------------------------------------------------
cmd_upload() {
    echo "=== S3アップロード ==="
    
    if [ -z "$S3_BUCKET" ]; then
        echo "エラー: S3_BUCKET を設定してください"
        echo "  export S3_BUCKET=your-bucket-name"
        exit 1
    fi
    
    echo "アップロード中: models/ -> s3://$S3_BUCKET/grasp-models/models/"
    aws s3 sync models/ "s3://$S3_BUCKET/grasp-models/models/"
    
    echo "完了！"
}

cmd_download() {
    echo "=== S3ダウンロード ==="
    
    if [ -z "$S3_BUCKET" ]; then
        echo "エラー: S3_BUCKET を設定してください"
        exit 1
    fi
    
    echo "ダウンロード中: s3://$S3_BUCKET/grasp-models/data/ -> data/"
    aws s3 sync "s3://$S3_BUCKET/grasp-models/data/" data/
    
    echo "完了！"
}

# -----------------------------------------------------------------------------
# 8. クリーンアップ
# -----------------------------------------------------------------------------
cmd_clean() {
    echo "=== クリーンアップ ==="
    
    read -p "モデルとチェックポイントを削除しますか？ (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        rm -rf models/*/checkpoint-*
        echo "チェックポイント削除完了"
    fi
}

# -----------------------------------------------------------------------------
# メイン
# -----------------------------------------------------------------------------
case "${1:-help}" in
    setup)
        cmd_setup
        ;;
    teacher)
        shift
        cmd_teacher "$@"
        ;;
    train)
        shift
        cmd_train "$@"
        ;;
    baseline)
        shift
        cmd_baseline "$@"
        ;;
    export)
        shift
        cmd_export "$@"
        ;;
    eval)
        shift
        cmd_eval "$@"
        ;;
    upload)
        cmd_upload
        ;;
    download)
        cmd_download
        ;;
    clean)
        cmd_clean
        ;;
    help|--help|-h|*)
        echo "Usage: ./run.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  setup      - 環境セットアップ（初回のみ）"
        echo "  teacher    - 教師データ生成（NLLB-200）"
        echo "  train      - モデル学習（Knowledge Distillation）"
        echo "  baseline   - ベースライン学習（OPUS韓国語）"
        echo "  export     - ONNXエクスポート"
        echo "  eval       - テストセットで評価"
        echo "  upload     - S3にアップロード"
        echo "  download   - S3からダウンロード"
        echo "  clean      - チェックポイント削除"
        echo ""
        echo "環境変数:"
        echo "  EPOCHS=10           学習エポック数"
        echo "  BATCH_SIZE=32       バッチサイズ"
        echo "  LR=3e-4             学習率"
        echo "  WANDB_API_KEY=xxx   Wandb APIキー"
        echo "  S3_BUCKET=xxx       S3バケット名"
        echo ""
        echo "例:"
        echo "  ./run.sh setup"
        echo "  ./run.sh teacher"
        echo "  EPOCHS=5 BATCH_SIZE=16 ./run.sh train"
        echo "  ./run.sh export models/ja-ko"
        ;;
esac
