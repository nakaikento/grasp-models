#!/bin/bash
# RunPodでのLLM評価環境セットアップ

set -e

echo "🚀 RunPod LLM評価環境セットアップ"

# 1. 必要なパッケージインストール
echo "📦 パッケージインストール..."
pip install -q vllm sacrebleu unbabel-comet httpx

# 2. 評価ディレクトリ作成
mkdir -p /workspace/llm-eval/{samples,translations,results}

echo "✅ セットアップ完了"
echo ""
echo "次のステップ:"
echo "1. サンプルファイルをアップロード"
echo "   scp samples/source_ko.txt samples/reference_ja.txt root@IP:/workspace/llm-eval/samples/"
echo ""
echo "2. vLLMサーバー起動 (別ターミナル)"
echo "   vllm serve Qwen/Qwen3-32B --port 8000 --tensor-parallel-size 1"
echo ""
echo "3. 翻訳実行"
echo "   python3 translate_with_llm.py --provider vllm --base-url http://localhost:8000/v1 ..."
