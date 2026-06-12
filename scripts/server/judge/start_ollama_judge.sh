#!/bin/bash
# Pull and verify Ollama judge model (run once; ollama serve may already be running as a service)
set -eu

MODEL="${QWEN_OLLAMA_MODEL:-qwen2.5:14b-instruct}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Install Ollama first:"
  echo "  curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi

echo "==> Pull judge model: $MODEL"
ollama pull "$MODEL"

echo "==> Test API"
curl -s http://127.0.0.1:11434/v1/models | head || {
  echo "Start ollama: ollama serve &"
  exit 1
}

echo "==> OK. Evaluator: http://127.0.0.1:11434/v1  model=$MODEL"
