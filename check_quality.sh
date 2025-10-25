#!/usr/bin/env bash
# Proposed quality evaluation template (lm-evaluation-harness via vLLM).

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/llama3-8b-instruct-gptq}"
TASKS="${TASKS:-hellaswag,arc_challenge}"
BATCH="${BATCH:-8}"

echo "[INFO] lm-eval tasks: $TASKS"
lm_eval --model vllm --model_args "pretrained=$MODEL_PATH"   --tasks "$TASKS"   --batch_size "$BATCH"   --device cuda
