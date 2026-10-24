#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$WORK_DIR/verl:${PYTHONPATH:-}"

OPENAI_URL="${OPENAI_URL:-https://api2.aigcbest.top/v1/chat/completions}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-sk-1MThkz7tq4dD0ysdvVYiFUVfXE0J5oH6SB3XwI6VLDQ9oCak}}"
MODEL_NAME="${MODEL_NAME:-gpt-5.1}"
OUTPUT_CSV="${OUTPUT_CSV:-recipe/graph_guide/results/prompt/gpt5_1_weshop.csv}"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-recipe/graph_guide/results/prompt/gpt5_1_weshop}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/public/dingbowen/model_zoo/Qwen2.5-1.5B-Instruct}"

if [[ -z "$API_KEY" ]]; then
  echo "ERROR: set API_KEY or OPENAI_API_KEY" >&2
  exit 1
fi

python3 recipe/graph_guide/run_sh_official/prompt/eval_prompt_llm.py \
  --env weshop \
  --model-name "$MODEL_NAME" \
  --openai-url "$OPENAI_URL" \
  --api-key "$API_KEY" \
  --output-csv "$OUTPUT_CSV" \
  --trajectory-dir "$TRAJECTORY_DIR" \
  --env-seed "${ENV_SEED:-0}" \
  --val-size "${VAL_SIZE:-128}" \
  --val-reset-skip "${VAL_RESET_SKIP:-0}" \
  --max-steps "${MAX_STEPS:-15}" \
  --max-prompt-tokens "${MAX_PROMPT_TOKENS:-4096}" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --max-workers "${MAX_WORKERS:-4}" \
  --temperature "${TEMPERATURE:-0.4}" \
  --max-tokens "${MAX_TOKENS:-512}" \
  --timeout "${TIMEOUT:-300}" \
  --retries "${RETRIES:-8}" \
  --retry-sleep "${RETRY_SLEEP:-5}"
