#!/usr/bin/env bash
# Run React API evaluation on fixed_val_set for both ALFWorld and WebShop.
# Usage: bash recipe/graph_guide/react_infer/run_eval_gpt5.6_sol.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../run_sh_fair/env_setup/common_paths.sh"
REPO_ROOT="${VERL_AGENT_ROOT}"
cd "$REPO_ROOT"

MODEL_NAME="${MODEL_NAME:-gpt-5.6-sol}"
API_URL="${OPENAI_CHAT_COMPLETIONS_URL:?Set OPENAI_CHAT_COMPLETIONS_URL}"
API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
OUTPUT_DIR="recipe/graph_guide/react_infer/results"

ALFWORLD_VENV="${ALFWORLD_VENV:-${VERL_VENV_ROOT}/alfworld}"
WEBSHOP_VENV="${WEBSHOP_VENV:-${WEBSHOP_VENV_DIR}}"

mkdir -p "$OUTPUT_DIR"

echo "=== ALFWorld fixed-val-set evaluation (venv: alfworld) ==="
"$ALFWORLD_VENV/bin/python" recipe/graph_guide/react_infer/eval_react_fixed_val.py \
    --env alfworld \
    --model-name "$MODEL_NAME" \
    --openai-url "$API_URL" \
    --api-key "$API_KEY" \
    --output-csv "$OUTPUT_DIR/alfworld_gpt5.6_sol.csv" \
    --trajectory-dir "$OUTPUT_DIR/alfworld_gpt5.6_sol" \
    --temperature 0.0 \
    --max-tokens 4096 \
    --max-workers 32 \
    --timeout 300 \
    --retries 3

echo ""
echo "=== WebShop fixed-val-set evaluation (venv: webshop) ==="
"$WEBSHOP_VENV/bin/python" recipe/graph_guide/react_infer/eval_react_fixed_val.py \
    --env webshop \
    --model-name "$MODEL_NAME" \
    --openai-url "$API_URL" \
    --api-key "$API_KEY" \
    --output-csv "$OUTPUT_DIR/webshop_gpt5.6_sol.csv" \
    --trajectory-dir "$OUTPUT_DIR/webshop_gpt5.6_sol" \
    --temperature 0.0 \
    --max-tokens 4096 \
    --max-workers 32 \
    --timeout 300 \
    --retries 3

echo ""
echo "=== Done. Results in $OUTPUT_DIR ==="
