#!/usr/bin/env bash
# Build ALFWorld oracle graphs from the successful trajectories of the 7B GRPO
# teacher's rollouts. One oracle set is shared by all training seeds.
#
# Usage:
#   bash recipe/graph_guide/oracle_build/scripts/build_alfworld_oracles_from_7b.sh
#
# Env overrides:
#   TEACHER_ROLLOUT_ROOT  directory holding validation_data/0.jsonl from teacher inference
#   TEACHER_MODEL         teacher checkpoint path (recorded in oracle metadata only)
#   OVERWRITE_ORACLES=1   rebuild even if the output already exists
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../run_sh_fair/env_setup/common_paths.sh"

WORK_DIR="${VERL_AGENT_ROOT}"
VENV_DIR="${ALFWORLD_VENV_DIR:-${VERL_VENV_ROOT}/alfworld}"
PYTHON_BIN=${VENV_DIR}/bin/python3
input_root=${TEACHER_ROLLOUT_ROOT:-${WORK_DIR}/official_outputs/7B/alfworld/GRPO_7B_infer}
input_path=${input_root}/validation_data/0.jsonl
output_root=${GRAPH_GUIDE_ROOT}/oracle_build/results/oracle_graphs_textworld_train/oracle_graphs_from_7b
teacher_model=${TEACHER_MODEL:-${WORK_DIR}/official_outputs/7B/alfworld/grpo_alfworld/grpo_alfworld/global_step_140/actor}

for required in "$PYTHON_BIN" "$input_path" "$ALFWORLD_DATA"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required file not found: $required" >&2
    if [[ "$required" == "$input_path" ]]; then
      echo "Run 7B teacher inference on the ALFWorld training games first," >&2
      echo "or point TEACHER_ROLLOUT_ROOT at an existing rollout directory." >&2
    fi
    exit 1
  fi
done

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$WORK_DIR/verl:${PYTHONPATH:-}"
export ALFWORLD_DATA

overwrite_arg=()
if [[ "${OVERWRITE_ORACLES:-0}" == "1" ]]; then
  overwrite_arg=(--overwrite)
fi

set -x
"$PYTHON_BIN" "${SCRIPT_DIR}/../build_alfworld_oracles_from_7b_rollouts.py" \
  --input "$input_path" \
  --output "$output_root" \
  --teacher-model "$teacher_model" \
  "${overwrite_arg[@]}"
set +x

echo "Oracle index: ${output_root}/oracle_graphs/index.json"
