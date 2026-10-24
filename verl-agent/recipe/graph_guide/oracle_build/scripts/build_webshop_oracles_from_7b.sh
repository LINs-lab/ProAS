#!/usr/bin/env bash
# Build seed-aligned WebShop oracle graphs from the 7B GRPO teacher's rollouts
# on fixed_train_set/seed${SEED}/webshop.
#
# Usage:
#   bash recipe/graph_guide/oracle_build/scripts/build_webshop_oracles_from_7b.sh SEED
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

seed=${1:?"Usage: $0 SEED"}
case "$seed" in
  0|1|2) ;;
  *) echo "ERROR: seed must be 0, 1, or 2" >&2; exit 2 ;;
esac

WORK_DIR="${VERL_AGENT_ROOT}"
VENV_DIR="${WESHOP_VENV_DIR:-${VERL_VENV_ROOT}/weshop}"
PYTHON_BIN=${VENV_DIR}/bin/python3
input_root=${TEACHER_ROLLOUT_ROOT:-${WORK_DIR}/official_outputs/7B/weshop/GRPO_7B_infer_fixed_train_seed${seed}}
input_path=${input_root}/validation_data/0.jsonl
successful_output=${input_root}/successful_trajectories/0.jsonl
output_root=${GRAPH_GUIDE_ROOT}/oracle_build/results/oracle_graphs_weshop_train/oracle_graphs_from_7b_seed${seed}
schedule_path=${WORK_DIR}/fixed_train_set/seed${seed}/webshop/schedule.jsonl
teacher_model=${TEACHER_MODEL:-${WORK_DIR}/official_outputs/7B/weshop/grpo_webshop/grpo_webshop/global_step_140/actor}
items_path=${WEBSHOP_DATA_DIR}/items_shuffle_1000.json

for required in "$PYTHON_BIN" "$input_path" "$schedule_path" "$items_path"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required file not found: $required" >&2
    if [[ "$required" == "$input_path" ]]; then
      echo "Run teacher inference first (see prepare_webshop_oracles_from_7b_all_seeds.sh)," >&2
      echo "or point TEACHER_ROLLOUT_ROOT at an existing rollout directory." >&2
    fi
    exit 1
  fi
done

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$WORK_DIR/verl:${PYTHONPATH:-}"

overwrite_arg=()
if [[ "${OVERWRITE_ORACLES:-0}" == "1" ]]; then
  overwrite_arg=(--overwrite)
fi

set -x
"$PYTHON_BIN" "${SCRIPT_DIR}/../build_webshop_oracles_from_7b_rollouts.py" \
  --input "$input_path" \
  --successful-output "$successful_output" \
  --output "$output_root" \
  --schedule "$schedule_path" \
  --teacher-model "$teacher_model" \
  --items "$items_path" \
  --seed "$seed" \
  "${overwrite_arg[@]}"
set +x

echo "Successful trajectories: $successful_output"
echo "Oracle index: ${output_root}/oracle_graphs/index.json"
