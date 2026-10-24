#!/usr/bin/env bash
# Run 7B teacher inference on fixed_train_set seeds 0/1/2 in parallel, then build
# the WebShop oracle graphs for each seed with build_webshop_oracles_from_7b.sh.
#
# Usage:
#   TEACHER_INFER_SCRIPT=/path/to/infer.sh \
#     bash recipe/graph_guide/oracle_build/scripts/prepare_webshop_oracles_from_7b_all_seeds.sh
#
# TEACHER_INFER_SCRIPT is called as `bash $TEACHER_INFER_SCRIPT SEED GPUS` and must write
# official_outputs/7B/webshop/GRPO_7B_infer_fixed_train_seed${SEED}/validation_data/0.jsonl.
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../run_sh_fair/env_setup/common_paths.sh"
WORK_DIR="${VERL_AGENT_ROOT}"
infer_script=${TEACHER_INFER_SCRIPT:-${GRAPH_GUIDE_ROOT}/run_sh_official/7B/seed0/webshop/GRPO_7B_infer_fixed_train.sh}
build_script=${SCRIPT_DIR}/build_webshop_oracles_from_7b.sh
gpu_pairs=("0,1" "2,3" "4,5")
pids=()

if [[ ! -f "$infer_script" ]]; then
  echo "ERROR: teacher inference script not found: $infer_script" >&2
  echo "Set TEACHER_INFER_SCRIPT to the 7B teacher inference launcher." >&2
  exit 1
fi

for seed in 0 1 2; do
  log_dir=${WORK_DIR}/official_outputs/7B/webshop/GRPO_7B_infer_fixed_train_seed${seed}
  mkdir -p "$log_dir"
  echo "Launching teacher inference seed${seed} on GPUs ${gpu_pairs[$seed]}"
  RUN_FOREGROUND=1 bash "$infer_script" "$seed" "${gpu_pairs[$seed]}" \
    >"${log_dir}/launcher.log" 2>&1 &
  pids+=("$!")
done

failed=0
for seed in 0 1 2; do
  if ! wait "${pids[$seed]}"; then
    echo "ERROR: teacher inference failed for seed${seed}" >&2
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  exit 1
fi

for seed in 0 1 2; do
  bash "$build_script" "$seed"
done

echo "Built seed-aligned WebShop oracle graphs from the 7B step-140 teacher."
