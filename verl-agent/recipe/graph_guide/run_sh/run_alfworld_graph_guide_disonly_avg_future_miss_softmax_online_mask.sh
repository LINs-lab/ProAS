#!/usr/bin/env bash
set -euo pipefail

# Online multi-reference oracle + Avg_Future_MISS softmax reweighting.
# Full graph-guide parameter format is passed explicitly below; common PPO,
# rollout, model, data, and environment parameters live in the shared base
# runner and can still be overridden by environment variables or trailing args.

export WANDB_EXPERIMENT_NAME=${WANDB_EXPERIMENT_NAME:-graph_guide_alfworld_disonly_avg_future_miss_softmax_online_mask}

export GRAPH_GUIDE_NORMALIZE_BLAME=true
export GRAPH_GUIDE_ONLINE_UPDATE=true
export GRAPH_GUIDE_ONLINE_MAX_PATHS_PER_TASK=${GRAPH_GUIDE_ONLINE_MAX_PATHS_PER_TASK:-32}
export GRAPH_GUIDE_BLAME_SOURCE=${GRAPH_GUIDE_BLAME_SOURCE:-avg_future_miss}
export GRAPH_GUIDE_BLAME_NORMALIZATION=${GRAPH_GUIDE_BLAME_NORMALIZATION:-softmax_mean1}
export GRAPH_GUIDE_SOFTMAX_TEMPERATURE=${GRAPH_GUIDE_SOFTMAX_TEMPERATURE:-1.0}
export GRAPH_GUIDE_MASK_POSITIVE_UNREASONABLE=true
export GRAPH_GUIDE_MASK_NEGATIVE_REASONABLE=true
export GRAPH_GUIDE_REWEIGHT_DISADVANTAGE_UNREASONABLE=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR=/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent
ORACLE_INDEX_PATH=${ORACLE_INDEX_PATH:-${WORK_DIR}/recipe/graph_guide/oracle_graphs_textworld_train/oracle_graphs/index.json}

bash "$SCRIPT_DIR/_run_alfworld_graph_guide_base.sh" \
  ++algorithm.graph_guide.enabled=true \
  ++algorithm.graph_guide.oracle_index_path="$ORACLE_INDEX_PATH" \
  ++algorithm.graph_guide.signature_mode=entity \
  ++algorithm.graph_guide.beta="${GRAPH_GUIDE_BETA:-1.0}" \
  ++algorithm.graph_guide.normalize_blame=true \
  ++algorithm.graph_guide.strict_missing_oracle=true \
  ++algorithm.graph_guide.online_oracle_update=true \
  ++algorithm.graph_guide.online_max_paths_per_task="$GRAPH_GUIDE_ONLINE_MAX_PATHS_PER_TASK" \
  ++algorithm.graph_guide.disadvantage_mode=trajectory \
  ++algorithm.graph_guide.blame_source="$GRAPH_GUIDE_BLAME_SOURCE" \
  ++algorithm.graph_guide.blame_normalization="$GRAPH_GUIDE_BLAME_NORMALIZATION" \
  ++algorithm.graph_guide.softmax_temperature="$GRAPH_GUIDE_SOFTMAX_TEMPERATURE" \
  ++algorithm.graph_guide.mask_positive_unreasonable=true \
  ++algorithm.graph_guide.mask_negative_reasonable=true \
  ++algorithm.graph_guide.reweight_disadvantage_unreasonable=true \
  "$@"

