#!/usr/bin/env bash
set -euo pipefail

export WANDB_EXPERIMENT_NAME=${WANDB_EXPERIMENT_NAME:-graph_guide_alfworld_disonly_mean1_static}
export GRAPH_GUIDE_NORMALIZE_BLAME=true
export GRAPH_GUIDE_ONLINE_UPDATE=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/_run_alfworld_graph_guide_base.sh" "$@"
