#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SFT baseline on WebShop oracle trajectories (non-invasive).
#
# Trains Qwen2.5-{1.5B,7B}-Instruct via supervised fine-tuning on goal-driven
# oracle trajectories collected from the SAME task set that the RL experiments
# use (fixed_train_set/seedN/webshop), so the SFT-vs-RL comparison is fair.
#
# Pipeline:
#   1. (optional) collect oracle data  ->  data/sft_webshop/seedN/train.parquet
#   2. SFT with verl.trainer.fsdp_sft_trainer (multiturn mode)
#
# Usage:
#   bash recipe/graph_guide/sft_baseline/run_sft_webshop.sh [seed] [model_size]
#
#   seed:        0 | 1 | 2            (default: 0)
#   model_size:  1.5B | 7B            (default: 1.5B)
#
# Env overrides:
#   SKIP_COLLECT=1                  skip step 1 (reuse existing train.parquet)
#   CUDA_VISIBLE_DEVICES=0,1         pin GPUs (defaults: seed0=2,3 seed1=4,5 seed2=6,7)
#   NPROC_PER_NODE=2                number of GPUs for torchrun (default: 2)
# ---------------------------------------------------------------------------
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -e
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../run_sh_fair/env_setup/common_paths.sh"

# ---- args ----
seed=${1:-0}
model_size=${2:-1.5B}
env_name=webshop

# Normalise model_size tag -> dir tag + model path + batch size
case "$model_size" in
  1.5B|1_5B|1.5b)
    size_tag="1_5B"
    model_path="${MODEL_PATH_1_5B}"
    default_micro_batch=4
    ;;
  7B|7b)
    size_tag="7B"
    model_path="${MODEL_PATH_7B}"
    default_micro_batch=2
    ;;
  *)
    echo "ERROR: unsupported model_size '$model_size' (use 1.5B or 7B)" >&2
    exit 1
    ;;
esac

output_size_tag=${SFT_OUTPUT_SIZE_TAG:-$size_tag}
if [[ "$size_tag" == "7B" && -z "${SFT_OUTPUT_SIZE_TAG:-}" ]]; then
  output_size_tag="7B_new"
fi

WORK_DIR="${VERL_AGENT_ROOT}"
echo "WORK_DIR=$WORK_DIR"
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$WORK_DIR/verl:${PYTHONPATH:-}"

# ---- venv (weshop has gym + webshop deps; SFT itself runs in the alfworld venv) ----
# Step 1 (data collection) uses the weshop venv (gym + web_agent_site).
# Step 2 (training) uses the alfworld venv (torch + verl + fsdp_sft_trainer).
WESHOP_VENV_DIR="${WESHOP_VENV_DIR:-${VERL_VENV_ROOT}/weshop}"
SFT_VENV_DIR="${SFT_VENV_DIR:-${VERL_VENV_ROOT}/alfworld}"

if [[ ! -f "${WESHOP_VENV_DIR}/bin/activate" ]]; then
  echo "ERROR: weshop venv not found: ${WESHOP_VENV_DIR}/bin/activate" >&2
  exit 1
fi
if [[ ! -f "${SFT_VENV_DIR}/bin/activate" ]]; then
  echo "ERROR: sft venv not found: ${SFT_VENV_DIR}/bin/activate" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-DEBUG}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"

# ---- paths ----
schedule_path="${WORK_DIR}/fixed_train_set/seed${seed}/${env_name}/schedule.jsonl"
sft_data_dir="${WORK_DIR}/data/sft_${env_name}/seed${seed}"
sft_train_file="${sft_data_dir}/train.parquet"

wandb_experiment_name=sft_oracle_webshop_seed${seed}
output_dir=${WORK_DIR}/outputs_fair/${output_size_tag}/seed${seed}/${env_name}/${wandb_experiment_name}
tensorboard_save_dir=${WORK_DIR}/tensorboard_fair/${output_size_tag}/seed${seed}/${env_name}/${wandb_experiment_name}
mkdir -p "$output_dir" "$sft_data_dir" "$tensorboard_save_dir"

# model_path set above based on model_size

# ---------------------------------------------------------------------------
# Step 1: Collect oracle trajectories (uses weshop venv for gym + webshop)
# ---------------------------------------------------------------------------
if [[ "${SKIP_COLLECT:-0}" != "1" || ! -f "$sft_train_file" ]]; then
  echo "=== Collecting oracle SFT data from $schedule_path ==="
  source "${WESHOP_VENV_DIR}/bin/activate"
  VENV_SITE_PACKAGES=$(find "${WESHOP_VENV_DIR}/lib" -maxdepth 2 -type d -name site-packages 2>/dev/null | head -n 1)
  if [[ -n "$VENV_SITE_PACKAGES" ]]; then
    export PYTHONPATH="${VENV_SITE_PACKAGES}:${PYTHONPATH:-}"
  fi
  python recipe/graph_guide/sft_baseline/collect_webshop_oracle_sft_data.py \
      --schedule "$schedule_path" \
      --output_dir "$sft_data_dir" \
      --history_length 2 \
      --max_steps 15 \
      --seed "$seed"
  echo "=== Data collection done: $sft_train_file ==="
else
  echo "=== Skipping data collection (SKIP_COLLECT=1, data exists) ==="
fi

# ---------------------------------------------------------------------------
# Step 2: SFT training (uses alfworld venv for torch + verl)
# ---------------------------------------------------------------------------
source "${SFT_VENV_DIR}/bin/activate"
VENV_SITE_PACKAGES=$(find "${SFT_VENV_DIR}/lib" -maxdepth 2 -type d -name site-packages 2>/dev/null | head -n 1)
if [[ -n "$VENV_SITE_PACKAGES" ]]; then
  export PYTHONPATH="${VENV_SITE_PACKAGES}:${PYTHONPATH:-}"
fi

export TENSORBOARD_DIR="$tensorboard_save_dir"

# Pick a Python that has torch + verl
PYTHON_BIN=""
for py_candidate in "${SFT_VENV_DIR}/bin/python" "${SFT_VENV_DIR}/bin/python3"; do
  if [[ -x "$py_candidate" ]] && "$py_candidate" - <<'PY' >/dev/null 2>&1
import torch
import pandas
import verl
PY
  then
    PYTHON_BIN="$py_candidate"
    break
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: could not find a usable Python for SFT training." >&2
  exit 1
fi
echo "PYTHON_BIN=$PYTHON_BIN"

case "$seed" in
  0) default_cudas="2,3" ;;
  1) default_cudas="4,5" ;;
  2) default_cudas="6,7" ;;
  *) default_cudas="0,1" ;;
esac
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-$default_cudas}

nproc=$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
nproc=${NPROC_PER_NODE:-$nproc}
[[ -z "$nproc" || "$nproc" -lt 1 ]] && nproc=2

micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-$default_micro_batch}
max_length=${MAX_LENGTH:-4096}
total_epochs=${TOTAL_EPOCHS:-3}
lr=${LR:-1e-5}

CUDA_VISIBLE_DEVICES=$cuda_visible_devices torchrun --standalone --nnodes=1 --nproc_per_node=$nproc \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$sft_train_file" \
    data.val_files="$sft_train_file" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    data.max_length=$max_length \
    data.truncation=left \
    model.partial_pretrain="$model_path" \
    model.enable_gradient_checkpointing=True \
    optim.lr=$lr \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.03 \
    optim.lr_scheduler=cosine \
    trainer.default_local_dir="$output_dir/checkpoints" \
    trainer.project_name='verl_agent_webshop_sft' \
    trainer.experiment_name="$wandb_experiment_name" \
    trainer.logger=['console','tensorboard'] \
    trainer.total_epochs=$total_epochs \
    trainer.seed="$seed" \
    use_remove_padding=True \
    2>&1 | tee "${output_dir}/sft.log"

echo "Done. Checkpoints: ${output_dir}/checkpoints"
echo "Logs: ${output_dir}/sft.log"
