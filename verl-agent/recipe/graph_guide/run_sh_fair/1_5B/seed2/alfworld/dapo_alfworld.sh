#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -e

SCRIPT_DIR="${RUN_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../../../env_setup/common_paths.sh"
set -x

WORK_DIR="${VERL_AGENT_ROOT}"
seed=2
env_name=alfworld
fixed_train_env_name=alfworld
fixed_train_set_path="${WORK_DIR}/fixed_train_set/seed${seed}/${fixed_train_env_name}"
fixed_train_set_start_step="${FIXED_TRAIN_SET_START_STEP:-null}"
fixed_train_set_start_index="${FIXED_TRAIN_SET_START_INDEX:-null}"
fixed_val_set_root="${WORK_DIR}/fixed_val_set"
VENV_DIR="${VIRTUAL_ENV:-${VERL_VENV_ROOT}/${env_name}}"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "ERROR: venv activate script not found: ${VENV_DIR}/bin/activate" >&2
  exit 1
fi
source "${VENV_DIR}/bin/activate"
PYTHON_BIN="${VENV_DIR}/bin/python3"


echo "WORK_DIR=$WORK_DIR"
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$WORK_DIR/verl:${PYTHONPATH:-}"


VENV_SITE_PACKAGES=$(find "${VENV_DIR}/lib" -maxdepth 2 -type d -name site-packages 2>/dev/null | head -n 1)
if [[ -n "$VENV_SITE_PACKAGES" ]]; then
  export PYTHONPATH="${VENV_SITE_PACKAGES}:${PYTHONPATH:-}"
fi

python_candidates=(
  "${VENV_DIR}/bin/python"
  "${VENV_DIR}/bin/python3"
)
while IFS= read -r py_candidate; do
  python_candidates+=("$py_candidate")
done < <(find "${UV_PYTHON_INSTALL_DIR}" -path "*/bin/python3.12" -type f 2>/dev/null | sort -r)
python_candidates+=(
  "$(command -v python3 || true)"
  "/usr/bin/python3"
  "/usr/local/bin/python3"
)

PYTHON_BIN=""
for py_candidate in "${python_candidates[@]}"; do
  if [[ -n "$py_candidate" && -x "$py_candidate" ]] && "$py_candidate" - <<'PY' >/dev/null 2>&1
import hydra
import pandas
import torch
import ray
import verl
import agent_system
PY
  then
    PYTHON_BIN="$py_candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: could not find a usable Python for DAPO AlfWorld. Checked venv, uv Python cache, and system python3." >&2
  exit 1
fi
echo "PYTHON_BIN=$PYTHON_BIN"

# ray stop || true
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-DEBUG}

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export ALFWORLD_DATA


wandb_experiment_name=dapo_alfworld_seed${seed}
output_dir=${WORK_DIR}/outputs_fair/1_5B/seed${seed}/${env_name}/$wandb_experiment_name
tensorboard_save_dir=${WORK_DIR}/tensorboard_fair/1_5B/seed${seed}/${env_name}/$wandb_experiment_name
mkdir -p "$output_dir"

model_path="${MODEL_PATH_1_5B}"

num_cpus_per_env_worker=0.1

train_data_size=16
val_data_size=128
group_size=8
actor_ppo_mini_batch_size=256
actor_ppo_micro_batch_size_per_gpu=32
rollout_logprob_micro_batch_size_per_gpu=32
ref_logprob_micro_batch_size_per_gpu=32
data_dir="${VERL_DATA_ROOT}/text"
cnt_from=0
state=cnt_from_${cnt_from}

clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0
loss_agg_mode=token-mean

export TENSORBOARD_DIR="$tensorboard_save_dir"

current_data_size=$("$PYTHON_BIN" - <<'PY'
import os
import pandas as pd

train_path = os.path.join(os.environ["VERL_DATA_ROOT"], "text", "train.parquet")
val_path = os.path.join(os.environ["VERL_DATA_ROOT"], "text", "test.parquet")

train_size = len(pd.read_parquet(train_path))
val_size = len(pd.read_parquet(val_path))
print(f"{train_size} {val_size}")
PY
)

read -r current_train_size current_val_size <<< "$current_data_size"

if [[ "$current_train_size" -lt "$train_data_size" || "$current_val_size" -lt "$val_data_size" ]]; then
  echo "Refreshing placeholder parquet data: train=${current_train_size}->${train_data_size}, val=${current_val_size}->${val_data_size}"
  "$PYTHON_BIN" -m examples.data_preprocess.prepare \
      --mode 'text' \
      --train_data_size "$train_data_size" \
      --val_data_size "$val_data_size"
fi

ENGINE=${ENGINE:-vllm}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-6,7}
CUDA_VISIBLE_DEVICES=$cuda_visible_devices "$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$actor_ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=$clip_ratio_c \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$rollout_logprob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ref_logprob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=False \
    +ray_init.runtime_env.env_vars.ALFWORLD_DATA=$ALFWORLD_DATA \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=$seed \
    env.fixed_train_set.path=$fixed_train_set_path \
    env.fixed_train_set.start_step=$fixed_train_set_start_step \
    env.fixed_train_set.start_index=$fixed_train_set_start_index \
    env.fixed_val_set.enabled=true \
    env.fixed_val_set.root=$fixed_val_set_root \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name="$wandb_experiment_name" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$output_dir/checkpoints \
    trainer.validation_data_dir=$output_dir/validation_data \
    trainer.rollout_data_dir=$output_dir/rollout_data \
    >"${output_dir}/$state.log" 2>"${output_dir}/$state.progress" &
echo "Done. Logs: ${output_dir}/$state.log  ${output_dir}/$state.progress"
