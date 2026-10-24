#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -e
set -x

seed=${1:?"Usage: $0 SEED [CUDA_VISIBLE_DEVICES]"}
case "$seed" in
  0|1|2) ;;
  *) echo "ERROR: seed must be 0, 1, or 2" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../common_7b_env.sh"
setup_verl_7b_env weshop

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-DEBUG}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"

WORK_DIR=/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$WORK_DIR/verl:${PYTHONPATH:-}"

experiment_name=GRPO_7B_infer_fixed_train_seed${seed}
output_dir=${WORK_DIR}/official_outputs/7B/weshop/${experiment_name}
validation_data_dir=${output_dir}/validation_data
placeholder_root=${output_dir}/placeholder_data
data_dir=${placeholder_root}/text
fixed_train_set_path=${WORK_DIR}/fixed_train_set/seed${seed}/webshop
model_path=${WORK_DIR}/official_outputs/7B/weshop/grpo_webshop/grpo_webshop/global_step_140/actor
mkdir -p "$validation_data_dir" "$data_dir"

if [[ ! -d "$model_path" ]]; then
  echo "ERROR: 7B GRPO actor checkpoint not found: $model_path" >&2
  exit 1
fi
if [[ ! -f "${fixed_train_set_path}/schedule.jsonl" ]]; then
  echo "ERROR: fixed WebShop training schedule not found: $fixed_train_set_path" >&2
  exit 1
fi
if [[ -f "${validation_data_dir}/0.jsonl" && "${OVERWRITE_INFERENCE:-0}" != "1" ]]; then
  echo "ERROR: inference output already exists: ${validation_data_dir}/0.jsonl" >&2
  echo "Set OVERWRITE_INFERENCE=1 only when you intentionally want to replace it." >&2
  exit 1
fi
if [[ "${OVERWRITE_INFERENCE:-0}" == "1" ]]; then
  rm -f "${validation_data_dir}/0.jsonl"
fi

train_data_size=16
val_data_size=2400
val_batch_size=16
source_data_dir=/mnt/public/dingbowen/data_zoo/verl-agent/text

"$PYTHON_BIN" - "$source_data_dir" "$data_dir" "$train_data_size" "$val_data_size" <<'PY'
import math
import sys
from pathlib import Path

import pandas as pd


source_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
train_size = int(sys.argv[3])
val_size = int(sys.argv[4])


def repeat_to_size(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError("Cannot build placeholder parquet from an empty source dataframe.")
    repeats = math.ceil(size / len(frame))
    return pd.concat([frame] * repeats, ignore_index=True).iloc[:size].copy()


output_dir.mkdir(parents=True, exist_ok=True)
train_source = pd.read_parquet(source_dir / "train.parquet")
val_source = pd.read_parquet(source_dir / "test.parquet")
repeat_to_size(train_source, train_size).to_parquet(output_dir / "train.parquet", index=False)
repeat_to_size(val_source, val_size).to_parquet(output_dir / "test.parquet", index=False)
PY

state=cnt_from_0
ENGINE=${ENGINE:-vllm}
cuda_visible_devices=${2:-${CUDA_VISIBLE_DEVICES:-0,1}}
run_inference() {
  CUDA_VISIBLE_DEVICES=$cuda_visible_devices "$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    ++actor_rollout_ref.rollout.seed=$seed \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=$seed \
    env.max_steps=15 \
    env.history_length=2 \
    env.rollout.n=1 \
    env.webshop.use_small=True \
    env.webshop.human_goals=False \
    env.resources_per_worker.num_cpus=0.1 \
    env.fixed_train_set.path=$fixed_train_set_path \
    env.fixed_train_set.start_step=null \
    env.fixed_train_set.start_index=null \
    env.fixed_train_set.use_for_validation=true \
    env.fixed_val_set.enabled=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$output_dir/checkpoints \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.rollout_data_dir=$output_dir/rollout_data
}

if [[ "${RUN_FOREGROUND:-0}" == "1" ]]; then
  run_inference >"${output_dir}/${state}.log" 2>"${output_dir}/${state}.progress"
else
  run_inference >"${output_dir}/${state}.log" 2>"${output_dir}/${state}.progress" &
fi

echo "Started WebShop fixed-train inference for seed${seed} on GPUs ${cuda_visible_devices}."
echo "Logs: ${output_dir}/${state}.log  ${output_dir}/${state}.progress"
echo "All trajectories: ${validation_data_dir}/0.jsonl"
