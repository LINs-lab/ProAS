#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -e
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../../../env_setup/common_7b_env.sh"
setup_verl_7b_env alfworld

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-DEBUG}
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export ALFWORLD_DATA

WORK_DIR="${VERL_AGENT_ROOT}"
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$WORK_DIR/verl:${PYTHONPATH:-}"

env_name=alfworld
experiment_name=GRPO_7B_infer
output_dir=${VERL_OUTPUT_ROOT}/7B_new/seed0/${env_name}/${experiment_name}
tensorboard_save_dir=${VERL_TENSORBOARD_ROOT}/7B_new/seed0/${env_name}/${experiment_name}
export TENSORBOARD_DIR="$tensorboard_save_dir"
validation_data_dir=${output_dir}/val_data
placeholder_root=${VERL_DATA_ROOT}/teacher_infer/alfworld
data_dir=${placeholder_root}/text
mkdir -p "$validation_data_dir" "$data_dir" "$tensorboard_save_dir"

model_path="${TEACHER_MODEL:-${VERL_MODEL_ROOT}/GRPO-7B-ALFWorld}"

# The ALFWorld train split contains 3,553 supported and solvable TextWorld games.
# 187 task slots x 19 validation batches covers every game exactly once.
train_data_size=2
val_data_size=3553
val_batch_size=187
num_cpus_per_env_worker=0.1

# Tasks come from the fixed schedule; generate placeholders locally, without downloads.
"$PYTHON_BIN" -m recipe.graph_guide.prepare_ppo_data \
    --data-dir "$data_dir" \
    --train-data-size "$train_data_size" \
    --val-data-size "$val_data_size"

if [[ ! -d "$model_path" ]]; then
  echo "ERROR: 7B GRPO actor checkpoint not found: $model_path" >&2
  exit 1
fi

state=cnt_from_0
ENGINE=${ENGINE:-vllm}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7} "$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    ++actor_rollout_ref.rollout.seed=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +ray_init.runtime_env.env_vars.ALFWORLD_DATA=$ALFWORLD_DATA \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=1 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.fixed_val_set.enabled=False \
    env.alfworld.eval_dataset=train \
    env.alfworld.deterministic_eval_sharding=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$output_dir/ckpts \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.rollout_data_dir=$output_dir/rollout_data \
    >"${output_dir}/${state}.log" 2>"${output_dir}/${state}.progress" &

echo "Started one-pass ALFWorld train-split inference on GPUs 6,7."
echo "Logs: ${output_dir}/${state}.log  ${output_dir}/${state}.progress"
echo "Validation trajectories: ${validation_data_dir}/0.jsonl"
