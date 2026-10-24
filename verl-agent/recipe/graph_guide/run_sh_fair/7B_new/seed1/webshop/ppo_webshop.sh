#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -e
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../../../env_setup/common_7b_env.sh"
setup_verl_7b_env webshop

if [[ ! -x "$JAVA_HOME/bin/javac" ]]; then
  echo "ERROR: javac not found under JAVA_HOME=$JAVA_HOME" >&2
  exit 1
fi

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-DEBUG}

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"

WORK_DIR="${VERL_AGENT_ROOT}"
echo "WORK_DIR=$WORK_DIR"
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$WORK_DIR/verl:${PYTHONPATH:-}"

seed=1
env_name=webshop
wandb_experiment_name=ppo_webshop_seed${seed}
output_dir=${VERL_OUTPUT_ROOT}/7B_new/seed${seed}/${env_name}/${wandb_experiment_name}
tensorboard_save_dir=${VERL_TENSORBOARD_ROOT}/7B_new/seed${seed}/${env_name}/${wandb_experiment_name}
fixed_train_env_name=webshop
fixed_train_set_path="${VERL_DATA_ROOT}/fix_train_set_ppo/seed${seed}/${fixed_train_env_name}"
fixed_train_set_start_step="${FIXED_TRAIN_SET_START_STEP:-null}"
fixed_train_set_start_index="${FIXED_TRAIN_SET_START_INDEX:-null}"
fixed_val_set_root="${VERL_DATA_ROOT}/fixed_val_set"
mkdir -p "$output_dir" "$tensorboard_save_dir"

model_path="${MODEL_PATH_7B}"

num_cpus_per_env_worker=0.1
webshop_use_small=${WEBSHOP_USE_SMALL:-True}
webshop_human_goals=${WEBSHOP_HUMAN_GOALS:-False}

train_data_size=128
val_data_size=128
actor_ppo_mini_batch_size=64
actor_ppo_micro_batch_size_per_gpu=4
rollout_logprob_micro_batch_size_per_gpu=8
ref_logprob_micro_batch_size_per_gpu=8
critic_ppo_micro_batch_size_per_gpu=2
data_dir="${VERL_DATA_ROOT}/ppo/text"
cnt_from=0
state=cnt_from_${cnt_from}

export TENSORBOARD_DIR="$tensorboard_save_dir"

# One placeholder batch per epoch matches one fixed PPO schedule step.
"$PYTHON_BIN" -m recipe.graph_guide.prepare_ppo_data \
    --data-dir "$data_dir" \
    --train-data-size "$train_data_size" \
    --val-data-size "$val_data_size"

ENGINE=${ENGINE:-vllm}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
CUDA_VISIBLE_DEVICES=$cuda_visible_devices "$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$actor_ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$rollout_logprob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ref_logprob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$model_path \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=$critic_ppo_micro_batch_size_per_gpu \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=$seed \
    env.rollout.n=1 \
    env.fixed_train_set.path=$fixed_train_set_path \
    env.fixed_train_set.start_step=$fixed_train_set_start_step \
    env.fixed_train_set.start_index=$fixed_train_set_start_index \
    env.fixed_val_set.enabled=true \
    env.fixed_val_set.root=$fixed_val_set_root \
    env.max_steps=15 \
    env.webshop.use_small=$webshop_use_small \
    env.webshop.human_goals=$webshop_human_goals \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name="$wandb_experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$output_dir/ckpts \
    trainer.validation_data_dir=$output_dir/val_data \
    trainer.rollout_data_dir=$output_dir/rollout_data \
    >"${output_dir}/$state.log" 2>"${output_dir}/$state.progress" &
echo "Done. Logs: ${output_dir}/$state.log  ${output_dir}/$state.progress"
