set -x

source "/mnt/public/dingbowen/uv_envs/verl-agent/alfworld/bin/activate"

# 处理占卡
# TENSOR_HOLD_DIR="/mnt/public/dingbowen/tensor-hold"
# OCCUPY_PIDFILE="$TENSOR_HOLD_DIR/gpu_occupy.pid"

# stop_gpu_occupy() {
#   if [[ -f "$OCCUPY_PIDFILE" ]]; then
#     local pid
#     pid=$(cat "$OCCUPY_PIDFILE" 2>/dev/null || true)
#     if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
#       kill -9 "$pid" 2>/dev/null || true
#     fi
#     rm -f "$OCCUPY_PIDFILE"
#   fi
#   pkill -f "occupy_gpu.py" 2>/dev/null || true
# }

# start_gpu_occupy() {
#   if [[ -f "$TENSOR_HOLD_DIR/run_occupy.sh" ]]; then
#     (cd "$TENSOR_HOLD_DIR" && bash run_occupy.sh)
#   else
#     echo "WARN: ${TENSOR_HOLD_DIR}/run_occupy.sh not found, skip restarting GPU occupy." >&2
#   fi
# }

# stop_gpu_occupy
# trap start_gpu_occupy EXIT
# ----------------------------

# ray stop || true
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-DEBUG}
# 多卡+vLLM 下 CUDA_LAUNCH_BLOCKING=1 会强制逐算子同步，易放大 NCCL 等待/假死；调试单卡时再开
# export CUDA_LAUNCH_BLOCKING=1

# NCCL：异步错误处理 + 更长链路超时（秒，与 IB/RDMA 一致时常用）
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export ALFWORLD_DATA=/mnt/public/dingbowen/data_zoo/alfworld
# -----------------------------

WORK_DIR=/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent
echo "WORK_DIR=$WORK_DIR"
cd $WORK_DIR
export PYTHONPATH="$WORK_DIR/verl:$PYTHONPATH"
seed=1
wandb_experiment_name=gigpo_alfworld_seed${seed}
output_dir=${WORK_DIR}/official_outputs/$wandb_experiment_name
mkdir -p $output_dir


model_path=/mnt/public/dingbowen/model_zoo/Qwen2.5-1.5B-Instruct

# The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.
num_cpus_per_env_worker=0.1 

train_data_size=16
val_data_size=128
group_size=8
actor_ppo_mini_batch_size=256
actor_ppo_micro_batch_size_per_gpu=32
rollout_logprob_micro_batch_size_per_gpu=32
ref_logprob_micro_batch_size_per_gpu=32
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
data_dir=/mnt/public/dingbowen/data_zoo/verl-agent/text
cnt_from=0
state=cnt_from_${cnt_from}

tensorboard_save_dir=${WORK_DIR}/tensorboard_official/${wandb_experiment_name}
export TENSORBOARD_DIR="$tensorboard_save_dir"


# We only use data preparation to indicate the modality and the data size.
# Regenerate the placeholder parquet files when they are smaller than the
# configured batch sizes; otherwise the trainer will see an empty dataloader
# because train_dataloader uses drop_last=True.
current_data_size=$(python3 - <<'PY'
import pandas as pd

train_path = "/mnt/public/dingbowen/data_zoo/verl-agent/text/train.parquet"
val_path = "/mnt/public/dingbowen/data_zoo/verl-agent/text/test.parquet"

train_size = len(pd.read_parquet(train_path))
val_size = len(pd.read_parquet(val_path))
print(f"{train_size} {val_size}")
PY
)

read -r current_train_size current_val_size <<< "$current_data_size"

if [[ "$current_train_size" -lt "$train_data_size" || "$current_val_size" -lt "$val_data_size" ]]; then
  echo "Refreshing placeholder parquet data: train=${current_train_size}->${train_data_size}, val=${current_val_size}->${val_data_size}"
  python3 -m examples.data_preprocess.prepare \
      --mode 'text' \
      --train_data_size $train_data_size \
      --val_data_size $val_data_size
fi

CUDA_VISIBLE_DEVICES=4,5 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
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
    algorithm.use_kl_in_reward=False \
    +ray_init.runtime_env.env_vars.ALFWORLD_DATA=$ALFWORLD_DATA \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=$seed \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='gigpo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$output_dir/$wandb_experiment_name \
    trainer.validation_data_dir=$output_dir/validation_data \
    trainer.rollout_data_dir=$output_dir/rollout_data >${output_dir}/$state.log 2>${output_dir}/$state.progress &
echo "Done. Logs: ${output_dir}/$state.log  ${output_dir}/$state.progress"
