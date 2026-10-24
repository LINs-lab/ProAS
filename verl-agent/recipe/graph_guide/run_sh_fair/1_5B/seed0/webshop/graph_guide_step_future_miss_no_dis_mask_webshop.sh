#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -x

SCRIPT_DIR="${RUN_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../../../env_setup/common_paths.sh"

WORK_DIR="${VERL_AGENT_ROOT}"
seed=0
env_name=webshop
fixed_train_env_name=webshop
fixed_train_set_path="${VERL_DATA_ROOT}/fixed_train_set/seed${seed}/${fixed_train_env_name}"
fixed_train_set_start_step="${FIXED_TRAIN_SET_START_STEP:-null}"
fixed_train_set_start_index="${FIXED_TRAIN_SET_START_INDEX:-null}"
fixed_val_set_root="${VERL_DATA_ROOT}/fixed_val_set"
VENV_DIR="${VIRTUAL_ENV:-${WEBSHOP_VENV_DIR}}"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "ERROR: venv activate script not found: ${VENV_DIR}/bin/activate" >&2
  exit 1
fi
source "${VENV_DIR}/bin/activate"
PYTHON_BIN="${VENV_DIR}/bin/python3"



export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-DEBUG}

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"

echo "WORK_DIR=$WORK_DIR"
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$WORK_DIR/verl:${PYTHONPATH:-}"

wandb_experiment_name=graph_guide_step_future_miss_no_dis_mask_webshop_seed${seed}
output_dir=${VERL_OUTPUT_ROOT}/1_5B/seed${seed}/${env_name}/$wandb_experiment_name
tensorboard_save_dir=${VERL_TENSORBOARD_ROOT}/1_5B/seed${seed}/${env_name}/$wandb_experiment_name
mkdir -p "$output_dir" "$tensorboard_save_dir"

model_path="${MODEL_PATH_1_5B}"

webshop_data_dir=${WORK_DIR}/agent_system/environments/env_package/webshop/webshop/data
oracle_output_dir=${WEBSHOP_ORACLE_OUTPUT_DIR:-${WORK_DIR}/recipe/graph_guide/oracle_build/results/oracle_graphs_webshop_train}
oracle_index_path=${ORACLE_INDEX_PATH:-${oracle_output_dir}/oracle_graphs/index.json}
webshop_items_path=${WEBSHOP_ITEMS_PATH:-${webshop_data_dir}/items_shuffle_1000.json}
webshop_attr_path=${WEBSHOP_ATTR_PATH:-${webshop_data_dir}/items_ins_v2_1000.json}
webshop_human_ins_path=${WEBSHOP_HUMAN_INS_PATH:-${webshop_data_dir}/items_human_ins.json}
webshop_use_small=${WEBSHOP_USE_SMALL:-True}
webshop_human_goals=${WEBSHOP_HUMAN_GOALS:-False}
if [[ -n "${WEBSHOP_ORACLE_GOAL_MODE:-}" ]]; then
  webshop_oracle_goal_mode=$WEBSHOP_ORACLE_GOAL_MODE
elif [[ "${webshop_human_goals,,}" == "true" ]]; then
  webshop_oracle_goal_mode=human
else
  webshop_oracle_goal_mode=synthetic
fi
graph_guide_strict_missing_oracle=${GRAPH_GUIDE_STRICT_MISSING_ORACLE:-true}

if [[ ! -f "$oracle_index_path" ]]; then
  "$PYTHON_BIN" recipe/graph_guide/oracle_build/build_webshop_oracle_graphs.py \
    --items "$webshop_items_path" \
    --attr "$webshop_attr_path" \
    --human-ins "$webshop_human_ins_path" \
    --goal-mode "$webshop_oracle_goal_mode" \
    --output-dir "$oracle_output_dir" \
    ${WEBSHOP_ORACLE_EXTRA_ARGS:-}
fi
if [[ ! -f "$oracle_index_path" ]]; then
  echo "ERROR: WebShop oracle index was not built: $oracle_index_path" >&2
  exit 1
fi

num_cpus_per_env_worker=0.1

train_data_size=16
val_data_size=128
group_size=8
actor_ppo_mini_batch_size=64
actor_ppo_micro_batch_size_per_gpu=8
rollout_logprob_micro_batch_size_per_gpu=16
ref_logprob_micro_batch_size_per_gpu=16
data_dir="${VERL_DATA_ROOT}/text"
cnt_from=0
state=cnt_from_${cnt_from}

export TENSORBOARD_DIR="$tensorboard_save_dir"

# Tasks come from the fixed schedule; generate placeholders locally, without downloads.
"$PYTHON_BIN" -m recipe.graph_guide.prepare_ppo_data \
    --data-dir "$data_dir" \
    --train-data-size "$train_data_size" \
    --val-data-size "$val_data_size"


ENGINE=${ENGINE:-vllm}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0,1}
CUDA_VISIBLE_DEVICES=$cuda_visible_devices "$PYTHON_BIN" -m recipe.graph_guide.main_webshop_graph_guide \
    algorithm.adv_estimator=grpo \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
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
    algorithm.use_kl_in_reward=False \
    ++algorithm.graph_guide.enabled=true \
    ++algorithm.graph_guide.env_adapter=webshop \
    ++algorithm.graph_guide.oracle_index_path="$oracle_index_path" \
    ++algorithm.graph_guide.signature_mode=target \
    ++algorithm.graph_guide.beta=1.0 \
    ++algorithm.graph_guide.normalize_blame=true \
    ++algorithm.graph_guide.strict_missing_oracle=$graph_guide_strict_missing_oracle \
    ++algorithm.graph_guide.online_oracle_update=true \
    ++algorithm.graph_guide.online_max_paths_per_task=32 \
    ++algorithm.graph_guide.disadvantage_mode=trajectory \
    ++algorithm.graph_guide.blame_source=step_future_miss \
    ++algorithm.graph_guide.blame_normalization=mean \
    ++algorithm.graph_guide.mask_positive_unreasonable=true \
    ++algorithm.graph_guide.mask_negative_reasonable=false \
    ++algorithm.graph_guide.negative_mask_mode=none \
    ++algorithm.graph_guide.scale_eligible_mode=unmasked \
    ++algorithm.graph_guide.reweight_disadvantage_unreasonable=true \
    env.env_name=Webshop \
    env.seed=$seed \
    env.fixed_train_set.path=$fixed_train_set_path \
    env.fixed_train_set.start_step=$fixed_train_set_start_step \
    env.fixed_train_set.start_index=$fixed_train_set_start_index \
    env.fixed_val_set.enabled=true \
    env.fixed_val_set.root=$fixed_val_set_root \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.webshop.use_small=$webshop_use_small \
    env.webshop.human_goals=$webshop_human_goals \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name="$wandb_experiment_name" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$output_dir/ckpts \
    trainer.validation_data_dir=$output_dir/val_data \
    trainer.rollout_data_dir=$output_dir/rollout_data \
    >"${output_dir}/$state.log" 2>"${output_dir}/$state.progress" &
echo "Done. Logs: ${output_dir}/$state.log  ${output_dir}/$state.progress"
