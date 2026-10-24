#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

set -x

VENV_DIR="/mnt/public/dingbowen/uv_envs/verl-agent/weshop"
source "${VENV_DIR}/bin/activate"

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-DEBUG}

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"

WORK_DIR=/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent
echo "WORK_DIR=$WORK_DIR"
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$WORK_DIR/verl:${PYTHONPATH:-}"

wandb_experiment_name=graph_guide_llm_judge_step_future_miss_no_dis_mask_weshop
output_root=${WORK_DIR}/official_outputs/weshop
tensorboard_root=${WORK_DIR}/tensorboard_official/weshop
output_dir=${output_root}/$wandb_experiment_name
mkdir -p "$output_dir"

PYTHON_BIN="${VENV_DIR}/bin/python3"
PYTHON_MINOR="$(awk '/^version_info = / {split($3, v, "."); print v[1] "." v[2]}' "$VENV_DIR/pyvenv.cfg" 2>/dev/null)"
PYTHON_MINOR="${PYTHON_MINOR:-3.10}"

python_is_usable() {
  local candidate="$1"
  [[ -x "$candidate" ]] || return 1
  "$candidate" - <<'PY' >/dev/null 2>&1
import _contextvars
import _ctypes
import hydra
import pandas
import torch
import ray
import verl
import agent_system
PY
}

if ! python_is_usable "$PYTHON_BIN"; then
  echo "WARN: $PYTHON_BIN is missing, points to another node, or has an incomplete stdlib; trying to repair venv python" | tee -a "${output_dir}/cnt_from_0.progress"
  repaired_python=""
  mapfile -t python_candidates < <(
    {
      readlink -f "$PYTHON_BIN" 2>/dev/null || true
      command -v uv >/dev/null 2>&1 && uv python find "$PYTHON_MINOR" 2>/dev/null || true
      find /root/.local/share/uv/python -path "*/bin/python${PYTHON_MINOR}" -type f 2>/dev/null || true
      printf '%s\n' \
        "/usr/local/bin/python${PYTHON_MINOR}" \
        "/usr/bin/python${PYTHON_MINOR}" \
        "$(command -v python3 || true)"
    } | awk 'NF && !seen[$0]++'
  )
  for candidate in "${python_candidates[@]}"; do
    if python_is_usable "$candidate"; then
      ln -sfn "$candidate" "$VENV_DIR/bin/python"
      ln -sfn python "$VENV_DIR/bin/python3"
      ln -sfn python "$VENV_DIR/bin/python${PYTHON_MINOR}"
      if [[ -f "$VENV_DIR/pyvenv.cfg" ]]; then
        sed -i "s|^home = .*|home = $(dirname "$candidate")|" "$VENV_DIR/pyvenv.cfg"
      fi
      hash -r
      repaired_python="$candidate"
      echo "Repaired venv python symlink -> $candidate" | tee -a "${output_dir}/cnt_from_0.progress"
      break
    else
      echo "Skipping unusable python candidate: $candidate" | tee -a "${output_dir}/cnt_from_0.progress"
    fi
  done
  if [[ -z "$repaired_python" && -x "$(command -v uv 2>/dev/null)" ]]; then
    echo "No cached usable Python $PYTHON_MINOR found; trying: uv python install $PYTHON_MINOR" | tee -a "${output_dir}/cnt_from_0.progress"
    uv python install "$PYTHON_MINOR" 2>&1 | tee -a "${output_dir}/cnt_from_0.progress" || true
    candidate="$(uv python find "$PYTHON_MINOR" 2>/dev/null || true)"
    if python_is_usable "$candidate"; then
      ln -sfn "$candidate" "$VENV_DIR/bin/python"
      ln -sfn python "$VENV_DIR/bin/python3"
      ln -sfn python "$VENV_DIR/bin/python${PYTHON_MINOR}"
      if [[ -f "$VENV_DIR/pyvenv.cfg" ]]; then
        sed -i "s|^home = .*|home = $(dirname "$candidate")|" "$VENV_DIR/pyvenv.cfg"
      fi
      hash -r
      repaired_python="$candidate"
      echo "Repaired venv python symlink -> $candidate" | tee -a "${output_dir}/cnt_from_0.progress"
    fi
  fi
fi

if ! python_is_usable "$PYTHON_BIN"; then
  echo "ERROR: expected usable Python at: $PYTHON_BIN" | tee -a "${output_dir}/cnt_from_0.progress"
  echo "DEBUG: python symlinks:" | tee -a "${output_dir}/cnt_from_0.progress"
  ls -lah "$VENV_DIR"/bin/python* 2>&1 | tee -a "${output_dir}/cnt_from_0.progress"
  exit 1
fi
echo "PYTHON_BIN=$PYTHON_BIN"

model_path=/mnt/public/dingbowen/model_zoo/Qwen2.5-1.5B-Instruct

webshop_use_small=${WEBSHOP_USE_SMALL:-True}
webshop_human_goals=${WEBSHOP_HUMAN_GOALS:-False}

llm_judge_url=${LLM_JUDGE_URL:-http://172.16.55.136:8317/v1/chat/completions}
llm_judge_model=${LLM_JUDGE_MODEL:-glm-5}
llm_judge_cache_dir=${LLM_JUDGE_CACHE_DIR:-${output_dir}/llm_judge_cache}
llm_judge_max_workers=${LLM_JUDGE_MAX_WORKERS:-8}
llm_judge_timeout=${LLM_JUDGE_TIMEOUT:-90}
llm_judge_max_retries=${LLM_JUDGE_MAX_RETRIES:-2}
llm_judge_max_tokens=${LLM_JUDGE_MAX_TOKENS:-4096}
llm_judge_max_obs_chars=${LLM_JUDGE_MAX_OBS_CHARS:-1200}
llm_judge_fail_mode=${LLM_JUDGE_FAIL_MODE:-raise}
mkdir -p "$llm_judge_cache_dir"

set +x
export GRAPH_GUIDE_LLM_JUDGE_API_KEY="${GRAPH_GUIDE_LLM_JUDGE_API_KEY:-sk-aqIkyDNp4LiDpICSY}"
set -x

num_cpus_per_env_worker=0.1

train_data_size=16
val_data_size=128
group_size=8
actor_ppo_mini_batch_size=64
actor_ppo_micro_batch_size_per_gpu=8
rollout_logprob_micro_batch_size_per_gpu=16
ref_logprob_micro_batch_size_per_gpu=16
data_dir=/mnt/public/dingbowen/data_zoo/verl-agent/text
cnt_from=0
state=cnt_from_${cnt_from}

tensorboard_save_dir=${tensorboard_root}/${wandb_experiment_name}
export TENSORBOARD_DIR="$tensorboard_save_dir"

current_data_size=$("$PYTHON_BIN" - <<'PY'
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
  "$PYTHON_BIN" -m examples.data_preprocess.prepare \
      --mode 'text' \
      --train_data_size "$train_data_size" \
      --val_data_size "$val_data_size"
fi

ENGINE=${ENGINE:-vllm}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-4,5}
min_free_gpu_mb=${MIN_FREE_GPU_MB:-120000}
if command -v nvidia-smi >/dev/null 2>&1; then
  IFS=',' read -ra requested_gpus <<< "$cuda_visible_devices"
  gpu_mem_csv=$(nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits)
  for gpu_id in "${requested_gpus[@]}"; do
    gpu_id="$(echo "$gpu_id" | xargs)"
    gpu_line=$(echo "$gpu_mem_csv" | awk -F', ' -v id="$gpu_id" '$1 == id {print}')
    if [[ -z "$gpu_line" ]]; then
      echo "ERROR: CUDA_VISIBLE_DEVICES includes GPU $gpu_id, but nvidia-smi cannot find it." | tee -a "${output_dir}/$state.progress"
      exit 1
    fi
    gpu_free=$(echo "$gpu_line" | awk -F', ' '{print $4}')
    gpu_used=$(echo "$gpu_line" | awk -F', ' '{print $3}')
    gpu_total=$(echo "$gpu_line" | awk -F', ' '{print $2}')
    echo "GPU $gpu_id memory before launch: used=${gpu_used}MB free=${gpu_free}MB total=${gpu_total}MB" | tee -a "${output_dir}/$state.progress"
    if [[ "$gpu_free" -lt "$min_free_gpu_mb" ]]; then
      echo "ERROR: GPU $gpu_id has only ${gpu_free}MB free; need at least ${min_free_gpu_mb}MB. Pick idle GPUs via CUDA_VISIBLE_DEVICES or stop stale Ray/vLLM workers." | tee -a "${output_dir}/$state.progress"
      nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | tee -a "${output_dir}/$state.progress" || true
      exit 1
    fi
  done
fi
CUDA_VISIBLE_DEVICES=$cuda_visible_devices "$PYTHON_BIN" -m recipe.graph_guide.main_graph_guide \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_dir/test.parquet \
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
    ++algorithm.graph_guide.signature_mode=target \
    ++algorithm.graph_guide.hit_source=llm_judge \
    ++algorithm.graph_guide.llm_judge_url="$llm_judge_url" \
    ++algorithm.graph_guide.llm_judge_api_key_env=GRAPH_GUIDE_LLM_JUDGE_API_KEY \
    ++algorithm.graph_guide.llm_judge_model="$llm_judge_model" \
    ++algorithm.graph_guide.llm_judge_timeout=$llm_judge_timeout \
    ++algorithm.graph_guide.llm_judge_max_retries=$llm_judge_max_retries \
    ++algorithm.graph_guide.llm_judge_temperature=0.0 \
    ++algorithm.graph_guide.llm_judge_max_tokens=$llm_judge_max_tokens \
    ++algorithm.graph_guide.llm_judge_max_obs_chars=$llm_judge_max_obs_chars \
    ++algorithm.graph_guide.llm_judge_max_workers=$llm_judge_max_workers \
    ++algorithm.graph_guide.llm_judge_cache_dir="$llm_judge_cache_dir" \
    ++algorithm.graph_guide.llm_judge_fail_mode=$llm_judge_fail_mode \
    ++algorithm.graph_guide.beta=1.0 \
    ++algorithm.graph_guide.normalize_blame=true \
    ++algorithm.graph_guide.strict_missing_oracle=false \
    ++algorithm.graph_guide.online_oracle_update=false \
    ++algorithm.graph_guide.disadvantage_mode=trajectory \
    ++algorithm.graph_guide.blame_source=step_future_miss \
    ++algorithm.graph_guide.blame_normalization=mean \
    ++algorithm.graph_guide.mask_positive_unreasonable=true \
    ++algorithm.graph_guide.mask_negative_reasonable=false \
    ++algorithm.graph_guide.negative_mask_mode=none \
    ++algorithm.graph_guide.scale_eligible_mode=unmasked \
    ++algorithm.graph_guide.reweight_disadvantage_unreasonable=true \
    env.env_name=Webshop \
    env.seed=0 \
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
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$output_dir/$wandb_experiment_name \
    trainer.validation_data_dir=$output_dir/validation_data \
    trainer.rollout_data_dir=$output_dir/rollout_data \
    "$@" \
    >"${output_dir}/$state.log" 2>"${output_dir}/$state.progress"
echo "Done. Logs: ${output_dir}/$state.log  ${output_dir}/$state.progress"
