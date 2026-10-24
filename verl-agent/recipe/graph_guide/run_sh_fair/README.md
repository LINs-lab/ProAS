# Training Launchers (`run_sh_fair`)

Each launcher trains one method on one environment with one seed. The launchers control the rollout budget and validation tasks, with a separate task schedule for PPO:

- **Fixed tasks:** group-based methods use [`recipe/graph_guide/data/fixed_train_set/seed{N}/{env}`](../data/fixed_train_set/README.md). PPO uses [`recipe/graph_guide/data/fix_train_set_ppo/`](../data/fix_train_set_ppo/README.md), with schedules for seeds 0, 1, and 2 in ALFWorld and WebShop. All methods validate on the same 128 tasks from [`recipe/graph_guide/data/fixed_val_set/`](../data/fixed_val_set/README.md).
- **Same rollout budget:** 150 steps with 128 rollouts per step: 16 tasks × group size 8 for group-based methods, or 128 task slots × `env.rollout.n=1` for PPO. Validation runs every 10 steps.


## Repository Structure

- [`env_setup/`](env_setup/): Shared setup sourced by every launcher. `common_paths.sh` resolves the repository paths and the default data, model, and venv locations. `common_7b_env.sh` adds venv repair, JDK discovery for the WebShop search engine, and a Python header check for the 7B launchers. `paths.env.example` lists the variables you can override.
- [`1_5B/`](1_5B/): Qwen2.5-1.5B-Instruct launchers under `seed{0,1,2}/{alfworld,webshop}/`. Each run uses 2 GPUs.
- [`7B_new/`](7B_new/): Qwen2.5-7B-Instruct launchers with the same layout. Each run uses 4 GPUs. `seed0/alfworld/GRPO_7B_infer.sh` runs the teacher inference for the *Distill* ablation and writes to `recipe/graph_guide/outputs/7B_new/seed0/alfworld/GRPO_7B_infer/val_data/`, where the ARG builder reads by default (see [`oracle_build/`](../oracle_build/README.md)).

## Launcher Names

Launchers that train with ProAS call `recipe.graph_guide.main_alfworld_graph_guide`, or `main_webshop_graph_guide` for WebShop. The baseline launchers call the upstream `verl.trainer.main_ppo`. 

| Launcher | Paper setting |
|---|---|
| `grpo_{env}.sh`, `dapo_{env}.sh`, `gigpo_{env}.sh`, `ppo_{env}.sh`, `migpo_{env}.sh` | GRPO, DAPO, GiGPO, PPO, and MiGPO baselines |
| `graph_guide_step_future_miss_no_dis_mask_{env}.sh` | GRPO + ProAS |
| `dapo_graph_guide_step_future_miss_no_dis_mask_{env}.sh` | DAPO + ProAS |
| `gigpo_graph_guide_step_future_miss_v2_no_dis_mask_{env}.sh` | GiGPO + ProAS |
| `grpo_progress_reward_{env}.sh` | GRPO + PR (1.5B; per-step progress reward, no MUI/APR) |
| `graph_guide_step_future_miss_no_pos_mask_{env}.sh` | ProAS<sub>−</sub> (MUI disabled) |
| `graph_guide_step_future_miss_no_dis_mask_no_neg_reweight_{env}.sh` | ProAS<sub>+</sub> (APR disabled) |
| `graph_guide_current_miss_mask_{env}.sh` | ProAS<sub>msk±</sub> (APR replaced by masking) |
| `graph_guide_position_suffix_no_dis_mask_{env}.sh` | GRPO + ProAS<sub>suffix</sub> (1.5B; position-only APR) |
| `graph_guide_random_hit_step_future_miss_no_dis_mask_{env}.sh` | *Random* |
| `graph_guide_step_future_miss_no_dis_mask_no_oracle_evolve_{env}.sh` | *Static* |
| `graph_guide_step_future_miss_no_dis_mask_incomplete_oracle_40_{env}.sh` | *Incomplete* |
| `graph_guide_step_future_miss_no_dis_mask_incomplete_oracle_40_no_oracle_evolve_{env}.sh` | *Incomplete & Static* |
| `graph_guide_step_future_miss_no_dis_mask_oracle_from_7b_{env}.sh` | *Distill* |

Terminology:

- `step_future_miss`: APR weights penalties by Future-Drift.
- `position_suffix`: APR uses remaining trajectory length, normalized to mean 1; MUI and ARG evolution remain enabled.
- `progress_reward`: adds `0.5 * SPIT` to each step response before standard GRPO; 
- `no_dis_mask`: no step of a dispreferred trajectory is masked, so APR reweights all uniformly credited steps.
- `v2`: GiGPO uses the advantage-cluster mode, where $\mathcal{T}$ is the set of steps sharing an advantage value.

## Validation Protocol


| Setting | ALFWorld | WebShop |
|---|---|---|
| Sampling | `do_sample=True`, temperature `0.4` | Same |
| Sampling filters | `top_p=1.0`, `top_k=-1` (disabled in vLLM) | Same |
| Samples per validation task | 1 | 1 |
| Fixed validation tasks | 128 | 128 |
| Response limit | 512 tokens per turn | 512 tokens per turn |
| Prompt limit / truncation | 2048 tokens / error | 4096 tokens / error |
| History window | 2 | 2 |
| Episode limit | 50 steps | 15 steps |
| Rollout dtype | bfloat16 | bfloat16 |

The validation rollout uses Ray and vLLM, with tensor parallel size 2 for 1.5B and 4 for 7B. 

The [SFT evaluation entrypoints](../sft_baseline/README.md#usage) follow this same protocol through the shared [`eval_sft_online.py`](../sft_baseline/eval_sft_online.py), reusing the RL trajectory collector, environment managers, and `RayPPOTrainer._validate()`. They load HF checkpoint weights into the Ray/vLLM evaluation stack and perform one validation pass without training. 

## Usage

From `verl-agent/`, override the default paths if needed, then launch:

```bash
cp recipe/graph_guide/run_sh_fair/env_setup/paths.env.example my_paths.env   # edit as needed
source my_paths.env
bash recipe/graph_guide/run_sh_fair/1_5B/seed1/alfworld/graph_guide_step_future_miss_no_dis_mask_alfworld.sh
```

- **Python environment:** a launcher activates `${VERL_VENV_ROOT}/alfworld` or `${VERL_VENV_ROOT}/webshop`. Set `WEBSHOP_VENV_DIR` to reuse an existing WebShop venv at a different path. 
- **GPUs:** every launcher has a default GPU set. Most of them use an exported `CUDA_VISIBLE_DEVICES` when one is set; some 1.5B launchers hard-code the pair in the training command, so edit that line to fit your machine.
- **Data and placeholders:** `${VERL_DATA_ROOT}` defaults to `recipe/graph_guide/data/`. Fixed task schedules and validation sets are read beneath this root. All launchers use [`prepare_ppo_data.py`](../prepare_ppo_data.py), which generates local placeholders without downloads: `${VERL_DATA_ROOT}/text/{train,test}.parquet` has 16/128 rows for group methods; `${VERL_DATA_ROOT}/ppo/text/{train,test}.parquet` has 128/128 for PPO. One training batch per epoch keeps 150 epochs aligned with the 150-step schedule.
- **Outputs:** training runs in the background. `${VERL_OUTPUT_ROOT}` defaults to `recipe/graph_guide/outputs/`; each `{1_5B,7B_new}/seed{N}/{env}/{experiment}/` contains `ckpts/`, `rollout_data/`, `val_data/`, and `cnt_from_0.{log,progress}`. TensorBoard uses the same run layout under `${VERL_TENSORBOARD_ROOT}`, defaulting to `recipe/graph_guide/tensorboard_fair/`. Old output directories are not moved.
- **Models:** defaults are local HF weights in `recipe/graph_guide/models/Qwen2.5-{1.5B,7B}-Instruct/`. Override `MODEL_PATH_1_5B` / `MODEL_PATH_7B` if needed. Teacher inference instead needs trained HF weights at `models/GRPO-7B-ALFWorld/` or `TEACHER_MODEL`.
