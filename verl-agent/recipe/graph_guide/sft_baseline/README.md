# SFT Baseline (`sft_baseline`)

This directory provides the SFT baseline of the main results table. The model is fine-tuned on oracle trajectories collected from exactly the same training tasks the RL runs see (`recipe/graph_guide/data/fixed_train_set/seed{N}/{env}`), and it is evaluated on the same validation tasks (`recipe/graph_guide/data/fixed_val_set/`). This makes the SFT and RL results directly comparable. 

## Repository Structure

- [`collect_alfworld_oracle_sft_data.py`](collect_alfworld_oracle_sft_data.py): Runs ALFWorld's handcoded expert on the training schedule and writes turn-level SFT data.
- [`collect_webshop_oracle_sft_data.py`](collect_webshop_oracle_sft_data.py): Runs a goal-driven oracle policy on the WebShop training schedule and writes turn-level SFT data.
- [`run_sft_alfworld.sh`](run_sft_alfworld.sh), [`run_sft_webshop.sh`](run_sft_webshop.sh): One-command pipelines that collect the data (if needed) and train with `verl.trainer.fsdp_sft_trainer`.
- [`data/`](data/): Released SFT data for every environment and seed (see [SFT Data](#sft-data)).
- [`eval_sft_alfworld.py`](eval_sft_alfworld.py), [`eval_sft_webshop.py`](eval_sft_webshop.py): Evatl a trained checkpoint on the fixed validation set.
- [`eval_sft_online.py`](eval_sft_online.py): Shared evaluator using the RL `TrajectoryCollector`, environment managers, vLLM worker, and `RayPPOTrainer._validate()`.

## Design

- **Oracle labels.** ALFWorld uses its handcoded expert. WebShop has no such expert, so the oracle is a state machine driven by each task's goal (target ASIN, query, and required options): search → paginate until the target appears → open it → select each option → buy.
- **Turn-level rows.** Each row of `train.parquet` is one decision step, i.e. a single `(user, assistant)` turn, rather than a whole conversation. This matches the per-step inputs of the RL rollout and avoids long-context truncation.
- **Same prompts as RL.** User messages use the templates in `agent_system/environments/prompts/` and the history format of `SimpleMemory.fetch`, with the same history window as RL (`history_length=2`).

## SFT Data

The SFT data used for the paper is included in [`data/`](data/), so the baseline can be trained without collecting it again:

```
data/
├── sft_alfworld/seed{0,1,2}/{train.parquet, metadata.json}
└── sft_webshop/seed{0,1,2}/{train.parquet, metadata.json}
```

Each seed is collected from the matching training schedule `recipe/graph_guide/data/fixed_train_set/seed{N}/{env}/schedule.jsonl` (2,400 task slots: 150 steps × 16 tasks). `metadata.json` records the collection settings and statistics:

| Environment | Seed | Oracle trajectories | Successful | Avg. steps | Turn-level rows |
|---|---|---|---|---|---|
| ALFWorld | 0 | 1,777 | 1,406 | 22.99 | 40,853 |
| ALFWorld | 1 | 1,767 | 1,412 | 22.80 | 40,292 |
| ALFWorld | 2 | 1,784 | 1,413 | 22.88 | 40,819 |
| WebShop | 0 | 2,400 | 1,856 | 7.42 | 17,801 |
| WebShop | 1 | 2,400 | 1,933 | 7.25 | 17,410 |
| WebShop | 2 | 2,400 | 1,909 | 7.26 | 17,422 |

Each row of `train.parquet` is one decision step:

- `messages`: a `[user, assistant]` pair. The user message is the RL prompt for that step, and the assistant message is the oracle action in the RL output format (`<think> … </think>` followed by `<action> … </action>`).
- `task_desc`, `step_idx`, and the task identifier: `gamefile` for ALFWorld, `case_id` for WebShop.
- `won`: whether the oracle trajectory this step belongs to completed the task.
- `weight`: how many times the task appears in the schedule.

Notes:

- ALFWorld stores one trajectory per unique game, and repeated games are only recorded in `weight`. WebShop stores one trajectory per schedule slot, so `weight` is always 1. The training scripts do not use `weight`.
- Trajectories where the oracle does not finish within the step limit (50 for ALFWorld, 15 for WebShop) are kept and marked `won=False`. Training uses all rows.

## Usage

From `verl-agent/`:

```bash
# Train on the released data (args: seed {0,1,2}, model size {1.5B,7B})
SKIP_COLLECT=1 bash recipe/graph_guide/sft_baseline/run_sft_alfworld.sh 0 1.5B
SKIP_COLLECT=1 bash recipe/graph_guide/sft_baseline/run_sft_webshop.sh  0 1.5B

# Collect the data again from the oracle, then train
bash recipe/graph_guide/sft_baseline/run_sft_alfworld.sh 0 1.5B
```

- **Data:** read from and written to `recipe/graph_guide/sft_baseline/data/sft_{alfworld,webshop}/seed{N}/`. Set `SKIP_COLLECT=1` to train on the released data. Without it, the script collects the data again and overwrites the released files.
- **Training:** 3 epochs, learning rate `1e-5`, cosine schedule, and a maximum length of 4096. Override these with `TOTAL_EPOCHS`, `LR`, `MAX_LENGTH`, and `MICRO_BATCH_SIZE_PER_GPU`; choose GPUs with `CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE`.
- **Outputs:** checkpoints and `sft.log` go to `outputs_fair/{1_5B,7B_new}/seed{N}/{env}/sft_oracle_{env}_seed{N}/`, the same layout as the RL runs.

Evaluate a checkpoint on the fixed validation set in the same environment used
by the corresponding `run_sh_fair` launcher (including Ray, vLLM, and the
environment dependencies). Use the HF model directory saved by the SFT trainer:

```bash
CUDA_VISIBLE_DEVICES=0,1 python recipe/graph_guide/sft_baseline/eval_sft_alfworld.py \
    --checkpoint outputs_fair/1_5B/seed0/alfworld/sft_oracle_alfworld_seed0/checkpoints/global_step_N \
    --model_size 1.5B --seed 0 \
    --output_json results_sft_alfworld_seed0.json

# Run inside the WebShop RL environment
CUDA_VISIBLE_DEVICES=0,1 python recipe/graph_guide/sft_baseline/eval_sft_webshop.py \
    --checkpoint outputs_fair/1_5B/seed0/webshop/sft_oracle_webshop_seed0/checkpoints/global_step_N \
    --model_size 1.5B --seed 0 \
    --output_json results_sft_webshop_seed0.json

# 7B uses four GPUs and tensor parallel size 4, matching run_sh_fair/7B_new
CUDA_VISIBLE_DEVICES=0,1,2,3 python recipe/graph_guide/sft_baseline/eval_sft_webshop.py \
    --checkpoint outputs_fair/7B_new/seed1/webshop/sft_oracle_webshop_seed1/checkpoints/global_step_N \
    --model_size 7B --seed 1 \
    --output_json results_sft_webshop_7b_seed1.json
```

Both entrypoints perform one validation pass using the same rollout and metric
code as online RL validation. They do not perform optimizer updates or resume
an RL checkpoint. The validation settings match `run_sh_fair`:

| Setting | ALFWorld | WebShop |
|---|---|---|
| Sampling | `do_sample=True`, temperature `0.4` | Same |
| Sampling filters | `top_p=1.0`, `top_k=-1` (disabled in vLLM) | Same |
| Samples per validation task | 1 | 1 |
| Response limit | 512 tokens per turn | 512 tokens per turn |
| Prompt limit / truncation | 2048 tokens / error | 4096 tokens / error |
| History window | 2 | 2 |
| Episode limit | 50 steps | 15 steps |
| Fixed validation tasks | 128 | 128 |
| Rollout dtype | bfloat16 | bfloat16 |


