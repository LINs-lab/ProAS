# SFT Baseline for ALFWorld (non-invasive)

This directory provides a **self-contained** SFT baseline that trains on
**oracle (handcoded-expert) trajectories** collected from the *exact same*
game schedule the RL experiments use (`fixed_train_set/seedN/alfworld`), so the
SFT-vs-RL comparison is apples-to-apples.

All changes live here — no RL code is modified.

## What's here

| file | purpose |
|------|---------|
| `collect_alfworld_oracle_sft_data.py` | Run the ALFWorld handcoded expert on the fixed train schedule, emit **turn-level** SFT parquet (`messages` column). |
| `collect_webshop_oracle_sft_data.py` | Run a goal-driven oracle policy (search→paginate→open→select options→buy) on the WebShop fixed train schedule, emit turn-level SFT parquet. |
| `run_sft_alfworld.sh` | One-command launcher: collect data (if needed) + train with `fsdp_sft_trainer`. |
| `run_sft_webshop.sh` | Same for WebShop (uses `weshop` venv for data collection, `alfworld` venv for training). |
| `eval_sft_alfworld.py` | Standalone evaluator: rolls out a checkpoint on `fixed_val_set/alfworld.jsonl` and reports `success_rate` + per-task-family breakdown. |

## Why turn-level (not trajectory-level) SFT

Each row in `train.parquet` is a single `(user, assistant)` turn — one RL
decision step — rather than a full multi-turn conversation. This:

- matches the RL rollout's per-step input distribution (each step is a
  self-contained prompt with a `history_length` sliding window),
- avoids long-context truncation,
- and the `messages` column format is directly consumable by
  `verl.utils.dataset.multiturn_sft_dataset.MultiTurnSFTDataset` with
  `data.multiturn.enable=true`.

The prompt text is built with the **byte-identical** templates and history
format used by the RL path
(`agent_system/environments/prompts/alfworld.py` + `SimpleMemory.fetch`).

## Usage

### 1. Train (collects data automatically on first run)

```bash
cd "$(git rev-parse --show-toplevel)/verl-agent"
bash recipe/graph_guide/sft_baseline/run_sft_alfworld.sh 0   # seed 0
bash recipe/graph_guide/sft_baseline/run_sft_alfworld.sh 1   # seed 1
bash recipe/graph_guide/sft_baseline/run_sft_alfworld.sh 2   # seed 2
```

Outputs land in the same `outputs_fair/1_5B/seedN/alfworld/` layout as RL runs.
GPU defaults per seed match the RL launchers (seed0=2,3 seed1=4,5 seed2=6,7);
override with `CUDA_VISIBLE_DEVICES=...`.

Skip data re-collection once the parquet exists:

```bash
SKIP_COLLECT=1 bash recipe/graph_guide/sft_baseline/run_sft_alfworld.sh 0
```

### 2. Evaluate on the fixed val set

```bash
python recipe/graph_guide/sft_baseline/eval_sft_alfworld.py \
    --checkpoint outputs_fair/1_5B/seed0/alfworld/sft_oracle_alfworld_seed0/checkpoints/global_step_N \
    --val_file fixed_val_set/alfworld.jsonl \
    --output_json results_sft_alfworld_seed0.json
```

Reports overall `success_rate`, per-task-family success, avg steps, and
invalid-action rate.

## WebShop

WebShop has no handcoded expert like ALFWorld. Instead, the oracle is a
**goal-driven policy**: the fixed schedule's `goal` field (`asin`, `query`,
`goal_options`) deterministically tells us the optimal action sequence. The
policy is a state machine:

```
search[query] → paginate until target ASIN visible → click[ASIN]
  → click[each option_value] → click[buy now]
```

```bash
bash recipe/graph_guide/sft_baseline/run_sft_webshop.sh 0   # seed 0
```

Note: data collection uses the `weshop` venv (gym + webshop deps); training
switches to the `alfworld` venv (torch + verl). The launch script handles this
automatically.

## Data location

SFT parquet is written to `data/sft_alfworld/seedN/train.parquet` with a
`metadata.json` summarizing collection stats (games, success, avg steps,
turn-row count).
