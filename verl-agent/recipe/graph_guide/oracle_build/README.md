# ARG Construction (`oracle_build`)

This directory builds the Agentic Reference Graph (ARG) for every training task. It also provides the environment adapters that [`graph_guide_credit.py`](../graph_guide_credit.py) uses to annotate rollouts against these graphs. In the code, an ARG is called an *oracle graph*.

## Repository Structure

- [`alfworld_reference_graph.py`](alfworld_reference_graph.py): ALFWorld adapter. It abstracts states, replays expert walkthroughs in TextWorld, builds ARGs, annotates steps, and evolves ARGs.
- [`webshop_reference_graph.py`](webshop_reference_graph.py): WebShop adapter. It derives target specs from goals, builds subgoal-chain ARGs, annotates steps, and evolves ARGs.
- [`build_alfworld_oracle_graphs.py`](build_alfworld_oracle_graphs.py): Builds the ALFWorld ARGs for a full dataset split.
- [`build_webshop_oracle_graphs.py`](build_webshop_oracle_graphs.py): Builds the WebShop ARGs from the goal set.
- [`build_incomplete_oracle_graphs.py`](build_incomplete_oracle_graphs.py): Derives the *Incomplete* ablation by pruning a fraction of every ARG.
- [`build_alfworld_oracles_from_7b_rollouts.py`](build_alfworld_oracles_from_7b_rollouts.py): Builds the *Distill* ablation for ALFWorld from a 7B teacher's successful rollouts.
- [`build_webshop_oracles_from_7b_rollouts.py`](build_webshop_oracles_from_7b_rollouts.py): Builds the *Distill* ablation for WebShop, one set per training seed.
- [`scripts/`](scripts/): Shell wrappers for the *Distill* pipeline. They check inputs, pick the right virtual environment, and write to `results/`.
- [`results/`](results/): Prebuilt ARGs used by the launchers.

## Design

An ARG is a DAG whose nodes are abstract progress states and whose edges are the actions that move between them. Abstracting away the raw observation lets different successful trajectories land on the same nodes, so a single graph serves as the progress reference for a task.

- **ALFWorld** (`signature_mode=entity`): a node is a symbolic state signature tied to the task's PDDL targets, such as `at:CLEAN_STATION[sinkbasin 1] | holding:TARGET_OBJ[apple 1] | object:clean[TARGET_OBJ]`. The initial graph comes from replaying each game's expert walkthrough in TextWorld (`state_source=textworld_replay`).
- **WebShop** (`signature_mode=target`): a node is a stage of an ordered subgoal chain for the target product: `target_visible` → `target_product_opened` → `required_options_configured` → `target_purchased`. The graphs are generated directly from the WebShop goals and product catalog, so no rollouts are needed.

## Adapter Interface

[`graph_guide_credit.py`](../graph_guide_credit.py) loads an adapter with `_load_adapter(env_adapter)` and calls the same functions for both environments:

- **Task key** (`task_key_from_gamefile` for ALFWorld, `task_key` for WebShop): maps a rollout to its entry in `index.json`.
- **`annotate_trajectory(rows, oracle, signature_mode)`**: maps every step to the ARG and returns per-step annotations, mainly `reasonable` ($\text{Rea}_t$), `hit` ($\text{PA}_t$), and `step_future_miss` (Future-Drift $\mathcal{D}_{i,t}$).
- **`merge_success_trajectory_into_oracle(oracle, rows, signature_mode, traj_uid)`**: adds a successful rollout to the ARG as an alternative reference path (ARG evolution).

To support a new environment, implement these functions in a new module and register it in `_load_adapter`.

## Prebuilt Results

Each directory contains `oracle_graphs/index.json`, which maps task keys to graph files. Graph paths inside the index are relative to the directory that contains `oracle_graphs/`.

| Directory under `results/` | Graphs | Used by |
|---|---|---|
| `oracle_graphs_textworld_train/oracle_graphs/` | 3,553 | ALFWorld ProAS and most ablations |
| `oracle_graphs_textworld_train/oracle_graphs_incomplete_40_full/` | 3,553 | ALFWorld *Incomplete* |
| `oracle_graphs_textworld_train/oracle_graphs_from_7b/` | 2,566 | ALFWorld *Distill* |
| `oracle_graphs_webshop_train/oracle_graphs/` | 6,858 | WebShop ProAS and most ablations |
| `oracle_graphs_webshop_train/oracle_graphs_incomplete_40_full/` | 6,858 | WebShop *Incomplete* |
| `oracle_graphs_webshop_train/oracle_graphs_from_7b_seed{0,1,2}/` | 1,301 / 1,302 / 1,324 | WebShop *Distill* |

- **Incomplete** drops 40% of the intermediate ARG nodes (ALFWorld) or required-option subgoals (WebShop), sampled globally with seed 0.
- **Distill** builds the ARGs from the successful trajectories of a Qwen2.5-7B-Instruct GRPO checkpoint (step 140) instead of expert walkthroughs or goal specs. Tasks the teacher never solves have no graph, so the *Distill* launchers set `strict_missing_oracle=false`.

## Usage

Run all commands from `verl-agent/`. ALFWorld builders need the ALFWorld environment (TextWorld) and `ALFWORLD_DATA`; WebShop builders need the WebShop environment and its data.

```bash
# Main ARGs
python recipe/graph_guide/oracle_build/build_alfworld_oracle_graphs.py --split train
python recipe/graph_guide/oracle_build/build_webshop_oracle_graphs.py --goal-mode synthetic

# Incomplete ablation (use --env webshop with the WebShop paths for WebShop)
python recipe/graph_guide/oracle_build/build_incomplete_oracle_graphs.py --env alfworld \
    --input-root recipe/graph_guide/oracle_build/results/oracle_graphs_textworld_train \
    --output-root recipe/graph_guide/oracle_build/results/oracle_graphs_textworld_train/oracle_graphs_incomplete_40_full \
    --drop-fraction 0.4 --seed 0
```

The *Distill* ARGs need teacher rollouts first:

```bash
# ALFWorld: one inference pass of the 7B teacher over all 3,553 training games, then build
bash recipe/graph_guide/run_sh_fair/7B_new/seed0/alfworld/GRPO_7B_infer.sh
bash recipe/graph_guide/oracle_build/scripts/build_alfworld_oracles_from_7b.sh   # after inference finishes

# WebShop: teacher inference on each seed's training schedule, then build one set per seed
TEACHER_INFER_SCRIPT=/path/to/teacher_infer.sh \
    bash recipe/graph_guide/oracle_build/scripts/prepare_webshop_oracles_from_7b_all_seeds.sh
```

The bundled ALFWorld `GRPO_7B_infer.sh` runs in the background, loads a trained HF teacher from `models/GRPO-7B-ALFWorld/` (override with `TEACHER_MODEL`), and writes `outputs/7B_new/seed0/alfworld/GRPO_7B_infer/val_data/0.jsonl`. Paths are relative to `recipe/graph_guide/`. The ALFWorld builder consumes that new location by default; use `TEACHER_ROLLOUT_ROOT` to select another directory containing `val_data/0.jsonl` (for an older `validation_data/` export, pass `--input` directly to the Python builder).

The separately supplied WebShop `TEACHER_INFER_SCRIPT` retains its own external contract: it is called as `bash $TEACHER_INFER_SCRIPT SEED GPUS` and must write `official_outputs/7B/webshop/GRPO_7B_infer_fixed_train_seed${SEED}/validation_data/0.jsonl`, or use `TEACHER_ROLLOUT_ROOT` when invoking its builder separately. To rebuild a single WebShop seed from existing rollouts, run `scripts/build_webshop_oracles_from_7b.sh SEED`. Set `OVERWRITE_ORACLES=1` to replace existing outputs.
