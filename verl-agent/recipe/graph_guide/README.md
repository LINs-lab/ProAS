# ProAS Recipe (`graph_guide`)

This recipe implements ProAS on top of verl-agent without modifying the PPO trainer or the policy loss. The base estimator (GRPO, DAPO, or GiGPO) computes advantages as usual; ProAS then reshapes them into step-level advantages using progress signals from the Agentic Reference Graph (ARG). In the code, the ARG is called the *oracle graph* and ProAS is called *graph-guided credit*. 

## Repository Structure

- [`main_alfworld_graph_guide.py`](main_alfworld_graph_guide.py): ALFWorld training entry point and shared trainer implementation. It patches the verl-agent trainer and then runs the standard PPO loop.
- [`main_webshop_graph_guide.py`](main_webshop_graph_guide.py): WebShop training entry point. It imports `main` from `main_alfworld_graph_guide.py`, so both environments share the same training implementation; environment behavior is selected by the launcher configuration.
- [`graph_guide_credit.py`](graph_guide_credit.py): Core of ProAS: ARG lookup and evolution, step annotation, and MUI/APR advantage reshaping. All options and their defaults are listed in `GraphGuideConfig`.
- [`progress_reward.py`](progress_reward.py): The reward for `GRPO+PR`, which adds `0.5 * SPIT` to the orginal reward.
- [`prepare_ppo_data.py`](prepare_ppo_data.py): Creates local placeholder parquet files for all fixed-schedule launchers, without dataset downloads. It supports both batch sizes of 16 for GRPO, DAPO, GiGPO experiments and 128 for PPO experiments.
- [`oracle_build/`](oracle_build/README.md): ARG construction for ALFWorld and WebShop, the environment adapters, and the prebuilt graphs.
- [`run_sh_fair/`](run_sh_fair/README.md): Launchers for ProAS, the RL baselines, and all ablations, organized by model size, seed, and environment.
- [`sft_baseline/`](sft_baseline/README.md): SFT baseline trained on oracle trajectories of the same training tasks.
- [`react_infer/`](react_infer/README.md): ReAct prompting baseline with a proprietary API model.

## Data, Models, and Run Artifacts

All paths below are relative to this `graph_guide/` directory:

```text
graph_guide/
├── data/
│   ├── fixed_train_set/seed{0,1,2}/{alfworld,webshop}/
│   ├── fix_train_set_ppo/seed{0,1,2}/{alfworld,webshop}/
│   ├── fixed_val_set/
│   ├── text/{train,test}.parquet           # generated: 16 / 128 rows
│   ├── ppo/text/{train,test}.parquet       # generated: 128 / 128 rows
│   └── teacher_infer/alfworld/text/        # generated teacher placeholders
├── models/
│   ├── Qwen2.5-1.5B-Instruct/              # local HuggingFace model files
│   └── Qwen2.5-7B-Instruct/
├── outputs/{1_5B,7B_new}/seed{N}/{env}/{experiment}/
│   ├── ckpts/
│   ├── rollout_data/
│   ├── val_data/
│   ├── cnt_from_0.log
│   └── cnt_from_0.progress
└── tensorboard_fair/{1_5B,7B_new}/seed{N}/{env}/{experiment}/
```

The fixed schedules and validation manifests select the tasks. Launchers create missing placeholders locally and keep exactly one training batch per epoch. Environment assets (ALFWorld installation and WebShop catalogue/search index) remain at their environment-specific locations; these are separate from the fixed task schedules. 

[`run_sh_fair/env_setup/common_paths.sh`](run_sh_fair/env_setup/common_paths.sh) resolves these defaults from the script location, not the caller's working directory. Override `VERL_DATA_ROOT`, `VERL_OUTPUT_ROOT`, `VERL_TENSORBOARD_ROOT`, `VERL_MODEL_ROOT`, or the size-specific `MODEL_PATH_1_5B` / `MODEL_PATH_7B` if needed. Put downloaded HF weights in the model directories before launching; scripts do not download or move model weights. Existing outputs are not migrated, and an old checkpoint must be selected explicitly when resuming.



## How It Works

1. **Build initial ARGs.** [`oracle_build/`](oracle_build/README.md) builds one graph per training task and writes an `index.json` that maps task keys to graph files.
2. **Patch the trainer.** [`main_alfworld_graph_guide.py`](main_alfworld_graph_guide.py) wraps `ray_trainer.compute_advantage`, so ProAS/suffix advantage computation ends with `apply_graph_guide_credit`. 
3. **Reshape the advantages.** For each trajectory in a batch, `apply_graph_guide_credit`:
   1. looks up the task's ARG through the environment adapter;
   2. if the trajectory succeeded, merges it into the ARG as an alternative reference path (ARG evolution; in memory, at most 32 paths per task);
   3. annotates every step with $\text{Rea}_t$, $\text{PA}_t$, and Future-Drift $\mathcal{D}_{i,t}$;
   4. applies MUI to steps with positive advantage and APR to steps with negative advantage.


## Paper-to-Code Mapping

| Paper | Code |
|---|---|
| ARG | Oracle graph, indexed by `oracle_build/results/*/oracle_graphs/index.json` |
| Reasonableness $\text{Rea}_t$ | Step annotation `reasonable` |
| Progress advancing $\text{PA}_t$ | Step annotation `hit` |
| Future-Drift $\mathcal{D}_{i,t}$ | Step annotation `step_future_miss`, i.e. the number of `hit = 0` steps from $t$ to the end |
| ARG evolution | `online_oracle_update=true` |
| MUI | `mask_positive_unreasonable=true` |
| APR | `reweight_disadvantage_unreasonable=true` with `blame_source=step_future_miss` and `blame_normalization=mean` |
| Uniformly credited steps $\mathcal{T}$ | `disadvantage_mode=trajectory` for GRPO/DAPO (all steps of a trajectory); `advantage_cluster` for GiGPO (steps sharing the same advantage value) |

## Configuration

ProAS is enabled through `algorithm.graph_guide.*` Hydra overrides. The full method uses:

```bash
++algorithm.graph_guide.enabled=true
++algorithm.graph_guide.env_adapter=webshop                  # WebShop only; the default is alfworld
++algorithm.graph_guide.oracle_index_path=<.../oracle_graphs/index.json>
++algorithm.graph_guide.signature_mode=entity                # entity for ALFWorld, target for WebShop
++algorithm.graph_guide.online_oracle_update=true            # ARG evolution
++algorithm.graph_guide.mask_positive_unreasonable=true      # MUI
++algorithm.graph_guide.reweight_disadvantage_unreasonable=true  # APR
++algorithm.graph_guide.blame_source=step_future_miss
++algorithm.graph_guide.blame_normalization=mean
++algorithm.graph_guide.disadvantage_mode=trajectory         # advantage_cluster for GiGPO
++algorithm.graph_guide.mask_negative_reasonable=false
++algorithm.graph_guide.negative_mask_mode=none
++algorithm.graph_guide.scale_eligible_mode=unmasked
```

The ablations change only a few of these options, such as `hit_source=random`, `online_oracle_update=false`, or a different `oracle_index_path`. [`run_sh_fair/`](run_sh_fair/README.md) maps each launcher to its paper setting.

  
## Quick Start

Install the environments as described in the [top-level README](../../../README.md#getting-started) and configure the paths described in [`run_sh_fair/`](run_sh_fair/README.md). Then, from `verl-agent/`:

```bash
bash recipe/graph_guide/run_sh_fair/1_5B/seed1/alfworld/graph_guide_step_future_miss_no_dis_mask_alfworld.sh

# Reward-shaping baseline
bash recipe/graph_guide/run_sh_fair/1_5B/seed0/alfworld/grpo_progress_reward_alfworld.sh

# Position-only APR ablation (use seed1 or seed2 for the other seeds)
bash recipe/graph_guide/run_sh_fair/1_5B/seed0/webshop/graph_guide_position_suffix_no_dis_mask_webshop.sh
```

