# Fixed Train Set

This directory stores path-addressable fixed training schedules for ALFWorld and
WebShop. The runtime switch is intentionally small:

```bash
env.fixed_train_set.path=recipe/graph_guide/data/fixed_train_set/seed0/alfworld
```

If `env.fixed_train_set.path` is unset, `null`, or empty, training keeps the
original online sampler.

## Layout

- `seed0/alfworld`, `seed1/alfworld`, `seed2/alfworld`
- `seed0/webshop`, `seed1/webshop`, `seed2/webshop`

Each environment directory contains:

- `schedule.jsonl`: one JSON object per RL step, with 16 task slots.
- `manifest.json`: schema, source rollout path, horizon, hashes, and the
  fixed-validation overlap audit result.

The schedules are generated from the GRPO official rollout traces and preserve
the first-seen `episode_group_uid` order within each step. This keeps duplicate
task identities in distinct rollout groups when they occurred in the original
training batch.

## Regenerate

```bash
cd "$(git rev-parse --show-toplevel)/verl-agent"
PYTHONPATH=. python recipe/graph_guide/data/fixed_train_set/build_fixed_train_set.py --horizon 150
```

The overlap audit is performed during generation only. Runtime loading does not
repeat this audit.
