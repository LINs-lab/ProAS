# Fixed PPO Train Set

This directory contains the fixed task schedules used by the `ppo_correct`
launch scripts. Each seed has 150 batches of 128 task slots for ALFWorld and
WebShop. Validation continues to use the shared [fixed validation set](../fixed_val_set/README.md).

The generator mirrors the online PPO environment samplers:

- ALFWorld shuffles the eligible training games independently for every worker
  with seed `seed + slot`.
- WebShop samples 128 distinct goal indices per step with
  `numpy.random.RandomState(seed)`. Each slot resolves that index against the
  catalog generated with worker seed `seed + slot`.
- Every WebShop step contains 128 distinct base task identities. Validation
  overlaps and duplicate identities are replaced by the next eligible, unused
  goal index. Manifests record both replacement counts.

Regenerate from `verl-agent/`:

```bash
PYTHONPATH=. python recipe/graph_guide/data/fix_train_set_ppo/build_fix_train_set_ppo.py
```

ALFWorld generation expects `$ALFWORLD_DATA/json_2.1.1` to exist. The WebShop
source files are read from the repository's bundled small-data environment.
