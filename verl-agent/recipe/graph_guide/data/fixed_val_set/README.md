# Fixed Validation Set

This directory contains the fixed ALFWorld and WebShop validation tasks used by
training and evaluation launchers.

- `alfworld_gamefiles.txt`: ordered ALFWorld runtime gamefile list.
- `alfworld.jsonl`: ALFWorld task metadata used by SFT evaluation and dataset checks.
- `webshop.jsonl`: complete WebShop goal payloads loaded at runtime.
- `manifest.json`: runtime task counts, WebShop seed, and source-data hashes.

Set `env.fixed_val_set.enabled=true` and point `env.fixed_val_set.root` to this
directory (the default is `recipe/graph_guide/data/fixed_val_set`, relative to
`verl-agent/`). The validation batch size is expected to cover all 128 tasks.

`build_fixed_val_set.py` and its helper modules can regenerate the runtime assets
and optional local audit artifacts when the source datasets are available.
