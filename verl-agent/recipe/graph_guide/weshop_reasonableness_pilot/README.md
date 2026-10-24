# WebShop Reasonableness Pilot

Offline target-level oracle pilot for WebShop rollout analysis.

简要结论：这里用 `items_human_ins.json + items_shuffle.json` 构建的是
target-level oracle，不是 step-level 人类专家轨迹。`items_human_ins.json`
里一个 ASIN 可能对应多条人类 instruction/goal 标注；当前实现把它们保留在
oracle JSON 的 `human_records` 字段中，但 reasonableness/SPIT 的主判定以
目标商品、必选 option、购买结果这些目标级状态为准。若 rollout 成功样本能反推出
target ASIN/options，会优先补齐当前任务的 target spec。

This pilot differs from the ALFWorld pilot: WebShop does not have a single
walkthrough path for a task. Search queries and browsing paths are naturally
multi-solution. The oracle here is therefore hierarchical target-level:

```text
ordered subgoals:
  G0 target_visible -> G1 target_product_opened
  -> G2 required_options_configured -> C0/C1/... case-specific checks
  -> G3 target_purchased

unordered sub-steps inside G2:
  {color=..., size=..., style=..., ...}

case-specific checks:
  C0 price_constraint_satisfied
  C1 attribute_constraint_satisfied
  ...
```

Oracle targets are built from:

```text
agent_system/environments/env_package/webshop/webshop/data/items_human_ins.json
agent_system/environments/env_package/webshop/webshop/data/items_shuffle.json
```

The rollout files can also contribute successful trajectories. For the current
GRPO pilot this is important because many sampled tasks come from WebShop's
templated goal generator rather than exactly matching the raw human instruction
text. Successful rollouts provide the purchased asin/options for that task,
while the data files provide product metadata and human annotations when present.

Default model rollout inputs:

```text
official_outputs/weshop/grpo_webshop/rollout_data/1.jsonl
official_outputs/weshop/grpo_webshop/rollout_data/3.jsonl
official_outputs/weshop/grpo_webshop/rollout_data/5.jsonl
official_outputs/weshop/grpo_webshop/rollout_data/50.jsonl
official_outputs/weshop/grpo_webshop/rollout_data/105.jsonl
```

Run:

```bash
cd /mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent
python recipe/graph_guide/weshop_reasonableness_pilot/run_pilot.py
```

Outputs:

```text
recipe/graph_guide/weshop_reasonableness_pilot/outputs/
  oracle_graphs/*.json
  oracle_graphs/index.json
  annotated_cases.jsonl
  summary.json
  rl_step_1_reasonableness_audit.html
  rl_step_3_reasonableness_audit.html
  rl_step_5_reasonableness_audit.html
  rl_step_50_reasonableness_audit.html
  rl_step_105_reasonableness_audit.html
  figures/*.png
```

Interpretation:

- `reasonable=1`: the step moves toward the target-level oracle, such as
  revealing the target asin, opening it, covering a previously uncovered
  required option bit, or buying the target with the required options.
- `reasonable=0`: the step is off target, repeats without new target progress,
  buys the wrong product/options, or emits malformed/non-English action text.
- `reasonable=null`: currently reserved for neutral bookkeeping, but the default
  classifier is intentionally strict and usually emits 0/1.

SPIT is hierarchical:

```text
SPIT = completed ordered units plus completed unordered option bits / total units
```

Ordered dependencies gate credit. Option bits only count after the target product
page is open, while the option bits themselves are order-insensitive.
Case-specific `C*` checks are visualized as purchase-validity gates, but they
currently have `spit_units=0` so static metadata checks do not inflate model
step progress.
