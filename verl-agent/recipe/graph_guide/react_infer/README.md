# ReAct API Baseline (`react_infer`)

This directory evaluates a proprietary model behind an OpenAI-compatible chat API with ReAct prompting (reasoning, then an action). It provides the *Proprietary Model* row of the main results table. The evaluation uses the same 128 validation tasks as the RL runs (`recipe/graph_guide/data/fixed_val_set/`), the same environment wrappers, and the same action projection, so its scores are directly comparable to the RL checkpoints.

## Repository Structure

- [`eval_react_fixed_val.py`](eval_react_fixed_val.py): Runs the API model on the fixed validation set of ALFWorld or WebShop. It calls the API concurrently with retries, writes a one-row summary CSV, and can save every trajectory.
- [`run_eval_gpt5.6_sol.sh`](run_eval_gpt5.6_sol.sh): Evaluates GPT-5.6-Sol on both environments, each in its own venv.
- [`results/`](results/): Summary CSVs and per-task trajectories (`trajectories.jsonl`) of the reported GPT-5.6-Sol run.

## Usage

From `verl-agent/`:

```bash
export OPENAI_CHAT_COMPLETIONS_URL=<chat-completions endpoint>
export OPENAI_API_KEY=<key>
bash recipe/graph_guide/react_infer/run_eval_gpt5.6_sol.sh
```

The script uses `${VERL_VENV_ROOT}/alfworld` and `${VERL_VENV_ROOT}/webshop`; override them with `ALFWORLD_VENV` and `WEBSHOP_VENV` (the latter also defaults to the shared `WEBSHOP_VENV_DIR` when set). To evaluate another model, set `MODEL_NAME` or call `eval_react_fixed_val.py` directly:

```bash
python recipe/graph_guide/react_infer/eval_react_fixed_val.py \
    --env alfworld --model-name <model> \
    --openai-url "$OPENAI_CHAT_COMPLETIONS_URL" --api-key "$OPENAI_API_KEY" \
    --output-csv recipe/graph_guide/react_infer/results/alfworld_<model>.csv \
    --trajectory-dir recipe/graph_guide/react_infer/results/alfworld_<model>
```

Use `--env webshop` for WebShop. Generation defaults to greedy decoding (`--temperature 0.0`). `--no-fixed-val` switches to randomly sampled tasks, which is meant for dry runs only.

## Reported Results

| Environment | Metric | GPT-5.6-Sol |
|---|---|---|
| ALFWorld | SR (%): Pick / Look / Clean / Heat / Cool / Pick2 / All | 90.9 / 83.3 / 68.0 / 83.3 / 79.2 / 68.2 / **78.9** |
| WebShop | Task score / SR (%) | 27.6 / 21.9 |
