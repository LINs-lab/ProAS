#!/usr/bin/env python3
"""
Evaluate an OpenAI-compatible API model on the **fixed validation set** for
ALFWorld and WebShop, using the React (reasoning + action) prompting framework.

This script reuses the core API-calling and action-projection logic from
``recipe/graph_guide/run_sh_official/prompt/eval_prompt_llm.py`` but overrides
the environment construction so that evaluation runs on the exact 128 fixed
validation tasks defined in ``recipe/graph_guide/data/fixed_val_set/``.

Usage (ALFWorld):
    python recipe/graph_guide/react_infer/eval_react_fixed_val.py \
        --env alfworld \
        --model-name gpt-5.6-sol \
        --openai-url "$OPENAI_CHAT_COMPLETIONS_URL" \
        --api-key "$OPENAI_API_KEY" \
        --output-csv react_infer_results/alfworld_gpt5.6_sol.csv

Usage (WebShop):
    python recipe/graph_guide/react_infer/eval_react_fixed_val.py \
        --env webshop \
        --model-name gpt-5.6-sol \
        --openai-url "$OPENAI_CHAT_COMPLETIONS_URL" \
        --api-key "$OPENAI_API_KEY" \
        --output-csv react_infer_results/webshop_gpt5.6_sol.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# ALFWorld task families (must match env_manager._process_gamefile)
# ---------------------------------------------------------------------------
ALFWORLD_TASKS = [
    ("Pick", "pick_and_place"),
    ("Look", "look_at_obj_in_light"),
    ("Clean", "pick_clean_then_place_in_recep"),
    ("Heat", "pick_heat_then_place_in_recep"),
    ("Cool", "pick_cool_then_place_in_recep"),
    ("Pick2", "pick_two_obj_and_place"),
]


def nested_namespace(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


class DictNamespace(SimpleNamespace):
    """SimpleNamespace that also supports dict-like .get() access."""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Fixed-val-set environment construction
# ---------------------------------------------------------------------------
def make_config(env_name: str, env_seed: int, val_size: int, history_length: int, fixed_val: bool = True) -> SimpleNamespace:
    return DictNamespace(
        data=DictNamespace(train_batch_size=16, val_batch_size=val_size),
        env=DictNamespace(
            env_name="alfworld/AlfredTWEnv" if env_name == "alfworld" else "Webshop",
            seed=env_seed,
            max_steps=50 if env_name == "alfworld" else 15,
            history_length=history_length,
            rollout=DictNamespace(n=8),
            resources_per_worker=DictNamespace(num_cpus=0.1, num_gpus=0),
            alfworld=DictNamespace(eval_dataset="eval_in_distribution"),
            webshop=DictNamespace(use_small=True, human_goals=False),
            fixed_val_set=DictNamespace(enabled=fixed_val, root=str(REPO_ROOT / "recipe/graph_guide/data/fixed_val_set")),
        ),
    )


def build_eval_env(env_name: str, env_seed: int, val_size: int, history_length: int, fixed_val: bool = True):
    """Build environment that loads the **fixed validation set**."""
    config = make_config(env_name, env_seed, val_size, history_length, fixed_val=fixed_val)
    val_seed = env_seed + 1000
    resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0.0}

    if env_name == "alfworld":
        from functools import partial

        from agent_system.environments.env_manager import AlfWorldEnvironmentManager
        from agent_system.environments.env_package.alfworld import (
            alfworld_projection,
            build_alfworld_envs,
        )

        alf_config_path = REPO_ROOT / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
        env_kwargs = {"eval_dataset": "eval_in_distribution"}
        if fixed_val:
            from agent_system.environments.fixed_validation import load_alfworld_fixed_game_files

            env_kwargs["fixed_game_files"] = load_alfworld_fixed_game_files(config)
        envs = build_alfworld_envs(
            str(alf_config_path),
            val_seed,
            val_size,
            group_n=1,
            is_train=False,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )
        return AlfWorldEnvironmentManager(envs, partial(alfworld_projection), config)

    if env_name == "webshop":
        from functools import partial

        from agent_system.environments.env_manager import WebshopEnvironmentManager
        from agent_system.environments.env_package.webshop import (
            build_webshop_envs,
            webshop_projection,
        )

        webshop_dir = REPO_ROOT / "agent_system/environments/env_package/webshop/webshop/data"
        file_path = str(webshop_dir / "items_shuffle_1000.json")
        attr_path = str(webshop_dir / "items_ins_v2_1000.json")
        env_kwargs = {
            "observation_mode": "text",
            "num_products": None,
            "human_goals": False,
            "file_path": file_path,
            "attr_path": attr_path,
        }
        extra_kwargs = {}
        if fixed_val:
            from agent_system.environments.fixed_validation import (
                load_webshop_fixed_cases,
                load_webshop_fixed_seed,
                validate_webshop_fixed_data_files,
            )

            validate_webshop_fixed_data_files(config, file_path, attr_path)
            extra_kwargs["fixed_validation_cases"] = load_webshop_fixed_cases(config)
            extra_kwargs["fixed_validation_seed"] = load_webshop_fixed_seed(config)
        envs = build_webshop_envs(
            seed=val_seed,
            env_num=val_size,
            group_n=1,
            is_train=False,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
            **extra_kwargs,
        )
        return WebshopEnvironmentManager(envs, partial(webshop_projection), config)

    raise ValueError(f"Unsupported env: {env_name}")


# ---------------------------------------------------------------------------
# OpenAI-compatible chat-completions client
# ---------------------------------------------------------------------------
class ChatCompletionsClient:
    def __init__(
        self,
        url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        retries: int,
        retry_sleep: float,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep = retry_sleep

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "n": 1,
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"].strip()
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                OSError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                retry_after = None
                if isinstance(exc, urllib.error.HTTPError) and exc.headers is not None:
                    retry_after_value = exc.headers.get("Retry-After")
                    if retry_after_value:
                        try:
                            retry_after = float(retry_after_value)
                        except ValueError:
                            retry_after = None
                sleep_s = retry_after if retry_after is not None else self.retry_sleep * (2**attempt)
                sleep_s += random.uniform(0.0, min(1.0, self.retry_sleep))
                print(f"[WARN] API call failed ({exc}); retrying in {sleep_s:.1f}s", flush=True)
                time.sleep(sleep_s)
        raise RuntimeError(f"API call failed after {self.retries + 1} attempts: {last_error}")

    def complete_result(self, prompt: str) -> dict[str, str | None]:
        try:
            return {"output": self.complete(prompt), "error": None}
        except Exception as exc:
            return {"output": "", "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Prompt limiter (optional truncation)
# ---------------------------------------------------------------------------
class PromptLimiter:
    def __init__(self, tokenizer_path: str | None, max_prompt_tokens: int) -> None:
        self.max_prompt_tokens = max_prompt_tokens
        self.tokenizer = None
        if max_prompt_tokens <= 0:
            return
        if not tokenizer_path:
            raise ValueError("--tokenizer-path is required when --max-prompt-tokens > 0")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def limit(self, prompt: str) -> str:
        if self.max_prompt_tokens <= 0 or self.tokenizer is None:
            return prompt
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) <= self.max_prompt_tokens:
            return prompt
        head_budget = min(512, self.max_prompt_tokens // 4)
        tail_budget = self.max_prompt_tokens - head_budget
        kept_ids = token_ids[:head_budget] + token_ids[-tail_budget:]
        return self.tokenizer.decode(kept_ids, skip_special_tokens=False)

    def count_tokens(self, prompt: str) -> int | None:
        if self.tokenizer is None:
            return None
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Action projection (fallback when model doesn't emit <action> tags)
# ---------------------------------------------------------------------------
def fallback_action(text: str) -> str:
    lower = text.lower()
    if "<action>" in lower and "</action>" in lower and "<think>" in lower and "</think>" in lower:
        return text
    action = text.strip()
    match = re.search(r"<action>(.*?)</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        action = match.group(1).strip()
    else:
        action = action.splitlines()[-1].strip() if action.splitlines() else action
    return f"<think>{text}</think>\n<action>{action}</action>"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    env = build_eval_env(args.env, args.env_seed, args.val_size, args.history_length, fixed_val=not args.no_fixed_val)
    client = ChatCompletionsClient(
        url=args.openai_url,
        api_key=args.api_key,
        model=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    limiter = PromptLimiter(args.tokenizer_path, args.max_prompt_tokens)

    try:
        obs, infos = env.reset({})
        initial_gamefiles = [str(info.get("extra.gamefile", "")) for info in infos]
        distinct_gamefiles = len({gf for gf in initial_gamefiles if gf})
        print(
            f"[INFO] actual validation episodes={len(infos)} "
            f"distinct_initial_gamefiles={distinct_gamefiles}",
            flush=True,
        )
        done = [False for _ in range(args.val_size)]
        final_infos = [dict(info) for info in infos]
        final_rewards = np.zeros(args.val_size, dtype=np.float32)
        trajectories: list[dict[str, Any]] = [
            {
                "sample_id": idx,
                "env": args.env,
                "model": args.model_name,
                "env_seed": args.env_seed,
                "val_seed": args.env_seed + 1000,
                "max_steps": args.max_steps,
                "max_prompt_tokens": args.max_prompt_tokens,
                "max_response_tokens": args.max_tokens,
                "temperature": args.temperature,
                "initial_info": to_jsonable(infos[idx]),
                "steps": [],
            }
            for idx in range(args.val_size)
        ]

        for step_idx in range(args.max_steps):
            active = [idx for idx, is_done in enumerate(done) if not is_done]
            success_so_far = np.mean([bool(info.get("won", False)) for info in final_infos])
            print(
                f"[INFO] env={args.env} model={args.model_name} step={step_idx} "
                f"done={sum(done)}/{args.val_size} success_so_far={success_so_far:.4f}",
                flush=True,
            )
            if not active:
                break

            actions = ["<think>finished</think><action>None</action>" for _ in range(args.val_size)]
            prompt_by_idx: dict[int, str] = {}
            prompt_tokens_by_idx: dict[int, int | None] = {}
            prompts = []
            for idx in active:
                prompt = limiter.limit(obs["text"][idx])
                prompt_by_idx[idx] = prompt
                prompt_tokens_by_idx[idx] = limiter.count_tokens(prompt)
                prompts.append(prompt)

            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                results = list(executor.map(client.complete_result, prompts))

            for idx, result in zip(active, results):
                output = result["output"] or ""
                actions[idx] = fallback_action(output)
                if result["error"]:
                    print(f"[ERROR] sample={idx} step={step_idx} API failed after retries: {result['error']}", flush=True)
                trajectories[idx]["steps"].append(
                    {
                        "step": step_idx,
                        "prompt": prompt_by_idx[idx],
                        "prompt_tokens": prompt_tokens_by_idx[idx],
                        "raw_model_output": output,
                        "api_error": result["error"],
                        "projected_action_text": actions[idx],
                    }
                )

            obs, rewards, dones, infos = env.step(actions)
            for idx in active:
                final_infos[idx] = dict(infos[idx])
                final_rewards[idx] = float(rewards[idx])
                trajectories[idx]["steps"][-1].update(
                    {
                        "reward": float(rewards[idx]),
                        "done": bool(dones[idx]),
                        "info": to_jsonable(infos[idx]),
                    }
                )
                if bool(dones[idx]):
                    done[idx] = True
            for idx in range(args.val_size):
                trajectories[idx]["final_reward"] = float(final_rewards[idx])
                trajectories[idx]["final_info"] = to_jsonable(final_infos[idx])
                trajectories[idx]["success"] = bool(final_infos[idx].get("won", False))
            write_jsonl(resolve_partial_trajectory_path(args), trajectories, quiet=True)

        for idx in range(args.val_size):
            trajectories[idx]["final_reward"] = float(final_rewards[idx])
            trajectories[idx]["final_info"] = to_jsonable(final_infos[idx])
            trajectories[idx]["success"] = bool(final_infos[idx].get("won", False))
        write_jsonl(resolve_trajectory_path(args), trajectories)

        if args.env == "alfworld":
            return summarize_alfworld(args, final_infos, final_rewards)
        return summarize_webshop(args, final_infos)
    finally:
        try:
            env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------
def summarize_alfworld(args: argparse.Namespace, infos: list[dict[str, Any]], rewards: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {
        "Method": "React API",
        "Model": args.model_name,
        "Env": "ALFWorld",
        "Val Seed": args.env_seed + 1000,
        "Val Size": args.val_size,
    }
    wins = [float(bool(info.get("won", False))) for info in infos]
    for label, needle in ALFWORLD_TASKS:
        indices = [i for i, info in enumerate(infos) if needle in str(info.get("extra.gamefile", ""))]
        row[label] = round(100.0 * float(np.mean([wins[i] for i in indices])), 1) if indices else ""
        row[f"{label} N"] = len(indices)
    row["ALL"] = round(100.0 * float(np.mean(wins)), 1)
    row["TestScore"] = round(float(np.mean(rewards)), 3)
    return row


def summarize_webshop(args: argparse.Namespace, infos: list[dict[str, Any]]) -> dict[str, Any]:
    success = [float(bool(info.get("won", False))) for info in infos]
    scores = [float(info.get("task_score", 0.0)) for info in infos]
    return {
        "Method": "React API",
        "Model": args.model_name,
        "Env": "WebShop",
        "Val Seed": args.env_seed + 1000,
        "Val Size": args.val_size,
        "Score.": round(100.0 * float(np.mean(scores)), 1),
        "Succ.": round(100.0 * float(np.mean(success)), 1),
        "TaskScore": round(float(np.mean(scores)), 3),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def resolve_trajectory_path(args: argparse.Namespace) -> Path:
    if args.trajectory_dir:
        trajectory_dir = Path(args.trajectory_dir)
    else:
        output_csv = Path(args.output_csv)
        trajectory_dir = output_csv.parent / output_csv.stem
    return trajectory_dir / "trajectories.jsonl"


def resolve_partial_trajectory_path(args: argparse.Namespace) -> Path:
    path = resolve_trajectory_path(args)
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def write_jsonl(path: Path, records: list[dict[str, Any]], quiet: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
    if not quiet:
        print(f"[INFO] wrote {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate API model on fixed val set via React prompting"
    )
    parser.add_argument("--env", choices=["alfworld", "webshop"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--openai-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--trajectory-dir", default=None)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument(
        "--tokenizer-path",
        default=os.environ.get("MODEL_PATH_1_5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--no-fixed-val", action="store_true", help="Disable fixed val set (random sampling, for dry-run)")
    args = parser.parse_args()
    if args.max_steps is None:
        args.max_steps = 50 if args.env == "alfworld" else 15
    if args.max_prompt_tokens is None:
        args.max_prompt_tokens = 2048 if args.env == "alfworld" else 4096
    return args


def main() -> None:
    args = parse_args()
    row = run_eval(args)
    write_csv(Path(args.output_csv), row)
    print(f"[INFO] wrote {args.output_csv}")
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
