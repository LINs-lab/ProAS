#!/usr/bin/env python3
"""
Lightweight, non-invasive evaluator for SFT WebShop checkpoints.

Mirrors eval_sft_alfworld.py but for WebShop. Loads a trained SFT checkpoint
(plain HuggingFace weights) and rolls it out on the fixed val set
(fixed_val_set/webshop.jsonl), reporting:

  - success_rate                    (fraction of tasks with won == True)
  - test_score                     (average reward, 0-10 scale)
  - webshop_task_score              (average task_score, 0-1 scale)

Note: runs greedily (temperature 0, do_sample=False).
Must be run in the weshop venv (has gym + webshop + torch + transformers).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


THINK_START = "<" + "think" + ">"
THINK_END = "<" + "/think" + ">"


WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within __THINK_START__ __THINK_END__ tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
""".replace("__THINK_START__", THINK_START).replace("__THINK_END__", THINK_END)


WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within __THINK_START__ __THINK_END__ tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
""".replace("__THINK_START__", THINK_START).replace("__THINK_END__", THINK_END)


def project_action(raw_text: str) -> tuple[str, bool]:
    """Extract <action>...</action>, validate think tags present and no CJK chars."""
    lowered = raw_text.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = lowered.find(start_tag)
    end_idx = lowered.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        return lowered[-30:], False
    extracted = lowered[start_idx + len(start_tag):end_idx].strip()
    valid = True
    if THINK_START not in raw_text or THINK_END not in raw_text:
        valid = False
    if re.search(r"[\u4e00-\u9fff]", raw_text):
        valid = False
    return extracted, valid


def fallback_action(text: str) -> str:
    """Wrap plain-text model output into think + action format."""
    lower = text.lower()
    if "<action>" in lower and "</action>" in lower and THINK_START in lower and THINK_END in lower:
        return text
    action = text.strip()
    match = re.search(r"<action>(.*?)</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        action = match.group(1).strip()
    else:
        lines = action.splitlines()
        action = lines[-1].strip() if lines else action
    return f"{THINK_START}{text}{THINK_END}\n<action>{action}</action>"


class DictNamespace(SimpleNamespace):
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def load_model(checkpoint: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    device: str,
) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    attention_mask = torch.ones_like(input_ids)
    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    new_tokens = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def make_config(val_size: int, history_length: int) -> DictNamespace:
    return DictNamespace(
        data=DictNamespace(train_batch_size=16, val_batch_size=val_size),
        env=DictNamespace(
            env_name="Webshop",
            seed=0,
            max_steps=15,
            history_length=history_length,
            rollout=DictNamespace(n=8),
            resources_per_worker=DictNamespace(num_cpus=0.1, num_gpus=0),
            alfworld=DictNamespace(eval_dataset="eval_in_distribution"),
            webshop=DictNamespace(use_small=True, human_goals=False),
            fixed_val_set=DictNamespace(enabled=True, root=str(REPO_ROOT / "fixed_val_set")),
        ),
    )


def build_eval_env(val_size: int, history_length: int):
    from functools import partial

    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop import (
        build_webshop_envs,
        webshop_projection,
    )
    from agent_system.environments.fixed_validation import (
        load_webshop_fixed_cases,
        load_webshop_fixed_seed,
        validate_webshop_fixed_data_files,
    )

    config = make_config(val_size, history_length)
    val_seed = 1000
    resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0.0}

    webshop_dir = REPO_ROOT / "agent_system/environments/env_package/webshop/webshop/data"
    file_path = str(webshop_dir / "items_shuffle_1000.json")
    attr_path = str(webshop_dir / "items_ins_v2_1000.json")
    validate_webshop_fixed_data_files(config, file_path, attr_path)
    fixed_cases = load_webshop_fixed_cases(config)
    fixed_seed = load_webshop_fixed_seed(config)
    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": False,
        "file_path": file_path,
        "attr_path": attr_path,
    }
    envs = build_webshop_envs(
        seed=val_seed,
        env_num=val_size,
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
        fixed_validation_cases=fixed_cases,
        fixed_validation_seed=fixed_seed,
    )
    return WebshopEnvironmentManager(envs, partial(webshop_projection), config)


def run_eval(args) -> dict[str, Any]:
    val_size = args.val_size
    env = build_eval_env(val_size, args.history_length)

    device = args.device
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model, tokenizer = load_model(args.checkpoint, device, dtype)
    print(f"Model loaded on {device} ({dtype})", flush=True)

    obs, infos = env.reset({})
    print(f"[INFO] validation episodes={len(infos)}", flush=True)

    done = [False] * val_size
    final_infos = [dict(info) for info in infos]
    final_rewards = np.zeros(val_size, dtype=np.float32)
    total_invalid = 0

    for step_idx in range(args.max_steps):
        active = [i for i, d in enumerate(done) if not d]
        if not active:
            break

        success_so_far = np.mean([bool(info.get("won", False)) for info in final_infos])
        print(
            f"[INFO] step={step_idx} done={sum(done)}/{val_size} "
            f"success_so_far={success_so_far:.4f}",
            flush=True,
        )

        actions = [f"{THINK_START}finished{THINK_END}\n<action>None</action>" for _ in range(val_size)]
        for idx in active:
            prompt = obs["text"][idx]
            messages = [{"role": "user", "content": prompt}]
            raw = generate_response(model, tokenizer, messages, args.max_new_tokens, device)
            action, valid = project_action(raw)
            if not valid:
                total_invalid += 1
                actions[idx] = fallback_action(raw)
            else:
                actions[idx] = raw

        obs, rewards, dones, infos = env.step(actions)
        for idx in active:
            final_infos[idx] = dict(infos[idx])
            final_rewards[idx] = float(rewards[idx])
            if bool(dones[idx]):
                done[idx] = True

    success = [bool(info.get("won", False)) for info in final_infos]
    scores = [float(info.get("task_score", 0.0)) for info in final_infos]
    n_success = sum(success)
    success_rate = n_success / max(val_size, 1)
    test_score = float(np.mean(final_rewards))
    task_score = float(np.mean(scores))

    print(f"\n=========== RESULTS ===========", flush=True)
    print(f"checkpoint     : {args.checkpoint}", flush=True)
    print(f"tasks          : {val_size}", flush=True)
    print(f"success        : {n_success}/{val_size} = {success_rate:.4f}", flush=True)
    print(f"test_score     : {test_score:.4f}", flush=True)
    print(f"task_score     : {task_score:.4f}", flush=True)
    print(f"invalid_actions: {total_invalid}", flush=True)

    summary = {
        "checkpoint": args.checkpoint,
        "val_file": "fixed_val_set/webshop.jsonl",
        "history_length": args.history_length,
        "max_steps": args.max_steps,
        "tasks": val_size,
        "success": n_success,
        "success_rate": success_rate,
        "test_score": test_score,
        "webshop_task_score": task_score,
        "total_invalid_actions": total_invalid,
        "per_task": [
            {
                "idx": i,
                "won": success[i],
                "reward": float(final_rewards[i]),
                "task_score": scores[i],
                "case_id": final_infos[i].get("fixed_val_case_id", ""),
            }
            for i in range(val_size)
        ],
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SFT WebShop checkpoint on fixed val set"
    )
    parser.add_argument("--checkpoint", required=True, help="HF checkpoint dir")
    parser.add_argument("--val_file", default="fixed_val_set/webshop.jsonl",
                        help="Path to fixed_val_set/webshop.jsonl (for reference)")
    parser.add_argument("--history_length", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_json", default=None, help="Where to dump per-task results")
    parser.add_argument("--val_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=-1, help="Limit N tasks (debug)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    if args.limit > 0 and args.limit < args.val_size:
        args.val_size = args.limit

    summary = run_eval(args)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote results to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
