#!/usr/bin/env python3
"""
Lightweight, **non-invasive** evaluator for SFT ALFWorld checkpoints.

The verl ``fsdp_sft_trainer`` only reports token-level val loss; it does NOT run
the ALFWorld environment. This script closes that gap by loading a trained SFT
checkpoint (plain HuggingFace weights, no Ray/vLLM) and rolling it out on the
**fixed val set** (``fixed_val_set/alfworld.jsonl``), reporting:

  - ``success_rate``        (fraction of games with ``won == True``)
  - per-task-family success rates
  - average steps, invalid-action rate

It reuses the *exact same* prompt construction and action projection as the RL
rollout (``ALFWORLD_TEMPLATE``/``ALFWORLD_TEMPLATE_NO_HIS`` from
``agent_system/environments/prompts/alfworld.py`` and the ``<action>`` extraction
from ``agent_system/environments/env_package/alfworld/projection.py``), so the
numbers are directly comparable to the RL ``val/success_rate``.

Usage
-----
    python recipe/graph_guide/sft_baseline/eval_sft_alfworld.py \
        --checkpoint outputs_fair/1_5B/seed0/alfworld/sft_oracle_alfworld_seed0/checkpoints/global_step_N \
        --val_file fixed_val_set/alfworld.jsonl \
        --output_json results_sft_alfworld_seed0.json

Note: runs greedily (temperature 0, do_sample=False).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch


# ---------------------------------------------------------------------------
# Prompt templates + projection — duplicated from the RL code path so this script
# has NO dependency on the full agent_system / verl training stack at import time.
# They are kept byte-identical to:
#   agent_system/environments/prompts/alfworld.py
#   agent_system/environments/env_package/alfworld/projection.py
#   agent_system/memory/memory.py (SimpleMemory.fetch)
# ---------------------------------------------------------------------------
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def format_admissible_actions(admissible_commands: List[str]) -> str:
    return "\n ".join(f"'{s}'" for s in admissible_commands if s != "help")


def build_history_lines(history: List[Dict[str, str]], history_length: int) -> str:
    recent = history[-history_length:] if history_length > 0 else []
    start_idx = len(history) - len(recent)
    lines = []
    for j, rec in enumerate(recent):
        step_num = start_idx + j + 1
        lines.append(
            f"[Observation {step_num}: '{rec['text_obs']}', Action {step_num}: '{rec['action']}']"
        )
    return "\n".join(lines)


def build_user_message(
    task_desc: str,
    text_obs: str,
    admissible_actions: List[str],
    history: List[Dict[str, str]],
    history_length: int,
    init: bool,
) -> str:
    if init or history_length <= 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=text_obs,
            admissible_actions=format_admissible_actions(admissible_actions),
        )
    action_history = build_history_lines(history, history_length)
    valid_len = min(len(history), history_length)
    return ALFWORLD_TEMPLATE.format(
        task_description=task_desc,
        step_count=len(history),
        history_length=valid_len,
        action_history=action_history,
        current_step=len(history) + 1,
        current_observation=text_obs,
        admissible_actions=format_admissible_actions(admissible_actions),
    )


def project_action(raw_text: str) -> tuple[str, bool]:
    """Replicate alfworld_projection: extract <action>...</action>, lowercase,
    and validate that <think>...</think> is present and there are no CJK chars.

    Returns (action, valid).
    """
    lowered = raw_text.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = lowered.find(start_tag)
    end_idx = lowered.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        return raw_text.lower()[-30:], False
    extracted = lowered[start_idx + len(start_tag):end_idx].strip()
    valid = True
    if "<think>" not in raw_text or "</think>" not in raw_text:
        valid = False
    if re.search(r"[\u4e00-\u9fff]", raw_text):
        valid = False
    return extracted, valid


# ---------------------------------------------------------------------------
# Environment — scalar textworld env WITHOUT the expert wrapper (we ARE the agent)
# ---------------------------------------------------------------------------
def build_eval_env_factory(alfworld_data: str):
    import textworld
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredInfos,
    )

    def _make_env(gamefile: str):
        # NOTE: no AlfredExpert wrapper — we evaluate the model, not the oracle.
        wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
        request_infos = textworld.EnvInfos(
            won=True, admissible_commands=True, facts=True, extras=["gamefile"]
        )
        return textworld.start(gamefile, request_infos=request_infos, wrappers=wrappers)

    return _make_env


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(checkpoint: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_response(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
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


# ---------------------------------------------------------------------------
# Single-game rollout
# ---------------------------------------------------------------------------
def evaluate_one_game(
    env_factory,
    model,
    tokenizer,
    gamefile: str,
    history_length: int,
    max_steps: int,
    max_new_tokens: int,
    device: str,
) -> Dict[str, Any]:
    env = env_factory(gamefile)
    env.load(gamefile)
    game_state = env.reset()

    task_desc = ""
    feedback = game_state.get("feedback", "")
    task_start = feedback.find("Your task is to: ")
    if task_start != -1:
        task_desc = feedback[task_start + len("Your task is to: "):].strip()

    history: List[Dict[str, str]] = []
    won = False
    n_invalid = 0
    done = False
    n_step = 0
    try:
        while not done and n_step < max_steps:
            init = (n_step == 0)
            text_obs = game_state.get("feedback", "")
            admissible = list(game_state.get("admissible_commands", []))

            user_msg = build_user_message(
                task_desc=task_desc,
                text_obs=text_obs,
                admissible_actions=admissible,
                history=history,
                history_length=history_length,
                init=init,
            )
            messages = [{"role": "user", "content": user_msg}]
            raw = generate_response(model, tokenizer, messages, max_new_tokens)
            action, valid = project_action(raw)
            if not valid:
                n_invalid += 1

            game_state, _reward, done = env.step(action)
            history.append({"text_obs": text_obs, "action": action})
            n_step += 1
        won = bool(game_state.get("won", False))
    except Exception as e:
        print(f"  [WARN] eval failed on {gamefile}: {e}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    return {
        "won": won,
        "steps": n_step,
        "invalid_actions": n_invalid,
        "task_desc": task_desc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT ALFWorld checkpoint on fixed val set")
    parser.add_argument("--checkpoint", required=True, help="HF checkpoint dir (or model id)")
    parser.add_argument(
        "--val_file", default="fixed_val_set/alfworld.jsonl",
        help="Path to fixed_val_set/alfworld.jsonl",
    )
    parser.add_argument(
        "--alfworld_data",
        default=os.environ.get("ALFWORLD_DATA", str(Path.home() / ".cache/alfworld")),
    )
    parser.add_argument("--history_length", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_json", default=None, help="Where to dump per-game results")
    parser.add_argument("--limit", type=int, default=-1, help="Limit N games (debug)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load val cases
    with open(args.val_file) as f:
        cases = [json.loads(l) for l in f if l.strip()]
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"Loaded {len(cases)} val cases from {args.val_file}", flush=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model, tokenizer = load_model(args.checkpoint, args.device, dtype)
    print(f"Model loaded on {args.device} ({dtype})", flush=True)

    env_factory = build_eval_env_factory(args.alfworld_data)

    results = []
    by_family = defaultdict(lambda: {"won": 0, "total": 0})
    for i, case in enumerate(cases):
        gamefile = os.path.join(args.alfworld_data, "json_2.1.1", case["gamefile_relative"])
        if not os.path.exists(gamefile):
            print(f"[{i+1}/{len(cases)}] SKIP (missing): {gamefile}", flush=True)
            continue
        res = evaluate_one_game(
            env_factory, model, tokenizer, gamefile,
            args.history_length, args.max_steps, args.max_new_tokens, args.device,
        )
        res["case_id"] = case.get("case_id")
        res["task_family"] = case.get("task_family")
        res["gamefile"] = case["gamefile_relative"]
        results.append(res)
        by_family[res["task_family"]]["total"] += 1
        if res["won"]:
            by_family[res["task_family"]]["won"] += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(cases):
            n_done = len(results)
            n_won = sum(r["won"] for r in results)
            print(
                f"[{i+1}/{len(cases)}] done={n_done} success={n_won}/{n_done} "
                f"({100*n_won/max(n_done,1):.1f}%)",
                flush=True,
            )

    # Aggregate
    n_done = len(results)
    n_won = sum(r["won"] for r in results)
    overall = n_won / max(n_done, 1)
    avg_steps = sum(r["steps"] for r in results) / max(n_done, 1)
    total_invalid = sum(r["invalid_actions"] for r in results)

    print("\n=========== RESULTS ===========", flush=True)
    print(f"checkpoint : {args.checkpoint}", flush=True)
    print(f"games      : {n_done}", flush=True)
    print(f"success    : {n_won}/{n_done} = {overall:.4f}", flush=True)
    print(f"avg_steps  : {avg_steps:.2f}", flush=True)
    print(f"invalid_act: {total_invalid} (avg {total_invalid/max(n_done,1):.2f}/game)", flush=True)
    print("\n--- by task family ---", flush=True)
    for fam in sorted(by_family):
        s = by_family[fam]
        print(f"  {fam:45s} {s['won']:3d}/{s['total']:3d} = {s['won']/max(s['total'],1):.4f}", flush=True)

    summary = {
        "checkpoint": args.checkpoint,
        "val_file": args.val_file,
        "history_length": args.history_length,
        "max_steps": args.max_steps,
        "games": n_done,
        "success": n_won,
        "success_rate": overall,
        "avg_steps": round(avg_steps, 3),
        "total_invalid_actions": total_invalid,
        "by_family": {fam: dict(by_family[fam]) for fam in sorted(by_family)},
        "per_game": results,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote results to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
