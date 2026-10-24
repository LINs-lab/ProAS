#!/usr/bin/env python3
"""
Collect oracle trajectories from WebShop games and export them as a
**turn-level** SFT parquet dataset compatible with ``verl.trainer.fsdp_sft_trainer``
(``data.multiturn.enable=true``).

Unlike ALFWorld (which has a handcoded expert), WebShop has no rule-based
agent. Instead, the oracle is a **goal-driven policy**: the fixed training
schedule's ``goal`` field tells us the target ASIN, the search query, and the
required product options, so we can deterministically drive the WebShop
environment to the optimal purchase.

Oracle policy (state machine):
    SEARCH → (paginate until) FIND_TARGET → OPEN_PRODUCT → SELECT_OPTIONS → BUY

Each step records ``(text_obs, available_actions, expert_action)`` and is
converted to a turn-level ``(user, assistant)`` row, byte-identical to what the
RL rollout sees (``WEBSHOP_TEMPLATE``/``WEBSHOP_TEMPLATE_NO_HIS`` from
``agent_system/environments/prompts/webshop.py`` + ``SimpleMemory.fetch``).

Usage
-----
    cd "$(git rev-parse --show-toplevel)/verl-agent"
    python recipe/graph_guide/sft_baseline/collect_webshop_oracle_sft_data.py \
        --schedule fixed_train_set/seed0/webshop/schedule.jsonl \
        --output_dir data/sft_webshop/seed0 \
        --history_length 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Prompt templates — byte-identical to agent_system/environments/prompts/webshop.py
# ---------------------------------------------------------------------------
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def norm_text(value: Any) -> str:
    """Normalize text for fuzzy option matching (mirrors oracle_build.webshop_reference_graph.norm_text)."""
    text = "" if value is None else str(value)
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^a-z0-9$.'&|+:/ -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_history_lines(history: List[Dict[str, str]], history_length: int) -> str:
    """Reproduce SimpleMemory.fetch formatting."""
    recent = history[-history_length:] if history_length > 0 else []
    start_idx = len(history) - len(recent)
    lines = []
    for j, rec in enumerate(recent):
        step_num = start_idx + j + 1
        lines.append(
            f"[Observation {step_num}: '{rec['text_obs']}', Action {step_num}: '{rec['action']}']"
        )
    return "\n".join(lines)


def format_available_actions(avail: Dict[str, Any]) -> str:
    """Mirror WebshopEnvironmentManager.format_avail_actions + the comma-join in build_text_obs."""
    actions: List[str] = []
    if avail.get("has_search_bar"):
        actions.append("search[<your query>]")
    for txt in avail.get("clickables", []):
        actions.append(f"click[{txt}]")
    return "\n".join(f"'{s}'," for s in actions)


def format_obs(raw_obs: str, task_desc: str) -> str:
    """Mirror WebshopEnvironmentManager.format_obs: strip instruction prefix.

    RL splits the [SEP]-joined text and drops everything up to and including
    the task description, keeping only the page content after it.
    """
    parts = raw_obs.split(" [SEP] ")
    try:
        index = parts.index(task_desc)
        return " [SEP] ".join(f"'{p}'" for p in parts[index + 1:])
    except ValueError:
        return raw_obs


def build_user_message(
    task_desc: str,
    raw_obs: str,
    avail: Dict[str, Any],
    history: List[Dict[str, str]],
    history_length: int,
    init: bool,
) -> str:
    text_obs = format_obs(raw_obs, task_desc)
    available_actions = format_available_actions(avail)
    if init or history_length <= 0:
        return WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task_desc,
            current_observation=text_obs,
            available_actions=available_actions,
        )
    action_history = build_history_lines(history, history_length)
    valid_len = min(len(history), history_length)
    return WEBSHOP_TEMPLATE.format(
        task_description=task_desc,
        step_count=len(history),
        history_length=valid_len,
        action_history=action_history,
        current_step=len(history) + 1,
        current_observation=text_obs,
        available_actions=available_actions,
    )


def build_assistant_message(action: str, reason: str) -> str:
    return f"<think> {reason} </think>\n<action> {action} </action>"


# ---------------------------------------------------------------------------
# Schedule loading with replication weights
# ---------------------------------------------------------------------------
def load_schedule_weighted(schedule_path: str, dedupe: bool = False) -> List[Tuple[Dict[str, Any], int]]:
    """Load webshop tasks from schedule.jsonl, returning (task, weight) pairs."""
    counts: Counter = Counter()
    first_seen: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    with open(schedule_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            for task in entry.get("tasks", []):
                key = task.get("case_id") or task.get("exact_task_key")
                if not key:
                    continue
                if key not in first_seen:
                    first_seen[key] = task
                    order.append(key)
                counts[key] += 1
    if dedupe:
        return [(first_seen[k], 1) for k in order]
    return [(first_seen[k], counts[k]) for k in order]


# ---------------------------------------------------------------------------
# Oracle policy (goal-driven state machine)
# ---------------------------------------------------------------------------
def decide_oracle_action(
    avail: Dict[str, Any],
    goal: Dict[str, Any],
    state: Dict[str, Any],
    raw_obs: str,
    max_paginate: int = 10,
) -> Tuple[str, str]:
    """Decide the oracle action given current available actions and goal.

    Returns (action, reason).
    State machine: search → find target → open → select options → buy.
    """
    clickables = list(avail.get("clickables", []))
    has_search = avail.get("has_search_bar", False)
    target_asin = str(goal.get("asin", "")).upper()
    target_options = goal.get("goal_options") or {}
    target_values = {norm_text(v) for v in target_options.values()}
    selected = state.get("selected_options", set())
    product_opened = state.get("product_opened", False)

    clickable_norm = {norm_text(c): c for c in clickables}

    # Phase 1: search
    if has_search and not state.get("searched", False):
        query = goal.get("query") or goal.get("name") or ""
        query = norm_text(query)
        if query:
            return f"search[{query}]", f"Search for the target product using query '{query}'."

    # Phase 2: find and open target product
    if not product_opened:
        # Check if target asin is directly clickable (on this page)
        for c in clickables:
            if norm_text(c) == norm_text(target_asin):
                return f"click[{c}]", f"Open the target product {target_asin}."
        # Also check if asin appears as a clickable (sometimes lowercased)
        if norm_text(target_asin) in clickable_norm:
            c = clickable_norm[norm_text(target_asin)]
            return f"click[{c}]", f"Open the target product {target_asin}."
        # Paginate
        if state.get("paginate_count", 0) < max_paginate:
            next_norm = norm_text("next >")
            for c in clickables:
                if norm_text(c) == next_norm:
                    return "click[next >]", "Go to the next page of search results to find the target product."
        # Target not found; fallback: try a refined search from product name keywords
        if has_search:
            name = goal.get("name") or ""
            # Use first few words of the product name as fallback query
            words = norm_text(name).split()[:3]
            if words:
                query = " ".join(words)
                return f"search[{query}]", f"Target not found in results; refine search to '{query}'."
        return "click[buy now]", "Unable to locate target; no better action available."

    # Phase 3: select required options (unordered)
    for key, value in target_options.items():
        nv = norm_text(value)
        if nv in selected:
            continue
        # Match the option value against clickables
        if nv in clickable_norm:
            c = clickable_norm[nv]
            return f"click[{c}]", f"Select required option {key}={value}."
        # Fuzzy: check if any clickable contains the value
        for cn, c in clickable_norm.items():
            if nv and nv in cn:
                return f"click[{c}]", f"Select required option {key}={value} (fuzzy match)."

    # Phase 4: buy
    for c in clickables:
        if norm_text(c) == "buy now":
            return "click[buy now]", "All required options selected; purchase the target product."
    # If buy not available, maybe we need to be on the product page
    return "click[buy now]", "Attempt to purchase."


def update_state(state: Dict[str, Any], action: str, avail: Dict[str, Any], goal: Dict[str, Any]) -> Dict[str, Any]:
    """Update oracle state after taking an action."""
    new_state = dict(state)
    action_lower = action.lower()
    target_asin = str(goal.get("asin", "")).upper()
    target_options = goal.get("goal_options") or {}

    if action_lower.startswith("search["):
        new_state["searched"] = True
        new_state["paginate_count"] = 0
    elif action_lower.startswith("click["):
        arg = action[6:-1]  # strip "click[" and "]"
        arg_norm = norm_text(arg)
        if arg_norm == norm_text(target_asin):
            new_state["product_opened"] = True
        elif arg_norm == "next >":
            new_state["paginate_count"] = state.get("paginate_count", 0) + 1
        else:
            # Check if it's a target option value
            for key, value in target_options.items():
                if norm_text(value) == arg_norm:
                    selected = set(state.get("selected_options", set()))
                    selected.add(arg_norm)
                    new_state["selected_options"] = selected
                    break
    return new_state


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
WEBSHOP_PKG = os.path.join(
    REPO_ROOT, "agent_system", "environments", "env_package", "webshop", "webshop"
)


def build_webshop_env():
    """Create a WebAgentTextEnv with the same config as RL training."""
    import sys as _sys
    import gym

    if WEBSHOP_PKG not in _sys.path:
        _sys.path.insert(0, WEBSHOP_PKG)

    # Importing the envs module registers 'WebAgentTextEnv-v0' with gym.
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401

    data_dir = os.path.join(WEBSHOP_PKG, "data")
    file_path = os.path.join(data_dir, "items_shuffle_1000.json")
    attr_path = os.path.join(data_dir, "items_ins_v2_1000.json")

    env = gym.make(
        'WebAgentTextEnv-v0',
        observation_mode='text',
        num_products=None,
        human_goals=False,
        file_path=file_path,
        attr_path=attr_path,
    )
    return env


def install_goal(env, fixed_case: Dict[str, Any]) -> str:
    """Install a fixed goal into the WebShop env, mirroring install_fixed_webshop_train_goal."""
    import copy
    case_id = str(fixed_case["case_id"])
    session_suffix = re.sub(r"[^a-zA-Z0-9]+", "-", case_id).strip("-")
    session_id = f"fixed-train-{session_suffix}"
    env.server.user_sessions[session_id] = {
        "goal": copy.deepcopy(fixed_case["goal"]),
        "done": False,
    }
    return session_id


# ---------------------------------------------------------------------------
# Single-task rollout
# ---------------------------------------------------------------------------
def run_oracle_on_task(env, fixed_case: Dict[str, Any], max_steps: int = 15) -> Optional[Dict[str, Any]]:
    """Run the oracle policy on one fixed task. Returns trajectory or None."""
    goal = fixed_case["goal"]
    task_desc = fixed_case.get("task_desc") or goal.get("instruction_text", "")

    session_id = install_goal(env, fixed_case)
    obs, _ = env.reset(session=session_id)

    state: Dict[str, Any] = {
        "searched": False,
        "product_opened": False,
        "selected_options": set(),
        "paginate_count": 0,
    }
    steps: List[Dict[str, Any]] = []
    done = False
    n_step = 0
    final_reward = 0.0
    won = False

    try:
        while not done and n_step < max_steps:
            avail = env.get_available_actions()
            action, reason = decide_oracle_action(avail, goal, state, obs)
            steps.append({
                "text_obs": obs,
                "available_actions": dict(avail),
                "expert_action": action,
                "reason": reason,
            })
            obs, reward, done, info = env.step(action)
            final_reward = float(reward or 0.0)
            info = dict(info or {})
            won = bool(info.get("won", False)) or final_reward >= 1.0
            state = update_state(state, action, avail, goal)
            n_step += 1
    except Exception as e:
        print(f"  [WARN] oracle failed on {fixed_case.get('case_id')}: {e}", flush=True)
        return None

    return {
        "task_desc": task_desc,
        "steps": steps,
        "won": won,
        "reward": final_reward,
        "asin": goal.get("asin"),
        "goal_options": goal.get("goal_options"),
        "case_id": fixed_case.get("case_id"),
    }


def trajectory_to_turn_level_rows(
    traj: Dict[str, Any], history_length: int, weight: int = 1,
) -> List[Dict[str, Any]]:
    """Convert an oracle trajectory into turn-level SFT rows."""
    rows: List[Dict[str, Any]] = []
    history: List[Dict[str, str]] = []
    for i, step in enumerate(traj["steps"]):
        init = (i == 0)
        user_msg = build_user_message(
            task_desc=traj["task_desc"],
            raw_obs=step["text_obs"],
            avail=step["available_actions"],
            history=history,
            history_length=history_length,
            init=init,
        )
        assistant_msg = build_assistant_message(step["expert_action"], step.get("reason", ""))
        row = {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "task_desc": traj["task_desc"],
            "case_id": traj.get("case_id"),
            "step_idx": i,
            "won": traj["won"],
            "weight": weight,
        }
        rows.append(row)
        history.append({"text_obs": step["text_obs"], "action": step["expert_action"]})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect WebShop oracle SFT data (turn-level)")
    parser.add_argument("--schedule", required=True, help="Path to fixed_train_set/seedN/webshop/schedule.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--history_length", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--limit", type=int, default=-1, help="Limit unique tasks (debug)")
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    weighted_tasks = load_schedule_weighted(args.schedule, dedupe=args.dedupe)
    total_slots = sum(w for _, w in weighted_tasks)
    print(f"Loaded {len(weighted_tasks)} unique tasks (slots={total_slots}) from {args.schedule}", flush=True)
    if args.limit > 0:
        weighted_tasks = weighted_tasks[: args.limit]
        print(f"Limited to first {len(weighted_tasks)} tasks", flush=True)

    env = build_webshop_env()
    print("WebShop env built.", flush=True)

    all_rows: List[Dict[str, Any]] = []
    n_tasks = 0
    n_success = 0
    n_fail = 0
    total_steps = 0

    for i, (task, weight) in enumerate(weighted_tasks):
        traj = run_oracle_on_task(env, task, max_steps=args.max_steps)
        if traj is None or len(traj["steps"]) == 0:
            n_fail += 1
            continue
        rows = trajectory_to_turn_level_rows(traj, args.history_length, weight=weight)
        all_rows.extend(rows)
        n_tasks += 1
        total_steps += len(traj["steps"])
        if traj["won"]:
            n_success += 1
        else:
            n_fail += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(weighted_tasks):
            print(
                f"[{i+1}/{len(weighted_tasks)}] tasks_done={n_tasks} "
                f"turn_rows={len(all_rows)} success={n_success} fail={n_fail} "
                f"avg_steps={total_steps/max(n_tasks,1):.1f}",
                flush=True,
            )

    avg_steps = total_steps / max(n_tasks, 1)
    print(
        f"\nCollection done. tasks={n_tasks} success={n_success} fail={n_fail} "
        f"avg_steps={avg_steps:.1f} turn_rows={len(all_rows)}",
        flush=True,
    )

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(args.output_dir, "train.parquet")
    df.to_parquet(out_path)
    print(f"Wrote {len(df)} turn-level rows to {out_path}", flush=True)

    meta = {
        "schedule": os.path.abspath(args.schedule),
        "history_length": args.history_length,
        "max_steps": args.max_steps,
        "dedupe": args.dedupe,
        "unique_tasks": n_tasks,
        "schedule_slots": total_slots,
        "success_tasks": n_success,
        "failed_tasks": n_fail,
        "avg_oracle_steps": round(avg_steps, 2),
        "turn_level_rows": len(all_rows),
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {meta_path}", flush=True)


if __name__ == "__main__":
    main()
