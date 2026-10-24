#!/usr/bin/env python3
"""
Collect oracle (expert) trajectories from ALFWorld games and export them as a
**turn-level** SFT parquet dataset compatible with ``verl.trainer.fsdp_sft_trainer``
(``data.multiturn.enable=true``), where each row is a single (user, assistant)
turn — i.e. one decision step — rather than a full multi-turn trajectory.

Design goals
------------
* **Turn-level rows.** Each row = one RL decision step: the user message is the
  observation prompt (with ``history_length`` sliding window), the assistant
  message is the expert action. This matches the RL rollout's per-step input
  distribution and avoids long-context truncation issues.
* **Same prompts as RL.** The user message uses the *exact* templates from
  ``agent_system/environments/prompts/alfworld.py``
  (``ALFWORLD_TEMPLATE`` / ``ALFWORLD_TEMPLATE_NO_HIS``).
* **Schedule-weighted.** By default, each gamefile is replicated according to
  how many times it appears in the fixed training ``schedule.jsonl`` (so the
  SFT training distribution matches the RL sampling distribution). Use
  ``--dedupe`` to keep only unique gamefiles.
* **Expert labels.** The handcoded ALFWorld expert (``HandCodedTWAgent``)
  produces the target action at every step.

Usage
-----
    cd "$(git rev-parse --show-toplevel)/verl-agent"
    export ALFWORLD_DATA="$HOME/.cache/alfworld"
    python recipe/graph_guide/sft_baseline/collect_alfworld_oracle_sft_data.py \
        --schedule recipe/graph_guide/data/fixed_train_set/seed0/alfworld/schedule.jsonl \
        --output_dir recipe/graph_guide/sft_baseline/data/sft_alfworld/seed0 \
        --history_length 2

Outputs:
  - ``train.parquet``: turn-level SFT data (column ``messages``)
  - ``metadata.json``: collection statistics
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Prompt templates — kept in sync with agent_system/environments/prompts/alfworld.py
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
    """Mirror env_manager.py: exclude 'help', comma-join quoted strings."""
    return "\n ".join(f"'{s}'" for s in admissible_commands if s != "help")


def build_history_lines(history: List[Dict[str, str]], history_length: int) -> str:
    """Reproduce SimpleMemory.fetch formatting: [Observation N: '...', Action N: '...']"""
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
    """Build the user-side observation message identical to env_manager.build_text_obs."""
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


def build_assistant_message(expert_action: str) -> str:
    """Wrap the expert action in the <think>/<action> format the RL rollout expects."""
    return f"<think> The best action is to {expert_action}. </think>\n<action> {expert_action} </action>"


# ---------------------------------------------------------------------------
# Schedule loading with replication weights
# ---------------------------------------------------------------------------
def load_schedule_weighted(schedule_path: str, dedupe: bool = False) -> List[Tuple[Dict[str, Any], int]]:
    """Load gamefiles from a fixed_train_set schedule.jsonl.

    Returns a list of ``(task_dict, weight)`` where ``weight`` is the number of
    times the gamefile appears in the schedule (1 if ``dedupe=True``).
    """
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
                gf = task.get("gamefile_relative")
                if not gf:
                    continue
                if gf not in first_seen:
                    first_seen[gf] = task
                    order.append(gf)
                counts[gf] += 1
    if dedupe:
        return [(first_seen[gf], 1) for gf in order]
    return [(first_seen[gf], counts[gf]) for gf in order]


# ---------------------------------------------------------------------------
# Expert rollout
# ---------------------------------------------------------------------------
def build_alfworld_env_factory(alfworld_data: str):
    """Return a factory that creates a fresh textworld env for ONE gamefile,
    with the handcoded-expert wrapper attached (so ``extra.expert_plan`` is populated).

    We use ``textworld.start`` (the scalar, non-gym path) — exactly what ALFWorld's
    own ``is_solvable`` uses internally — so the expert wrapper sees scalar states
    rather than batched lists.
    """
    import textworld
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredInfos,
        AlfredExpert,
        AlfredExpertType,
    )

    def _make_env(gamefile: str):
        wrappers = [
            AlfredDemangler(shuffle=False),
            AlfredInfos,
            AlfredExpert(expert_type=AlfredExpertType.HANDCODED),
        ]
        request_infos = textworld.EnvInfos(
            won=True, admissible_commands=True, facts=True, extras=["gamefile", "expert_plan"]
        )
        return textworld.start(gamefile, request_infos=request_infos, wrappers=wrappers)

    return _make_env


def run_expert_on_game(env_factory, gamefile: str, max_steps: int = 50) -> Dict[str, Any] | None:
    """Run the handcoded expert on a single gamefile and collect a trajectory.

    Returns a dict with per-step observations, admissible actions, expert actions,
    the task description, and whether the expert succeeded. Returns None if the
    expert failed to produce a valid trajectory.
    """
    env = env_factory(gamefile)
    env.load(gamefile)
    game_state = env.reset()

    task_desc = ""
    feedback = game_state.get("feedback", "")
    task_start = feedback.find("Your task is to: ")
    if task_start != -1:
        task_desc = feedback[task_start + len("Your task is to: "):].strip()

    steps: List[Dict[str, Any]] = []
    done = False
    n_step = 0
    try:
        while not done and n_step < max_steps:
            expert_action = game_state["extra.expert_plan"][0]
            text_obs = game_state.get("feedback", "")
            admissible = list(game_state.get("admissible_commands", []))

            steps.append({
                "text_obs": text_obs,
                "admissible_actions": admissible,
                "expert_action": expert_action,
            })

            game_state, _reward, done = env.step(expert_action)
            n_step += 1
    except Exception as e:
        print(f"  [WARN] expert failed on {gamefile}: {e}", flush=True)
        try:
            env.close()
        except Exception:
            pass
        return None

    won = bool(game_state.get("won", False))
    try:
        env.close()
    except Exception:
        pass
    return {
        "task_desc": task_desc,
        "steps": steps,
        "won": won,
        "gamefile": gamefile,
    }


def trajectory_to_turn_level_rows(
    traj: Dict[str, Any], history_length: int, weight: int = 1,
) -> List[Dict[str, Any]]:
    """Convert an expert trajectory into a list of turn-level SFT rows.

    Each row has:
      - ``messages``: [{role: user, ...}, {role: assistant, ...}]
      - ``task_desc``, ``task_family``, ``gamefile``, ``step_idx``, ``won``
        (metadata columns useful for filtering / analysis)
    """
    rows: List[Dict[str, Any]] = []
    history: List[Dict[str, str]] = []
    for i, step in enumerate(traj["steps"]):
        init = (i == 0)
        user_msg = build_user_message(
            task_desc=traj["task_desc"],
            text_obs=step["text_obs"],
            admissible_actions=step["admissible_actions"],
            history=history,
            history_length=history_length,
            init=init,
        )
        assistant_msg = build_assistant_message(step["expert_action"])
        row = {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "task_desc": traj["task_desc"],
            "gamefile": traj["gamefile"],
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
    parser = argparse.ArgumentParser(description="Collect ALFWorld oracle SFT data (turn-level)")
    parser.add_argument(
        "--schedule", required=True,
        help="Path to recipe/graph_guide/data/fixed_train_set/seedN/alfworld/schedule.jsonl",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for train.parquet + metadata.json",
    )
    parser.add_argument("--history_length", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument(
        "--alfworld_data",
        default=os.environ.get("ALFWORLD_DATA", str(Path.home() / ".cache/alfworld")),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=-1,
        help="Limit number of unique games (for debugging). -1 = all.",
    )
    parser.add_argument(
        "--dedupe", action="store_true",
        help="Keep only unique gamefiles (weight=1), ignore schedule replication.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load the fixed training schedule (same games as RL) ---
    weighted_games = load_schedule_weighted(args.schedule, dedupe=args.dedupe)
    total_slots = sum(w for _, w in weighted_games)
    print(
        f"Loaded {len(weighted_games)} unique games "
        f"(total schedule slots={total_slots}) from {args.schedule}",
        flush=True,
    )
    if args.limit > 0:
        weighted_games = weighted_games[: args.limit]
        print(f"Limited to first {len(weighted_games)} games", flush=True)

    # --- Build the expert env factory ---
    env_factory = build_alfworld_env_factory(args.alfworld_data)

    # --- Collect expert trajectories (run each unique gamefile ONCE) ---
    all_rows: List[Dict[str, Any]] = []
    n_games = 0
    n_success = 0
    n_fail = 0
    total_expert_steps = 0
    for i, (task, weight) in enumerate(weighted_games):
        gamefile_rel = task["gamefile_relative"]
        gamefile = os.path.join(args.alfworld_data, "json_2.1.1", gamefile_rel)
        if not os.path.exists(gamefile):
            print(f"[{i+1}/{len(weighted_games)}] SKIP (missing): {gamefile}", flush=True)
            continue

        traj = run_expert_on_game(env_factory, gamefile, max_steps=args.max_steps)
        if traj is None or len(traj["steps"]) == 0:
            n_fail += 1
            continue

        rows = trajectory_to_turn_level_rows(traj, args.history_length, weight=weight)
        all_rows.extend(rows)
        n_games += 1
        total_expert_steps += len(traj["steps"])
        if traj["won"]:
            n_success += 1
        else:
            n_fail += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(weighted_games):
            print(
                f"[{i+1}/{len(weighted_games)}] games_done={n_games} "
                f"turn_rows={len(all_rows)} success={n_success} fail={n_fail}",
                flush=True,
            )

    avg_steps = total_expert_steps / max(n_games, 1)
    print(
        f"\nCollection done. games={n_games} success={n_success} fail={n_fail} "
        f"avg_steps={avg_steps:.1f} turn_rows={len(all_rows)}",
        flush=True,
    )

    # --- Write parquet ---
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(args.output_dir, "train.parquet")
    df.to_parquet(out_path)
    print(f"Wrote {len(df)} turn-level rows to {out_path}", flush=True)

    # --- Write metadata ---
    meta = {
        "schedule": os.path.abspath(args.schedule),
        "history_length": args.history_length,
        "max_steps": args.max_steps,
        "dedupe": args.dedupe,
        "unique_games": n_games,
        "schedule_slots": total_slots,
        "success_games": n_success,
        "failed_games": n_fail,
        "avg_expert_steps": round(avg_steps, 2),
        "total_expert_steps": total_expert_steps,
        "turn_level_rows": len(all_rows),
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {meta_path}", flush=True)


if __name__ == "__main__":
    main()
