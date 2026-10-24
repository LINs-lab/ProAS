from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from contextlib import suppress
from copy import deepcopy
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]


def load_traj_data(trial_dir: Path) -> Dict[str, Any]:
    return json.loads((trial_dir / "traj_data.json").read_text())


def load_game_data(trial_dir: Path) -> Dict[str, Any]:
    return json.loads((trial_dir / "game.tw-pddl").read_text())


def load_walkthrough(trial_dir: Path) -> List[str]:
    game_path = trial_dir / "game.tw-pddl"
    if not game_path.exists():
        return []
    return list(load_game_data(trial_dir).get("walkthrough", []))


def get_task_targets(traj: Dict[str, Any]) -> Dict[str, str]:
    params = traj["pddl_params"]
    return {
        "object": params.get("object_target", "") or "",
        "parent": params.get("parent_target", "") or "",
        "toggle": params.get("toggle_target", "") or "",
        "mrecep": params.get("mrecep_target", "") or "",
    }


def normalize_symbol(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def normalize_entity_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_base_name(value: str) -> str:
    value = normalize_entity_label(value)
    value = re.sub(r"\s+\d+$", "", value)
    return value


def strip_markup(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"<[^>]+>", "", str(value))


def normalize_rollout_label(value: Any) -> str:
    return normalize_entity_label(strip_markup(value))


def _text_mentions_base(text: str, base_name: str) -> bool:
    normalized_text = normalize_entity_label(strip_markup(text))
    normalized_base = normalize_base_name(base_name)
    if not normalized_base:
        return False
    return normalized_base in normalized_text


def _extract_instance_label_from_text(text: str, base_name: str) -> Optional[str]:
    normalized_base = normalize_base_name(base_name)
    if not normalized_base:
        return None
    normalized_text = normalize_entity_label(strip_markup(text))
    match = re.search(rf"\b{re.escape(normalized_base)}\s+\d+\b", normalized_text)
    if match:
        return match.group(0)
    return None


def extract_current_observation_text(text: Any) -> str:
    normalized_text = normalize_entity_label(strip_markup(text))
    if not normalized_text:
        return ""
    patterns = [
        r"your current observation is:\s*(.+?)(?:\s*your task is:|\s*your admissible actions)",
        r"current observation is:\s*(.+?)(?:\s*your admissible actions)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized_text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return normalized_text


def role_for_argument(arg: str, traj: Dict[str, Any]) -> str:
    norm_arg = normalize_symbol(arg)
    targets = get_task_targets(traj)
    for role_name, target in targets.items():
        if target and norm_arg == normalize_symbol(target):
            return {
                "object": "TARGET_OBJ",
                "parent": "TARGET_PARENT",
                "toggle": "TARGET_TOGGLE",
                "mrecep": "TARGET_MRECEP",
            }[role_name]
    return f"LITERAL::{norm_arg or 'unknown'}"


def binding_for_argument(arg: str, traj: Dict[str, Any]) -> Dict[str, str]:
    role = role_for_argument(arg, traj)
    entity = normalize_symbol(arg) or "unknown"
    return {"role": role, "entity": entity}


def format_binding(role: Optional[str], entity: Optional[str]) -> str:
    if role is None:
        return "none"
    if entity:
        return f"{role}[{entity}]"
    return role


def infer_goto_binding(
    high_steps: Sequence[Dict[str, Any]],
    idx: int,
    traj: Dict[str, Any],
    alias_override: Optional[str] = None,
) -> Dict[str, str]:
    inert_actions = {"ExamineObject", "Look", "Inventory"}
    args = high_steps[idx]["discrete_action"].get("args") or ["unknown"]
    raw_arg = args[0] if args else "unknown"
    raw_entity = normalize_entity_label(alias_override) if alias_override else (normalize_symbol(raw_arg) or "unknown")
    fallback = binding_for_argument(raw_arg, traj)
    for look_ahead in range(idx + 1, len(high_steps)):
        next_step = high_steps[look_ahead]["discrete_action"]
        next_action = next_step["action"]
        if next_action == "NoOp" or next_action in inert_actions:
            continue
        next_roles = [role_for_argument(arg, traj) for arg in next_step.get("args", [])]
        if next_action == "PickupObject":
            if next_roles and next_roles[0] == "TARGET_OBJ":
                return {"role": "OBJECT_SOURCE", "entity": raw_entity}
            if next_roles and next_roles[0] == "TARGET_MRECEP":
                return {"role": "MRECEP_SOURCE", "entity": raw_entity}
        if next_action == "PutObject":
            if len(next_roles) >= 2 and next_roles[1] == "TARGET_PARENT":
                return {"role": "GOAL_RECEP", "entity": raw_entity}
            if len(next_roles) >= 2 and next_roles[1] == "TARGET_MRECEP":
                return {"role": "MRECEP_CONTAINER", "entity": raw_entity}
        if next_action == "ToggleObject":
            return {"role": "TOGGLE_SOURCE", "entity": raw_entity}
        if next_action == "CleanObject":
            return {"role": "CLEAN_STATION", "entity": raw_entity}
        if next_action == "HeatObject":
            return {"role": "HEAT_STATION", "entity": raw_entity}
        if next_action == "CoolObject":
            return {"role": "COOL_STATION", "entity": raw_entity}
        if next_action == "SliceObject":
            return {"role": "SLICE_STATION", "entity": raw_entity}
        break
    if fallback["role"].startswith("TARGET_"):
        return fallback
    return {"role": "LOC", "entity": raw_entity}


def abstract_high_step(
    high_steps: Sequence[Dict[str, Any]],
    idx: int,
    traj: Dict[str, Any],
    alias_map: Optional[Dict[int, List[str]]] = None,
    binding_overrides: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    step = high_steps[idx]["discrete_action"]
    action = step["action"]
    raw_args = list(step.get("args", []))
    aliases = list((alias_map or {}).get(idx, raw_args))
    if action == "GotoLocation":
        override = (binding_overrides or {}).get(idx)
        if override is not None:
            bindings = [
                {
                    "role": override.get("role", "LOC"),
                    "entity": normalize_entity_label(override.get("entity") or (aliases[0] if aliases else "unknown")),
                }
            ]
        else:
            alias_override = aliases[0] if aliases else None
            bindings = [infer_goto_binding(high_steps, idx, traj, alias_override=alias_override)]
    else:
        bindings = [
            {"role": role_for_argument(arg, traj), "entity": normalize_entity_label(alias)}
            for arg, alias in zip(raw_args, aliases)
        ]
    return {
        "high_idx": high_steps[idx]["high_idx"],
        "action": action,
        "raw_args": raw_args,
        "roles": [binding["role"] for binding in bindings],
        "entities": [binding["entity"] for binding in bindings],
        "bindings": bindings,
        "raw_step": high_steps[idx],
    }


def initial_abstract_state() -> Dict[str, Any]:
    return {
        "agent_location_role": "START",
        "agent_location_entity": None,
        "holding_role": None,
        "holding_entity": None,
        "placed_count": 0,
        "target_container_role": None,
        "target_container_entity": None,
        "mrecep_container_role": None,
        "mrecep_container_entity": None,
        "mrecep_at_goal": False,
        "target_in_mrecep": False,
        "object_clean": False,
        "object_hot": False,
        "object_cool": False,
        "object_sliced": False,
        "toggle_on": False,
        "toggled_role": None,
        "toggled_entity": None,
        "success": False,
    }


def abstract_goal_satisfied(traj: Dict[str, Any], state: Dict[str, Any]) -> bool:
    task_type = traj["task_type"]
    needs_slice = bool(traj["pddl_params"].get("object_sliced"))
    sliced_ok = (not needs_slice) or state["object_sliced"]

    if task_type == "pick_and_place_simple":
        return state["placed_count"] >= 1 and sliced_ok
    if task_type == "look_at_obj_in_light":
        return (
            state["holding_role"] == "TARGET_OBJ"
            and state["toggle_on"]
            and state["agent_location_role"] == "TOGGLE_SOURCE"
            and sliced_ok
        )
    if task_type == "pick_clean_then_place_in_recep":
        return state["placed_count"] >= 1 and state["object_clean"] and sliced_ok
    if task_type == "pick_heat_then_place_in_recep":
        return state["placed_count"] >= 1 and state["object_hot"] and sliced_ok
    if task_type == "pick_cool_then_place_in_recep":
        return state["placed_count"] >= 1 and state["object_cool"] and sliced_ok
    if task_type == "pick_two_obj_and_place":
        return state["placed_count"] >= 2 and sliced_ok
    if task_type == "pick_and_place_with_movable_recep":
        return state["mrecep_at_goal"] and state["target_in_mrecep"] and sliced_ok
    return False


def advance_abstract_state(
    traj: Dict[str, Any],
    state: Dict[str, Any],
    abstract_step_info: Dict[str, Any],
) -> Dict[str, Any]:
    next_state = deepcopy(state)
    action = abstract_step_info["action"]
    bindings = abstract_step_info["bindings"]

    if action == "GotoLocation":
        binding = bindings[0]
        next_state["agent_location_role"] = binding["role"]
        next_state["agent_location_entity"] = binding["entity"]
    elif action == "PickupObject":
        binding = bindings[0] if bindings else {"role": None, "entity": None}
        if binding["role"] in {"TARGET_OBJ", "TARGET_MRECEP"}:
            next_state["holding_role"] = binding["role"]
            next_state["holding_entity"] = binding["entity"]
        if binding["role"] == "TARGET_OBJ":
            next_state["target_container_role"] = None
            next_state["target_container_entity"] = None
        elif binding["role"] == "TARGET_MRECEP":
            next_state["mrecep_container_role"] = None
            next_state["mrecep_container_entity"] = None
    elif action == "PutObject":
        moved = bindings[0] if len(bindings) >= 1 else {"role": None, "entity": None}
        dest = bindings[1] if len(bindings) >= 2 else {"role": None, "entity": None}
        moved_role = moved["role"]
        dest_role = dest["role"]
        if moved_role == "TARGET_OBJ" and dest_role == "TARGET_PARENT":
            next_state["placed_count"] += 1
            next_state["target_container_role"] = dest["role"]
            next_state["target_container_entity"] = dest["entity"]
            next_state["holding_role"] = None
            next_state["holding_entity"] = None
        elif moved_role == "TARGET_OBJ" and dest_role == "TARGET_MRECEP":
            next_state["target_in_mrecep"] = True
            next_state["target_container_role"] = dest["role"]
            next_state["target_container_entity"] = dest["entity"]
            next_state["holding_role"] = None
            next_state["holding_entity"] = None
        elif moved_role == "TARGET_MRECEP" and dest_role == "TARGET_PARENT":
            next_state["mrecep_at_goal"] = True
            next_state["mrecep_container_role"] = dest["role"]
            next_state["mrecep_container_entity"] = dest["entity"]
            next_state["holding_role"] = None
            next_state["holding_entity"] = None
        else:
            next_state["holding_role"] = None
            next_state["holding_entity"] = None
    elif action == "ToggleObject":
        if bindings and bindings[0]["role"] == "TARGET_TOGGLE":
            next_state["toggle_on"] = True
            next_state["toggled_role"] = bindings[0]["role"]
            next_state["toggled_entity"] = bindings[0]["entity"]
    elif action == "CleanObject":
        if bindings and bindings[0]["role"] == "TARGET_OBJ":
            next_state["object_clean"] = True
    elif action == "HeatObject":
        if bindings and bindings[0]["role"] == "TARGET_OBJ":
            next_state["object_hot"] = True
            next_state["object_cool"] = False
    elif action == "CoolObject":
        if bindings and bindings[0]["role"] == "TARGET_OBJ":
            next_state["object_cool"] = True
            next_state["object_hot"] = False
    elif action == "SliceObject":
        if bindings and bindings[0]["role"] == "TARGET_OBJ":
            next_state["object_sliced"] = True

    next_state["success"] = abstract_goal_satisfied(traj, next_state)
    return next_state


def state_signature(state: Dict[str, Any], include_entities: bool = True) -> Tuple[str, ...]:
    pieces: List[str] = []
    location_value = format_binding(state["agent_location_role"], state["agent_location_entity"])
    if not include_entities and state["agent_location_role"] is not None:
        location_value = state["agent_location_role"]
    if state["agent_location_role"] != "START":
        pieces.append(f"at:{location_value}")
    if state["holding_role"] is not None:
        holding_value = format_binding(state["holding_role"], state["holding_entity"])
        if not include_entities:
            holding_value = state["holding_role"]
        pieces.append(f"holding:{holding_value}")
    if state["object_sliced"]:
        pieces.append("object:sliced[TARGET_OBJ]")
    if state["object_clean"]:
        pieces.append("object:clean[TARGET_OBJ]")
    if state["object_hot"]:
        pieces.append("object:hot[TARGET_OBJ]")
    if state["object_cool"]:
        pieces.append("object:cool[TARGET_OBJ]")
    if state["toggle_on"]:
        toggled_value = state["toggled_entity"] or "target_toggle"
        if not include_entities:
            toggled_value = state["toggled_role"] or "TARGET_TOGGLE"
        pieces.append(f"toggle:on[{toggled_value}]")
    if state["placed_count"]:
        container = format_binding(state["target_container_role"], state["target_container_entity"])
        if not include_entities:
            container = state["target_container_role"] or "TARGET_PARENT"
        pieces.append(f"placed:{state['placed_count']}@{container}")
    if state["mrecep_at_goal"]:
        container = format_binding(state["mrecep_container_role"], state["mrecep_container_entity"])
        if not include_entities:
            container = state["mrecep_container_role"] or "TARGET_PARENT"
        pieces.append(f"mrecep@goal:{container}")
    if state["target_in_mrecep"]:
        container = format_binding(state["target_container_role"], state["target_container_entity"])
        if not include_entities:
            container = state["target_container_role"] or "TARGET_MRECEP"
        pieces.append(f"target@mrecep:{container}")
    if state["success"]:
        pieces.append("success")
    return tuple(pieces) if pieces else ("start",)


def _rollout_action_args(row: Dict[str, Any]) -> List[str]:
    raw_args = row.get("executed_action_args", [])
    if not isinstance(raw_args, list):
        return []
    return [normalize_rollout_label(arg) for arg in raw_args if normalize_rollout_label(arg)]


def _rollout_action_type(row: Dict[str, Any]) -> str:
    return str(row.get("executed_action_type") or "")


def infer_rollout_target_instance_labels(
    rollout_rows: Sequence[Dict[str, Any]],
    traj: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    targets = get_task_targets(traj)
    labels: Dict[str, Optional[str]] = {}
    for role_name, raw_target in targets.items():
        target_base = normalize_base_name(raw_target)
        label: Optional[str] = None
        if target_base:
            for row in rollout_rows:
                for arg in _rollout_action_args(row):
                    if normalize_base_name(arg) == target_base:
                        label = arg
                        break
                if label is not None:
                    break
        if label is None and target_base:
            for row in rollout_rows:
                for key in ("obs_before_text", "obs_after_text", "obs_before", "obs_after", "raw_action_text"):
                    value = row.get(key)
                    if not isinstance(value, str):
                        continue
                    label = _extract_instance_label_from_text(value, target_base)
                    if label is not None:
                        break
                if label is not None:
                    break
        labels[role_name] = label or (normalize_entity_label(raw_target) if raw_target else None)
    return labels


def _build_rollout_location_segments(
    rollout_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    current_segment: Optional[Dict[str, Any]] = None

    for idx, row in enumerate(rollout_rows):
        action_type = _rollout_action_type(row)
        action_args = _rollout_action_args(row)
        if action_type == "GotoLocation" and action_args:
            if current_segment is not None:
                segments.append(current_segment)
            current_segment = {
                "goto_idx": idx,
                "traj_step_idx": int(row.get("traj_step_idx", idx)),
                "location_label": action_args[0],
                "rows": [],
            }

        if current_segment is None:
            current_segment = {
                "goto_idx": None,
                "traj_step_idx": None,
                "location_label": None,
                "rows": [],
            }
        current_segment["rows"].append(row)

    if current_segment is not None:
        segments.append(current_segment)

    return segments


def _target_role_metadata(traj: Dict[str, Any], target_labels: Dict[str, Optional[str]]) -> Dict[str, Dict[str, Any]]:
    raw_targets = get_task_targets(traj)
    return {
        "object": {
            "base": normalize_base_name(raw_targets.get("object", "")),
            "label": target_labels.get("object"),
        },
        "parent": {
            "base": normalize_base_name(raw_targets.get("parent", "")),
            "label": target_labels.get("parent"),
        },
        "toggle": {
            "base": normalize_base_name(raw_targets.get("toggle", "")),
            "label": target_labels.get("toggle"),
        },
        "mrecep": {
            "base": normalize_base_name(raw_targets.get("mrecep", "")),
            "label": target_labels.get("mrecep"),
        },
    }


def _segment_observation_text(segment: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for row in segment.get("rows", []):
        for key in ("obs_after_text", "obs_after"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                current_obs = extract_current_observation_text(value)
                if current_obs:
                    pieces.append(current_obs)
    return "\n".join(pieces)


def infer_rollout_goto_binding_overrides(
    rollout_rows: Sequence[Dict[str, Any]],
    traj: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    target_labels = infer_rollout_target_instance_labels(rollout_rows, traj)
    target_meta = _target_role_metadata(traj, target_labels)
    segments = _build_rollout_location_segments(rollout_rows)

    vote_priority = [
        "OBJECT_SOURCE",
        "TOGGLE_SOURCE",
        "GOAL_RECEP",
        "MRECEP_SOURCE",
        "MRECEP_CONTAINER",
        "CLEAN_STATION",
        "HEAT_STATION",
        "COOL_STATION",
        "SLICE_STATION",
        "LOC",
    ]
    priority_rank = {role: idx for idx, role in enumerate(vote_priority)}

    overrides: Dict[int, Dict[str, Any]] = {}
    for segment in segments:
        goto_idx = segment.get("goto_idx")
        raw_location = segment.get("location_label")
        if goto_idx is None or not raw_location:
            continue

        observation_text = _segment_observation_text(segment)
        object_visible_here = bool(
            target_meta["object"]["label"] and _text_mentions_base(observation_text, target_meta["object"]["label"])
        )
        toggle_visible_here = bool(
            target_meta["toggle"]["label"] and _text_mentions_base(observation_text, target_meta["toggle"]["label"])
        )
        votes: List[Dict[str, Any]] = []

        def add_vote(role: str, entity: Optional[str], weight: int, reason: str) -> None:
            final_entity = normalize_entity_label(entity or raw_location)
            votes.append(
                {
                    "role": role,
                    "entity": final_entity,
                    "weight": weight,
                    "reason": reason,
                }
            )

        for row in segment.get("rows", [])[1:]:
            action_type = _rollout_action_type(row)
            action_args = _rollout_action_args(row)
            arg_bases = [normalize_base_name(arg) for arg in action_args]

            if action_type == "PickupObject" and arg_bases:
                if target_meta["object"]["base"] and arg_bases[0] == target_meta["object"]["base"]:
                    add_vote("OBJECT_SOURCE", raw_location, 6, "future pickup of target object under this location")
                if target_meta["mrecep"]["base"] and arg_bases[0] == target_meta["mrecep"]["base"]:
                    add_vote("MRECEP_SOURCE", raw_location, 6, "future pickup of target movable receptacle under this location")
            elif action_type == "ToggleObject" and arg_bases:
                if target_meta["toggle"]["base"] and arg_bases[0] == target_meta["toggle"]["base"]:
                    add_vote(
                        "TOGGLE_SOURCE",
                        target_meta["toggle"]["label"] or action_args[0],
                        6,
                        "future toggle of target light/tool after arriving here",
                    )
            elif action_type == "PutObject" and len(arg_bases) >= 2:
                moved_base, dest_base = arg_bases[0], arg_bases[1]
                if (
                    target_meta["object"]["base"]
                    and target_meta["parent"]["base"]
                    and moved_base == target_meta["object"]["base"]
                    and dest_base == target_meta["parent"]["base"]
                ):
                    add_vote(
                        "GOAL_RECEP",
                        target_meta["parent"]["label"] or action_args[1],
                        6,
                        "future placement into the goal receptacle after arriving here",
                    )
                if (
                    target_meta["object"]["base"]
                    and target_meta["mrecep"]["base"]
                    and moved_base == target_meta["object"]["base"]
                    and dest_base == target_meta["mrecep"]["base"]
                ):
                    add_vote(
                        "MRECEP_CONTAINER",
                        target_meta["mrecep"]["label"] or action_args[1],
                        6,
                        "future placement of target object into the movable receptacle after arriving here",
                    )
                if (
                    target_meta["mrecep"]["base"]
                    and target_meta["parent"]["base"]
                    and moved_base == target_meta["mrecep"]["base"]
                    and dest_base == target_meta["parent"]["base"]
                ):
                    add_vote(
                        "GOAL_RECEP",
                        target_meta["parent"]["label"] or action_args[1],
                        6,
                        "future placement of the movable receptacle into the goal receptacle after arriving here",
                    )
            elif action_type == "CleanObject" and arg_bases and target_meta["object"]["base"] and arg_bases[0] == target_meta["object"]["base"]:
                add_vote("CLEAN_STATION", raw_location, 6, "future clean action on target object after arriving here")
            elif action_type == "HeatObject" and arg_bases and target_meta["object"]["base"] and arg_bases[0] == target_meta["object"]["base"]:
                add_vote("HEAT_STATION", raw_location, 6, "future heat action on target object after arriving here")
            elif action_type == "CoolObject" and arg_bases and target_meta["object"]["base"] and arg_bases[0] == target_meta["object"]["base"]:
                add_vote("COOL_STATION", raw_location, 6, "future cool action on target object after arriving here")
            elif action_type == "SliceObject" and arg_bases and target_meta["object"]["base"] and arg_bases[0] == target_meta["object"]["base"]:
                add_vote("SLICE_STATION", raw_location, 6, "future slice action on target object after arriving here")
            elif action_type == "ExamineObject":
                joined_args = " ".join(action_args)
                mentions_target_object = bool(
                    target_meta["object"]["base"] and _text_mentions_base(joined_args, target_meta["object"]["base"])
                )
                mentions_target_toggle = bool(
                    target_meta["toggle"]["base"] and _text_mentions_base(joined_args, target_meta["toggle"]["base"])
                )
                if mentions_target_toggle:
                    add_vote(
                        "TOGGLE_SOURCE",
                        target_meta["toggle"]["label"] or raw_location,
                        4 if toggle_visible_here else 3,
                        "target toggle is being examined while the agent stays here",
                    )
                if mentions_target_object and mentions_target_toggle:
                    add_vote(
                        "TOGGLE_SOURCE",
                        target_meta["toggle"]["label"] or raw_location,
                        4,
                        "compound examine action references both the target object and the target toggle",
                    )
                elif mentions_target_object and object_visible_here:
                    add_vote("OBJECT_SOURCE", raw_location, 2, "target object is being examined while the agent stays here")

        if object_visible_here:
            add_vote("OBJECT_SOURCE", raw_location, 3, "target object is visible in observations around this location")
        if toggle_visible_here:
            add_vote(
                "TOGGLE_SOURCE",
                target_meta["toggle"]["label"],
                3,
                "target toggle is visible in observations around this location",
            )
        if target_meta["parent"]["label"] and _text_mentions_base(observation_text, target_meta["parent"]["label"]):
            add_vote(
                "GOAL_RECEP",
                target_meta["parent"]["label"],
                3,
                "goal receptacle is visible in observations around this location",
            )
        if target_meta["mrecep"]["label"] and _text_mentions_base(observation_text, target_meta["mrecep"]["label"]):
            add_vote(
                "MRECEP_CONTAINER",
                target_meta["mrecep"]["label"],
                2,
                "movable receptacle is visible in observations around this location",
            )

        if votes:
            score_by_role: Dict[str, int] = defaultdict(int)
            best_entity_for_role: Dict[str, Tuple[str, int]] = {}
            reasons_by_role: Dict[str, List[str]] = defaultdict(list)
            for vote in votes:
                role = vote["role"]
                score_by_role[role] += int(vote["weight"])
                reasons_by_role[role].append(vote["reason"])
                current = best_entity_for_role.get(role)
                if current is None or int(vote["weight"]) > current[1]:
                    best_entity_for_role[role] = (vote["entity"], int(vote["weight"]))

            best_role = sorted(
                score_by_role.items(),
                key=lambda item: (-item[1], priority_rank.get(item[0], len(priority_rank))),
            )[0][0]
            best_entity = best_entity_for_role[best_role][0]
            overrides[goto_idx] = {
                "role": best_role,
                "entity": best_entity,
                "score": int(score_by_role[best_role]),
                "evidence": list(dict.fromkeys(reasons_by_role[best_role])),
                "raw_location": raw_location,
                "traj_step_idx": segment.get("traj_step_idx"),
                "target_labels": target_labels,
            }
        else:
            overrides[goto_idx] = {
                "role": "LOC",
                "entity": normalize_entity_label(raw_location),
                "score": 0,
                "evidence": ["no strong hindsight evidence; keep the coarse location schema"],
                "raw_location": raw_location,
                "traj_step_idx": segment.get("traj_step_idx"),
                "target_labels": target_labels,
            }

    if overrides:
        role_scores_by_location: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        best_entity_by_location_role: Dict[Tuple[str, str], Tuple[str, int]] = {}
        evidence_by_location_role: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for override in overrides.values():
            raw_location = str(override.get("raw_location") or "")
            role = str(override.get("role") or "LOC")
            score = int(override.get("score", 0))
            entity = normalize_entity_label(override.get("entity") or raw_location)
            role_scores_by_location[raw_location][role] += score
            current = best_entity_by_location_role.get((raw_location, role))
            if current is None or score > current[1]:
                best_entity_by_location_role[(raw_location, role)] = (entity, score)
            evidence_by_location_role[(raw_location, role)].extend(list(override.get("evidence", [])))

        canonical_by_location: Dict[str, Dict[str, Any]] = {}
        for raw_location, role_scores in role_scores_by_location.items():
            best_role = sorted(
                role_scores.items(),
                key=lambda item: (-item[1], priority_rank.get(item[0], len(priority_rank))),
            )[0][0]
            best_entity = best_entity_by_location_role[(raw_location, best_role)][0]
            canonical_by_location[raw_location] = {
                "role": best_role,
                "entity": best_entity,
                "score": int(role_scores[best_role]),
                "evidence": list(dict.fromkeys(evidence_by_location_role[(raw_location, best_role)])),
            }

        for goto_idx, override in overrides.items():
            raw_location = str(override.get("raw_location") or "")
            canonical = canonical_by_location.get(raw_location)
            if canonical is not None:
                override["role"] = canonical["role"]
                override["entity"] = canonical["entity"]
                override["score"] = canonical["score"]
                override["evidence"] = canonical["evidence"]

    return overrides


def short_text(value: Any, limit: int = 240) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def state_text(sig: Sequence[str], limit: int = 180) -> str:
    return short_text(" | ".join(sig), limit=limit)


def truthy_number(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "0", "false", "none", "no"}:
            return False
        if lowered in {"1", "true", "yes"}:
            return True
    with suppress(Exception):
        return float(value or 0.0) > 0.0
    return bool(value)


def display_reasonable(value: Any) -> str:
    if value is None:
        return "neutral"
    return str(int(value))


def signature_has_success(sig: Sequence[str]) -> bool:
    return "success" in set(sig)


def target_aliases(pddl_params: Dict[str, Any]) -> Dict[str, str]:
    return {
        "object": normalize_base_name(str(pddl_params.get("object_target") or "")),
        "parent": normalize_base_name(str(pddl_params.get("parent_target") or "")),
        "toggle": normalize_base_name(str(pddl_params.get("toggle_target") or "")),
        "mrecep": normalize_base_name(str(pddl_params.get("mrecep_target") or "")),
    }


def obs_fact_summary(obs: Any, pddl_params: Dict[str, Any], limit: int = 180) -> str:
    text = normalize_entity_label(strip_markup(obs))
    if not text:
        return ""
    facts: List[str] = []
    arrival = re.search(r"\byou arrive at ([^.]+)", text)
    if arrival:
        facts.append(f"loc={arrival.group(1).strip()}")
    aliases = target_aliases(pddl_params)
    for role, alias in aliases.items():
        if alias and _text_mentions_base(text, alias):
            facts.append(f"sees_{role}={alias}")
    if "you are carrying" in text or "you have" in text:
        facts.append("inventory_hint")
    for obj, state in re.findall(r"\bthe ([^.]+?) is (open|closed)\b", text):
        facts.append(f"{obj.strip()}={state}")
    if "nothing happens" in text:
        facts.append("nothing_happens")
    if not facts:
        return short_text(text, limit=limit)
    return short_text("; ".join(dict.fromkeys(facts)), limit=limit)


def obs_indicates_no_effect(obs: Any) -> bool:
    text = normalize_entity_label(strip_markup(obs))
    return "nothing happens" in text


def normalize_action_for_admissible_match(value: Any) -> str:
    """Lowercase, strip XML-ish tags, collapse whitespace (matches prompt-facing commands)."""
    return normalize_entity_label(strip_markup(value))


def admissible_normalized_set(row: Dict[str, Any]) -> Optional[Set[str]]:
    raw = row.get("admissible_actions_before")
    if raw is None or not isinstance(raw, list):
        return None
    out: Set[str] = set()
    for item in raw:
        key = normalize_action_for_admissible_match(item)
        if key:
            out.add(key)
    return out or None


def executed_action_in_admissible(row: Dict[str, Any]) -> Optional[bool]:
    """None = rollout did not provide a usable admissible list; True/False = membership."""
    adm = admissible_normalized_set(row)
    if adm is None:
        return None
    act = normalize_action_for_admissible_match(row.get("executed_action"))
    if not act:
        return None
    return act in adm


def safe_filename(value: str, limit: int = 160) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value[:limit].strip("_") or "item"


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def alfworld_dataset_root() -> Path:
    root = Path(
        os.environ.get("ALFWORLD_DATA", Path.home() / ".cache/alfworld")
    ).expanduser()
    return root if root.name == "json_2.1.1" else root / "json_2.1.1"


def relative_alfworld_path(value: str) -> Optional[Path]:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return None
    marker = "json_2.1.1/"
    if marker in text:
        return Path(text.split(marker, 1)[1].lstrip("/"))
    path = Path(text)
    if not path.is_absolute() and path.parts and path.parts[0] in {
        "train",
        "valid_seen",
        "valid_unseen",
        "tests_seen",
        "tests_unseen",
    }:
        return path
    return None


def normalize_gamefile(gamefile: str) -> Path:
    path = Path(gamefile).expanduser()
    if path.exists():
        return path
    relative = relative_alfworld_path(gamefile)
    if relative is not None:
        return alfworld_dataset_root() / relative
    return path


def portable_alfworld_path(value: str) -> str:
    relative = relative_alfworld_path(value)
    return relative.as_posix() if relative is not None else str(Path(value))


def normalize_trial_dir(trial_dir: str) -> Path:
    path = Path(trial_dir).expanduser()
    if path.exists():
        return path
    relative = relative_alfworld_path(trial_dir)
    if relative is not None:
        return alfworld_dataset_root() / relative
    return path


def trial_dir_from_gamefile(gamefile: str) -> Path:
    return normalize_gamefile(gamefile).parent


def task_key_from_gamefile(gamefile: str) -> str:
    relative = relative_alfworld_path(gamefile)
    if relative is not None:
        return relative.with_suffix("").as_posix().replace("/", "__")
    path = normalize_gamefile(gamefile)
    return safe_filename(str(path))


def action_key(action: str, roles: Sequence[str]) -> str:
    return f"{action}::{'|'.join(roles)}"


def parse_alfworld_action_text(action: str) -> Tuple[str, List[str]]:
    text = (action or "").strip().lower()
    patterns = [
        ("GotoLocation", r"^go to (?P<arg1>.+)$"),
        ("PickupObject", r"^take (?P<arg1>.+?) from (?P<arg2>.+)$"),
        ("PutObject", r"^put (?P<arg1>.+?) (?:in|on) (?P<arg2>.+)$"),
        ("PutObject", r"^move (?P<arg1>.+?) to (?P<arg2>.+)$"),
        ("OpenObject", r"^open (?P<arg1>.+)$"),
        ("CloseObject", r"^close (?P<arg1>.+)$"),
        ("ToggleObject", r"^use (?P<arg1>.+)$"),
        ("CleanObject", r"^clean (?P<arg1>.+?) with (?P<arg2>.+)$"),
        ("HeatObject", r"^heat (?P<arg1>.+?) with (?P<arg2>.+)$"),
        ("CoolObject", r"^cool (?P<arg1>.+?) with (?P<arg2>.+)$"),
        ("SliceObject", r"^slice (?P<arg1>.+?) with (?P<arg2>.+)$"),
        ("ExamineObject", r"^examine (?P<arg1>.+)$"),
        ("Look", r"^look$"),
        ("Inventory", r"^inventory$"),
    ]
    for action_type, pattern in patterns:
        match = re.match(pattern, text)
        if match:
            groups = match.groupdict()
            if action_type in {"PickupObject", "CleanObject", "HeatObject", "CoolObject", "SliceObject"}:
                return action_type, [groups["arg1"]]
            return action_type, [value for _, value in sorted(groups.items()) if value]
    return "Unknown", [text] if text else []


def rollout_action_for_abstract(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_type = str(row.get("executed_action_type") or "")
    raw_args = _rollout_action_args(row)
    parsed_type, parsed_args = parse_alfworld_action_text(str(row.get("executed_action") or ""))
    use_fallback = raw_type in {"", "Unknown"} and parsed_type != "Unknown"
    use_parsed_args_for_empty_raw = (
        not use_fallback
        and not raw_args
        and parsed_args
        and parsed_type != "Unknown"
        and raw_type == parsed_type
    )
    action_type = parsed_type if use_fallback else raw_type
    action_args = parsed_args if (use_fallback or use_parsed_args_for_empty_raw) else raw_args
    warnings = []
    if use_fallback:
        warnings.append(f"fallback:{raw_type or 'empty'}->{parsed_type}")
    if use_parsed_args_for_empty_raw:
        warnings.append("empty_raw_args->text_args")
    if truthy_number(row.get("won")) and not bool(row.get("is_action_valid", True)):
        warnings.append("won_but_invalid_flag")
    if parsed_type != "Unknown" and raw_type not in {"", "Unknown", parsed_type}:
        warnings.append(f"text_type={parsed_type}")
    if action_type == "GotoLocation" and not action_args:
        # Some rollout rows carry a high-level GotoLocation type but no
        # structured argument. The lower-level ALFWorld abstraction helper
        # expects a concrete destination, so keep training alive by treating the
        # row as an untrusted unknown transition unless the text parser recovered
        # the missing destination above.
        warnings.append("empty_goto_args->unknown")
        action_type = "Unknown"
    return {
        "action_type": action_type,
        "action_args": [normalize_rollout_label(arg) for arg in action_args if normalize_rollout_label(arg)],
        "parsed_action_type": parsed_type,
        "parsed_action_args": [normalize_rollout_label(arg) for arg in parsed_args if normalize_rollout_label(arg)],
        "parser_fallback_used": use_fallback,
        "parser_warning": "; ".join(warnings),
    }


def walkthrough_abstract_states(
    traj: Dict[str, Any],
    walkthrough: Sequence[str],
    signature_mode: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pseudo_high_steps: List[Dict[str, Any]] = []
    alias_map: Dict[int, List[str]] = {}
    for idx, action_text in enumerate(walkthrough):
        action_type, action_args = parse_alfworld_action_text(action_text)
        pseudo_high_steps.append(
            {
                "high_idx": idx,
                "discrete_action": {
                    "action": action_type,
                    "args": [normalize_base_name(arg) for arg in action_args],
                },
            }
        )
        alias_map[idx] = [normalize_entity_label(arg) for arg in action_args]

    pseudo_traj = {
        "task_type": traj["task_type"],
        "pddl_params": traj["pddl_params"],
        "plan": {"high_pddl": pseudo_high_steps},
    }
    include_entities = signature_mode != "semantic"
    states = [initial_abstract_state()]
    edges: List[Dict[str, Any]] = []
    for idx, _ in enumerate(pseudo_high_steps):
        abstract_step_info = abstract_high_step(
            pseudo_high_steps,
            idx,
            pseudo_traj,
            alias_map=alias_map,
        )
        prev_state = states[-1]
        next_state = advance_abstract_state(pseudo_traj, prev_state, abstract_step_info)
        edges.append(
            {
                "step_index": len(edges),
                "high_idx": idx,
                "raw_action": walkthrough[idx],
                "raw_args": list(pseudo_high_steps[idx]["discrete_action"].get("args", [])),
                "abstract_action": abstract_step_info["action"],
                "abstract_roles": list(abstract_step_info["roles"]),
                "abstract_entities": list(abstract_step_info["entities"]),
                "from_signature": state_signature(prev_state, include_entities=include_entities),
                "to_signature": state_signature(next_state, include_entities=include_entities),
            }
        )
        states.append(next_state)
    return states, edges


def replay_walkthrough_with_textworld(gamefile: str, max_episode_steps: int = 80) -> Dict[str, Any]:
    """Replay the dataset walkthrough in a single TextWorld gym env.

    The normal training wrapper scans the whole ALFWorld split before creating an
    env. For this offline pilot we register only the requested gamefile, which is
    much cheaper when building many per-task oracle graphs.
    """
    alfworld_pkg = REPO_ROOT / "agent_system" / "environments" / "env_package" / "alfworld"
    if str(alfworld_pkg) not in sys.path:
        sys.path.insert(0, str(alfworld_pkg))

    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    game_path = normalize_gamefile(gamefile)
    walkthrough = json.loads(game_path.read_text()).get("walkthrough", [])
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    name = "alfworld_reason_" + sha1(str(game_path).encode("utf-8")).hexdigest()[:12]
    env_id = textworld.gym.register_games(
        [str(game_path)],
        request_infos,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        max_episode_steps=max_episode_steps,
        name=name,
    )
    env = textworld.gym.make(env_id)
    try:
        obs, infos = env.reset()
        replay_nodes = [
            {
                "depth": 0,
                "observation": obs,
                "admissible_commands": list(infos.get("admissible_commands", [])),
                "won": bool(infos.get("won", False)),
                "gamefile": infos.get("extra.gamefile"),
            }
        ]
        replay_edges = []
        for idx, action in enumerate(walkthrough):
            before_obs = obs
            before_admissible = list(infos.get("admissible_commands", []))
            obs, score, done, infos = env.step(action)
            edge = {
                "oracle_step_idx": idx,
                "action": action,
                "observation_before": before_obs,
                "observation_after": obs,
                "admissible_before": before_admissible,
                "admissible_after": list(infos.get("admissible_commands", [])),
                "score": score,
                "done": bool(done),
                "won": bool(infos.get("won", False)),
            }
            replay_edges.append(edge)
            replay_nodes.append(
                {
                    "depth": idx + 1,
                    "observation": obs,
                    "admissible_commands": list(infos.get("admissible_commands", [])),
                    "won": bool(infos.get("won", False)),
                    "gamefile": infos.get("extra.gamefile"),
                }
            )
            if done:
                break
        return {
            "ok": True,
            "nodes": replay_nodes,
            "edges": replay_edges,
            "num_walkthrough_actions": len(walkthrough),
            "num_replayed_actions": len(replay_edges),
            "completed": bool(replay_edges and replay_edges[-1].get("done")),
            "won": bool(replay_edges and replay_edges[-1].get("won")),
        }
    finally:
        with suppress(Exception):
            env.close()


def build_oracle_for_gamefile(
    gamefile: str,
    signature_mode: str = "entity",
    state_source: str = "abstract_replay_from_walkthrough",
) -> Dict[str, Any]:
    trial_dir = trial_dir_from_gamefile(gamefile)
    traj = load_traj_data(trial_dir)
    walkthrough = load_walkthrough(trial_dir)
    states, raw_edges = walkthrough_abstract_states(traj, walkthrough, signature_mode=signature_mode)
    include_entities = signature_mode != "semantic"

    node_signatures = [
        tuple(state_signature(state, include_entities=include_entities))
        for state in states
    ]
    textworld_replay: Optional[Dict[str, Any]] = None
    if state_source == "textworld_replay":
        textworld_replay = replay_walkthrough_with_textworld(gamefile)

    nodes = [
        {
            "node_id": f"O{idx}",
            "depth": idx,
            "signature": list(sig),
            "label": state_text(sig, limit=500),
            "observation": (
                textworld_replay["nodes"][idx]["observation"]
                if textworld_replay and idx < len(textworld_replay["nodes"])
                else None
            ),
            "admissible_commands": (
                textworld_replay["nodes"][idx]["admissible_commands"]
                if textworld_replay and idx < len(textworld_replay["nodes"])
                else []
            ),
            "won": (
                textworld_replay["nodes"][idx]["won"]
                if textworld_replay and idx < len(textworld_replay["nodes"])
                else False
            ),
        }
        for idx, sig in enumerate(node_signatures)
    ]

    edges = []
    for idx, edge in enumerate(raw_edges):
        roles = [
            format_binding(role, entity)
            for role, entity in zip(edge["abstract_roles"], edge["abstract_entities"])
        ]
        walkthrough_action = walkthrough[idx] if idx < len(walkthrough) else ""
        edges.append(
            {
                "oracle_step_idx": idx,
                "edge_id": f"expert_e{idx}",
                "path_source": "dataset_walkthrough",
                "src": f"O{idx}",
                "dst": f"O{idx + 1}",
                "from_signature": list(edge["from_signature"]),
                "to_signature": list(edge["to_signature"]),
                "walkthrough_action": walkthrough_action,
                "abstract_action": edge["abstract_action"],
                "abstract_roles": list(edge["abstract_roles"]),
                "abstract_entities": list(edge["abstract_entities"]),
                "role_text": " -> ".join(roles),
                "action_key": action_key(edge["abstract_action"], roles),
                "observation_before": (
                    textworld_replay["edges"][idx]["observation_before"]
                    if textworld_replay and idx < len(textworld_replay["edges"])
                    else None
                ),
                "observation_after": (
                    textworld_replay["edges"][idx]["observation_after"]
                    if textworld_replay and idx < len(textworld_replay["edges"])
                    else None
                ),
                "admissible_before": (
                    textworld_replay["edges"][idx]["admissible_before"]
                    if textworld_replay and idx < len(textworld_replay["edges"])
                    else []
                ),
                "done": (
                    textworld_replay["edges"][idx]["done"]
                    if textworld_replay and idx < len(textworld_replay["edges"])
                    else False
                ),
                "won": (
                    textworld_replay["edges"][idx]["won"]
                    if textworld_replay and idx < len(textworld_replay["edges"])
                    else False
                ),
            }
        )

    return {
        "task_key": task_key_from_gamefile(gamefile),
        "gamefile": portable_alfworld_path(gamefile),
        "trial_dir": portable_alfworld_path(str(trial_dir)),
        "state_source": state_source,
        "signature_mode": signature_mode,
        "task_type": traj.get("task_type"),
        "pddl_params": traj.get("pddl_params", {}),
        "walkthrough": walkthrough,
        "reference_paths": [
            {
                "path_id": "expert",
                "source": "dataset_walkthrough",
                "node_ids": [f"O{idx}" for idx in range(len(nodes))],
                "edge_ids": [f"expert_e{idx}" for idx in range(len(edges))],
                "walkthrough": list(walkthrough),
            }
        ],
        "nodes": nodes,
        "edges": edges,
        "goal_signature": list(node_signatures[-1]) if node_signatures else [],
        "textworld_replay": {
            key: value
            for key, value in (textworld_replay or {}).items()
            if key not in {"nodes", "edges"}
        } if textworld_replay else None,
    }


def build_oracle_index(
    gamefiles: Sequence[str],
    output_dir: Path,
    signature_mode: str,
    state_source: str,
) -> Dict[str, Any]:
    graph_dir = output_dir / "oracle_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    index: Dict[str, Any] = {
        "state_source": state_source,
        "signature_mode": signature_mode,
        "graphs": {},
    }
    unique_gamefiles = sorted(set(gamefiles))
    for ordinal, gamefile in enumerate(unique_gamefiles, start=1):
        if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(unique_gamefiles):
            print(
                f"[oracle] {ordinal}/{len(unique_gamefiles)} state_source={state_source} {short_text(gamefile, 96)}",
                flush=True,
            )
        try:
            oracle = build_oracle_for_gamefile(gamefile, signature_mode=signature_mode, state_source=state_source)
        except Exception as exc:
            index["graphs"][task_key_from_gamefile(gamefile)] = {
                "gamefile": portable_alfworld_path(gamefile),
                "error": repr(exc),
            }
            continue
        out = graph_dir / f"{safe_filename(oracle['task_key'])}.json"
        json_dump(out, oracle)
        index["graphs"][oracle["task_key"]] = {
            "gamefile": oracle["gamefile"],
            "path": str(out.relative_to(output_dir)),
                "num_edges": len(oracle["edges"]),
                "task_type": oracle["task_type"],
                "state_source": oracle["state_source"],
                "textworld_replay": oracle.get("textworld_replay"),
            }
    json_dump(graph_dir / "index.json", index)
    return index


def oracle_lookup(oracle: Dict[str, Any]) -> Dict[str, Any]:
    edge_by_exact: Dict[Tuple[Tuple[str, ...], Tuple[str, ...], str], List[Dict[str, Any]]] = defaultdict(list)
    edge_by_transition: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)
    edge_by_action_text: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    node_depth: Dict[Tuple[str, ...], int] = {}

    for node in oracle["nodes"]:
        node_depth[tuple(node["signature"])] = int(node["depth"])
    for edge in oracle["edges"]:
        src = tuple(edge["from_signature"])
        dst = tuple(edge["to_signature"])
        edge_by_exact[(src, dst, edge["action_key"])].append(edge)
        edge_by_transition[(src, dst)].append(edge)
        if edge.get("walkthrough_action"):
            edge_by_action_text[str(edge["walkthrough_action"]).strip().lower()].append(edge)
    return {
        "edge_by_exact": edge_by_exact,
        "edge_by_transition": edge_by_transition,
        "edge_by_action_text": edge_by_action_text,
        "node_depth": node_depth,
        "oracle_len": max(1, len(oracle["edges"])),
    }


def ensure_reference_paths(oracle: Dict[str, Any]) -> None:
    if oracle.get("reference_paths"):
        return
    nodes = sorted(oracle["nodes"], key=lambda item: int(item["depth"]))
    edges = sorted(oracle["edges"], key=lambda item: int(item["oracle_step_idx"]))
    for idx, edge in enumerate(edges):
        edge.setdefault("edge_id", f"expert_e{idx}")
        edge.setdefault("path_source", "dataset_walkthrough")
    oracle["reference_paths"] = [
        {
            "path_id": "expert",
            "source": "dataset_walkthrough",
            "node_ids": [str(node["node_id"]) for node in nodes],
            "edge_ids": [str(edge["edge_id"]) for edge in edges],
            "walkthrough": [str(edge.get("walkthrough_action") or "") for edge in edges],
        }
    ]


def oracle_view_for_reference_path(oracle: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    nodes_by_id = {str(node["node_id"]): node for node in oracle["nodes"]}
    edges_by_id = {str(edge.get("edge_id", f"expert_e{idx}")): edge for idx, edge in enumerate(oracle["edges"])}
    node_ids = [str(node_id) for node_id in ref.get("node_ids", [])]
    edge_ids = [str(edge_id) for edge_id in ref.get("edge_ids", [])]
    nodes = []
    for depth, node_id in enumerate(node_ids):
        node = deepcopy(nodes_by_id[node_id])
        node["depth"] = depth
        nodes.append(node)
    edges = []
    for idx, edge_id in enumerate(edge_ids):
        edge = deepcopy(edges_by_id[edge_id])
        edge["oracle_step_idx"] = idx
        if idx < len(node_ids):
            edge["src"] = node_ids[idx]
        if idx + 1 < len(node_ids):
            edge["dst"] = node_ids[idx + 1]
        edges.append(edge)
    view = {**oracle}
    view["nodes"] = nodes
    view["edges"] = edges
    view["walkthrough"] = [str(edge.get("walkthrough_action") or "") for edge in edges]
    view["goal_signature"] = list(nodes[-1]["signature"]) if nodes else []
    view["selected_reference_path"] = {
        "path_id": ref.get("path_id"),
        "source": ref.get("source"),
        "num_edges": len(edges),
    }
    return view


def ordered_oracle_node_signatures(oracle: Dict[str, Any]) -> List[Tuple[str, ...]]:
    return [tuple(n["signature"]) for n in sorted(oracle["nodes"], key=lambda item: int(item["depth"]))]


def spit_parts_from_oracle_nodes(matched_nodes: int, oracle_node_count: int) -> Tuple[int, int, float, float]:
    # Initial oracle state is prior context for the first action, not progress
    # created by that action. Progress therefore counts non-initial oracle nodes.
    denom = max(1, int(oracle_node_count) - 1)
    num = max(0, min(denom, int(matched_nodes) - 1))
    spit = max(0.0, min(1.0, float(num) / float(denom)))
    damage = 1.0 - spit
    return num, denom, spit, damage


def format_fraction(num: Any, denom: Any) -> str:
    return f"{int(num)}/{int(denom)}"


def _signature_entity(piece: str) -> str:
    match = re.search(r"\[([^\]]+)\]", piece)
    return match.group(1) if match else ""


def _signature_prefix(piece: str) -> str:
    return piece.split(":", 1)[0]


def signature_piece_compatible(model_piece: str, oracle_piece: str) -> bool:
    if model_piece == oracle_piece:
        return True
    if oracle_piece.startswith("at:") and model_piece.startswith("at:"):
        # Location role can differ for the same physical place (e.g. OBJECT_SOURCE
        # vs GOAL_RECEP), but a different entity fails the prerequisite check.
        oracle_entity = _signature_entity(oracle_piece)
        model_entity = _signature_entity(model_piece)
        return bool(oracle_entity and model_entity and oracle_entity == model_entity)
    if oracle_piece.startswith("placed:") and model_piece.startswith("placed:"):
        # The count before @ is bookkeeping for repeated placements. For graph
        # credit, same target receptacle means the placement fact is satisfied.
        oracle_entity = _signature_entity(oracle_piece)
        model_entity = _signature_entity(model_piece)
        return bool(oracle_entity and model_entity and oracle_entity == model_entity)
    return False


def signature_satisfies_prerequisites(
    model_signature: Sequence[str],
    oracle_from_signature: Sequence[str],
) -> bool:
    model_pieces = list(model_signature)
    for oracle_piece in oracle_from_signature:
        if oracle_piece == "start":
            if "start" not in model_pieces:
                return False
            continue
        oracle_prefix = _signature_prefix(oracle_piece)
        candidates = [piece for piece in model_pieces if _signature_prefix(piece) == oracle_prefix]
        if not any(signature_piece_compatible(piece, oracle_piece) for piece in candidates):
            return False
    return True


def abstract_state_covers_oracle_node(model_sig: Tuple[str, ...], oracle_node_sig: Tuple[str, ...]) -> bool:
    """True if model abstract state satisfies all atoms of an oracle node signature."""
    return signature_satisfies_prerequisites(model_sig, oracle_node_sig)


def best_oracle_node_match(
    model_sig: Tuple[str, ...],
    oracle_node_sigs: Sequence[Tuple[str, ...]],
    previous_matched: int,
) -> int:
    covered = [
        idx + 1
        for idx, oracle_sig in enumerate(oracle_node_sigs)
        if abstract_state_covers_oracle_node(model_sig, oracle_sig)
    ]
    if not covered:
        return 0
    forward = [value for value in covered if value > previous_matched]
    if forward:
        return min(forward)
    if previous_matched in covered:
        return previous_matched
    return max(covered)


def overwrite_spit_fields(
    label: Dict[str, Any],
    matched_before: int,
    matched_after: int,
    oracle_node_count: int,
) -> None:
    prev_spit_num, _prev_spit_den, prev_spit, _prev_damage = spit_parts_from_oracle_nodes(
        matched_before, oracle_node_count
    )
    spit_num, spit_den, spit, damage = spit_parts_from_oracle_nodes(matched_after, oracle_node_count)
    delta_spit_num = spit_num - prev_spit_num
    label.update(
        {
            "oracle_node_matched_before": matched_before,
            "oracle_node_matched_after": matched_after,
            "spit_num": spit_num,
            "spit_den": spit_den,
            "spit": spit,
            "prev_spit": prev_spit,
            "delta_spit_num": delta_spit_num,
            "delta_spit": float(delta_spit_num) / float(spit_den),
            "hit": int(delta_spit_num > 0),
            "turn_success_damage": damage,
        }
    )


def model_transition_rows(
    rollout_rows: Sequence[Dict[str, Any]],
    expert_traj: Dict[str, Any],
    signature_mode: str,
) -> List[Dict[str, Any]]:
    ordered_rows = sorted(
        [dict(row) for row in rollout_rows],
        key=lambda row: int(row.get("traj_step_idx", 0)),
    )
    pseudo_high_steps: List[Dict[str, Any]] = []
    alias_map: Dict[int, List[str]] = {}
    action_infos: Dict[int, Dict[str, Any]] = {}
    for idx, row in enumerate(ordered_rows):
        action_info = rollout_action_for_abstract(row)
        action_type = action_info["action_type"]
        action_args = action_info["action_args"]
        if not truthy_number(row.get("is_action_valid", True)) and not truthy_number(row.get("won")):
            # Invalid ALFWorld actions usually do not mutate the environment. Keep
            # their abstract transition as a self-loop while still showing the
            # parsed text in the HTML diagnostics.
            action_type = "InvalidAction"
            action_args = []
        action_infos[idx] = action_info
        pseudo_high_steps.append(
            {
                "high_idx": int(row.get("traj_step_idx", idx)),
                "discrete_action": {
                    "action": action_type,
                    "args": [normalize_base_name(arg) for arg in action_args],
                },
            }
        )
        alias_map[idx] = action_args

    pseudo_traj = {
        "task_type": expert_traj["task_type"],
        "pddl_params": expert_traj["pddl_params"],
        "plan": {"high_pddl": pseudo_high_steps},
    }
    binding_overrides = infer_rollout_goto_binding_overrides(ordered_rows, expert_traj)
    include_entities = signature_mode != "semantic"
    states = [initial_abstract_state()]
    transitions: List[Dict[str, Any]] = []

    for idx, row in enumerate(ordered_rows):
        abstract_step_info = abstract_high_step(
            pseudo_high_steps,
            idx,
            pseudo_traj,
            alias_map=alias_map,
            binding_overrides=binding_overrides,
        )
        prev_state = states[-1]
        next_state = advance_abstract_state(pseudo_traj, prev_state, abstract_step_info)
        if obs_indicates_no_effect(row.get("obs_after")):
            next_state = deepcopy(prev_state)
        from_sig = tuple(state_signature(prev_state, include_entities=include_entities))
        to_sig = tuple(state_signature(next_state, include_entities=include_entities))
        roles = [
            format_binding(binding.get("role"), binding.get("entity"))
            for binding in abstract_step_info["bindings"]
        ]
        evidence = []
        if idx in binding_overrides:
            evidence = list(binding_overrides[idx].get("evidence", []))
        action_info = action_infos[idx]
        transitions.append(
            {
                "row": row,
                "step_index": idx,
                "traj_step_idx": int(row.get("traj_step_idx", idx)),
                "from_signature": from_sig,
                "to_signature": to_sig,
                "abstract_action": abstract_step_info["action"],
                "role_text": " -> ".join(roles),
                "action_key": action_key(abstract_step_info["action"], roles),
                "binding_evidence": evidence,
                "parsed_action_type": action_info["parsed_action_type"],
                "parsed_action_args": action_info["parsed_action_args"],
                "parser_fallback_used": action_info["parser_fallback_used"],
                "parser_warning": action_info["parser_warning"],
            }
        )
        states.append(next_state)
    return transitions


def classify_transition(
    transition: Dict[str, Any],
    lookup: Dict[str, Any],
    edge_progress_before: int,
    prev_transition: Optional[Dict[str, Any]],
    oracle_node_matched_before: int,
    oracle_node_matched_after: int,
    oracle_node_count: int,
) -> Dict[str, Any]:
    row = transition["row"]
    src = transition["from_signature"]
    dst = transition["to_signature"]
    key = transition["action_key"]
    node_depth = lookup["node_depth"]
    progress_before = edge_progress_before

    admissible_ok = executed_action_in_admissible(row)
    if admissible_ok is False:
        oracle_node_matched_after = oracle_node_matched_before
        spit_num, spit_den, spit, damage = spit_parts_from_oracle_nodes(
            oracle_node_matched_after, oracle_node_count
        )
        prev_spit_num, _prev_spit_den, prev_spit, _prev_damage = spit_parts_from_oracle_nodes(
            oracle_node_matched_before, oracle_node_count
        )
        delta_spit_num = spit_num - prev_spit_num
        return {
            "reasonable": 0,
            "reason": "not_admissible",
            "oracle_step_idx": -1,
            "blocked_oracle_step_idx": -1,
            "required_from_signature": [],
            "oracle_node_matched_before": oracle_node_matched_before,
            "oracle_node_matched_after": oracle_node_matched_after,
            "progress_before": progress_before,
            "progress_after": progress_before,
            "spit_num": spit_num,
            "spit_den": spit_den,
            "spit": spit,
            "prev_spit": prev_spit,
            "delta_spit_num": delta_spit_num,
            "delta_spit": float(delta_spit_num) / float(spit_den),
            "hit": int(delta_spit_num > 0),
            "turn_success_damage": damage,
        }

    is_valid = truthy_number(row.get("is_action_valid", True))
    action = str(row.get("executed_action") or "")
    raw_action_type = str(row.get("executed_action_type") or "")
    action_type = transition["abstract_action"] if raw_action_type in {"", "Unknown"} else raw_action_type
    neutral_types = {"Look", "Inventory", "ExamineObject"}

    reason = "off_oracle_state"
    reasonable: Optional[int] = 0
    matched_oracle_step = -1
    blocked_oracle_step = -1
    required_from_signature: List[str] = []
    progress_after = progress_before
    reached_depth = max(0, oracle_node_matched_after - 1)
    reaches_new_oracle_node = reached_depth > progress_before

    won_but_invalid = bool(truthy_number(row.get("won", False))) or (
        "won_but_invalid_flag" in str(transition.get("parser_warning") or "")
    )

    if not is_valid and not won_but_invalid:
        reason = "invalid_action"
        reasonable = 0
    else:
        exact_edges = lookup["edge_by_exact"].get((src, dst, key), [])
        transition_edges = lookup["edge_by_transition"].get((src, dst), [])
        if exact_edges:
            edge = max(exact_edges, key=lambda item: int(item["oracle_step_idx"]))
            matched_oracle_step = int(edge["oracle_step_idx"])
            progress_after = max(progress_after, matched_oracle_step + 1)
            reason = "oracle_edge_exact"
            reasonable = 1
        elif action.strip().lower() in lookup["edge_by_action_text"]:
            all_candidates = [
                edge
                for edge in lookup["edge_by_action_text"][action.strip().lower()]
                if int(edge["oracle_step_idx"]) >= progress_before
            ]
            if not all_candidates:
                all_candidates = lookup["edge_by_action_text"][action.strip().lower()]
            candidates = [
                edge
                for edge in all_candidates
                if signature_satisfies_prerequisites(src, edge["from_signature"])
            ]
            if not candidates:
                if reaches_new_oracle_node:
                    progress_after = max(progress_after, reached_depth)
                    matched_oracle_step = max(0, reached_depth - 1)
                    reason = "oracle_node_reached"
                    reasonable = 1
                else:
                    edge = min(all_candidates, key=lambda item: abs(int(item["oracle_step_idx"]) - progress_before))
                    blocked_oracle_step = int(edge["oracle_step_idx"])
                    required_from_signature = list(edge["from_signature"])
                    reason = "oracle_action_wrong_state"
                    reasonable = 0
            else:
                edge = min(candidates, key=lambda item: abs(int(item["oracle_step_idx"]) - progress_before))
                matched_oracle_step = int(edge["oracle_step_idx"])
                progress_after = max(progress_after, matched_oracle_step + 1)
                reason = "oracle_action_exact"
                reasonable = 1
        elif transition_edges:
            edge = max(transition_edges, key=lambda item: int(item["oracle_step_idx"]))
            matched_oracle_step = int(edge["oracle_step_idx"])
            progress_after = max(progress_after, matched_oracle_step + 1)
            reason = "oracle_transition"
            reasonable = 1
        elif reaches_new_oracle_node:
            progress_after = max(progress_after, reached_depth)
            matched_oracle_step = max(0, reached_depth - 1)
            reason = "oracle_node_reached"
            reasonable = 1
        elif src == dst:
            repeated = (
                prev_transition is not None
                and str(prev_transition["row"].get("executed_action") or "") == action
            )
            if action_type in neutral_types and not repeated:
                reason = "neutral_probe_no_abstract_progress"
                reasonable = 0
            else:
                reason = "self_loop_no_progress"
                reasonable = 0
        elif action_type in neutral_types:
            reason = "neutral_probe_off_oracle"
            reasonable = 0
        else:
            reason = "off_oracle_state"
            reasonable = 0

    spit_num, spit_den, spit, damage = spit_parts_from_oracle_nodes(
        oracle_node_matched_after, oracle_node_count
    )
    prev_spit_num, _prev_spit_den, prev_spit, _prev_damage = spit_parts_from_oracle_nodes(
        oracle_node_matched_before, oracle_node_count
    )
    delta_spit_num = spit_num - prev_spit_num
    return {
        "reasonable": reasonable,
        "reason": reason,
        "oracle_step_idx": matched_oracle_step,
        "blocked_oracle_step_idx": blocked_oracle_step,
        "required_from_signature": required_from_signature,
        "oracle_node_matched_before": oracle_node_matched_before,
        "oracle_node_matched_after": oracle_node_matched_after,
        "progress_before": progress_before,
        "progress_after": progress_after,
        "spit_num": spit_num,
        "spit_den": spit_den,
        "spit": spit,
        "prev_spit": prev_spit,
        "delta_spit_num": delta_spit_num,
        "delta_spit": float(delta_spit_num) / float(spit_den),
        "hit": int(delta_spit_num > 0),
        "turn_success_damage": damage,
    }


def spit_step_regrets(annotations: Sequence[Dict[str, Any]]) -> List[float]:
    """
    SPIT-based step regret (see learning/bfcl_hierarchical_step_credit_reassignment_proposal3.md §5.3.3):
    from step i through episode end, average (1 - SPIT_v) over v >= i inclusive.
    SPIT_v is the rollout SPIT after executing step v.
    """
    n = len(annotations)
    out: List[float] = []
    for i in range(n):
        tail = n - i
        total = sum(1.0 - float(annotations[v]["spit"]) for v in range(i, n))
        out.append(total / tail if tail else 0.0)
    return out


def attach_step_regret_fields(annotations: List[Dict[str, Any]]) -> None:
    regrets = spit_step_regrets(annotations)
    dis_indices = [idx for idx, ann in enumerate(annotations) if ann["reasonable"] == 0]
    mean_r = sum(regrets[idx] for idx in dis_indices) / len(dis_indices) if dis_indices else 0.0
    dis_set = set(dis_indices)
    hits = [int(ann.get("hit", 0) > 0) for ann in annotations]
    misses = [1 - hit for hit in hits]
    turn_future_miss = sum(misses)
    for idx, ann in enumerate(annotations):
        future_miss = sum(misses[idx:])
        future_tail = len(misses[idx:])
        ann["step_future_miss"] = future_miss
        ann["turn_future_miss"] = turn_future_miss
        ann["avg_future_miss_num"] = future_miss
        ann["avg_future_miss_den"] = future_tail
        ann["avg_future_miss"] = float(future_miss) / float(future_tail) if future_tail else 0.0
        ann["step_regret"] = regrets[idx]
        if idx not in dis_set:
            ann["normalized_step_regret"] = 0.0
        elif mean_r <= 0:
            ann["normalized_step_regret"] = 1.0 / len(dis_indices) if dis_indices else 0.0
        else:
            ann["normalized_step_regret"] = float(regrets[idx]) / mean_r


def annotate_transitions_against_oracle(
    transitions: Sequence[Dict[str, Any]],
    oracle: Dict[str, Any],
    expert_traj: Dict[str, Any],
) -> Dict[str, Any]:
    lookup = oracle_lookup(oracle)
    oracle_node_sigs = ordered_oracle_node_signatures(oracle)
    oracle_node_count = len(oracle_node_sigs)
    matched_nodes = 1 if transitions and oracle_node_sigs and abstract_state_covers_oracle_node(
        transitions[0]["from_signature"], oracle_node_sigs[0]
    ) else 0
    if transitions and oracle_node_sigs:
        current_match = best_oracle_node_match(transitions[0]["from_signature"], oracle_node_sigs, 0)
        if current_match:
            matched_nodes = current_match

    annotations = []
    prev = None
    for transition in transitions:
        node_before = matched_nodes
        to_sig = transition["to_signature"]
        node_after = best_oracle_node_match(to_sig, oracle_node_sigs, node_before)
        if not node_after:
            node_after = 0
        progress_before = max(0, node_before - 1)
        label = classify_transition(
            transition,
            lookup,
            progress_before,
            prev,
            oracle_node_matched_before=node_before,
            oracle_node_matched_after=node_after,
            oracle_node_count=oracle_node_count,
        )
        if label["reasonable"] == 0:
            # Unreasonable steps should not consume forward oracle progress, but
            # a harmful move back to an earlier/off-path state should lower SPIT.
            overwrite_spit_fields(label, node_before, min(node_before, node_after), oracle_node_count)
        else:
            overwrite_spit_fields(label, node_before, node_after, oracle_node_count)
        matched_nodes = int(label["oracle_node_matched_after"])
        prev = transition
        row = transition["row"]
        annotations.append(
            {
                "traj_uid": row.get("traj_uid"),
                "traj_step_idx": transition["traj_step_idx"],
                "executed_action": row.get("executed_action"),
                "executed_action_type": row.get("executed_action_type"),
                "is_action_valid": row.get("is_action_valid"),
                "done": row.get("done"),
                "won": row.get("won"),
                "step_reward": row.get("step_reward"),
                "obs_before": row.get("obs_before"),
                "obs_after": row.get("obs_after"),
                "from_signature": list(transition["from_signature"]),
                "to_signature": list(transition["to_signature"]),
                "abstract_action": transition["abstract_action"],
                "role_text": transition["role_text"],
                "action_key": transition["action_key"],
                "binding_evidence": transition["binding_evidence"],
                "parsed_action_type": transition["parsed_action_type"],
                "parsed_action_args": transition["parsed_action_args"],
                "parser_fallback_used": transition["parser_fallback_used"],
                "parser_warning": transition["parser_warning"],
                "obs_facts": obs_fact_summary(row.get("obs_after"), expert_traj.get("pddl_params", {})),
                "abstract_success": signature_has_success(transition["to_signature"]),
                "action_in_admissible": executed_action_in_admissible(row),
                **label,
            }
        )

    attach_step_regret_fields(annotations)

    reason_counts = Counter(str(ann["reason"]) for ann in annotations)
    reasonable_counts = Counter(display_reasonable(ann["reasonable"]) for ann in annotations)
    return {
        "annotations": annotations,
        "reason_counts": dict(reason_counts),
        "reasonable_counts": dict(reasonable_counts),
        "final_spit": annotations[-1]["spit"] if annotations else 0.0,
        "final_spit_fraction": (
            format_fraction(annotations[-1]["spit_num"], annotations[-1]["spit_den"])
            if annotations
            else "0/1"
        ),
        "selected_reference_path": oracle.get("selected_reference_path"),
    }


def annotation_match_score(annotated: Dict[str, Any]) -> Tuple[float, int, int, int, int, int]:
    anns = annotated.get("annotations") or []
    final_spit = float(annotated.get("final_spit", 0.0))
    wrong_state = sum(int(ann.get("reason") == "oracle_action_wrong_state") for ann in anns)
    denom = int(anns[-1].get("spit_den", 1)) if anns else 1
    exact = sum(int(ann.get("reason") in {"oracle_edge_exact", "oracle_action_exact"}) for ann in anns)
    hits = sum(int(ann.get("hit", 0) > 0) for ann in anns)
    reasonable = sum(int(ann.get("reasonable") == 1) for ann in anns)
    return (round(final_spit, 3), -wrong_state, -denom, exact, hits, reasonable)


def annotate_trajectory(
    rollout_rows: Sequence[Dict[str, Any]],
    oracle: Dict[str, Any],
    signature_mode: str,
) -> Dict[str, Any]:
    ensure_reference_paths(oracle)
    trial_dir = normalize_trial_dir(str(oracle["trial_dir"]))
    expert_traj = load_traj_data(trial_dir)
    transitions = mark_official_success_transition(
        model_transition_rows(rollout_rows, expert_traj, signature_mode=signature_mode)
    )
    candidates = []
    for ref in oracle.get("reference_paths", []):
        view = oracle_view_for_reference_path(oracle, ref)
        candidates.append(annotate_transitions_against_oracle(transitions, view, expert_traj))
    if not candidates:
        candidates.append(annotate_transitions_against_oracle(transitions, oracle, expert_traj))
    return max(candidates, key=annotation_match_score)


def erase_repeated_state_cycles(transitions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove transition segments that leave an abstract state and later return to it."""
    if not transitions:
        return []

    kept: List[Dict[str, Any]] = []
    states: List[Tuple[str, ...]] = [tuple(transitions[0]["from_signature"])]
    state_positions: Dict[Tuple[str, ...], int] = {states[0]: 0}

    for transition in transitions:
        to_signature = tuple(transition["to_signature"])
        previous_position = state_positions.get(to_signature)
        if previous_position is not None:
            kept = kept[:previous_position]
            states = states[: previous_position + 1]
            state_positions = {signature: idx for idx, signature in enumerate(states)}
            continue

        kept.append(transition)
        states.append(to_signature)
        state_positions[to_signature] = len(states) - 1

    return kept


def clean_success_transitions(
    transitions: Sequence[Dict[str, Any]],
    remove_state_cycles: bool = False,
) -> List[Dict[str, Any]]:
    neutral_types = {"Look", "Inventory", "ExamineObject"}
    kept: List[Dict[str, Any]] = []
    for transition in transitions:
        row = transition["row"]
        action_type = str(transition.get("abstract_action") or "")
        won = truthy_number(row.get("won"))
        if executed_action_in_admissible(row) is False:
            continue
        if not truthy_number(row.get("is_action_valid", True)) and not won:
            continue
        if action_type in neutral_types:
            continue
        if action_type in {"", "Unknown", "InvalidAction"} and not won:
            continue
        if tuple(transition["from_signature"]) == tuple(transition["to_signature"]) and not won:
            continue
        if obs_indicates_no_effect(row.get("obs_after")) and not won:
            continue
        kept.append(transition)
        if won or truthy_number(row.get("done")):
            break
    if remove_state_cycles:
        return erase_repeated_state_cycles(kept)
    return kept


def mark_official_success_transition(transitions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add the terminal success marker when ALFWorld reports an official environment win."""
    marked = list(transitions)
    for idx, transition in enumerate(marked):
        if not truthy_number(transition["row"].get("won")):
            continue
        to_signature = tuple(transition["to_signature"])
        if not signature_has_success(to_signature):
            updated = dict(transition)
            updated["to_signature"] = tuple([*to_signature, "success"])
            marked[idx] = updated
        break
    return marked


def find_or_add_oracle_node(
    oracle: Dict[str, Any],
    sig: Sequence[str],
    observation: Any = None,
    admissible_commands: Optional[Sequence[str]] = None,
    won: bool = False,
) -> str:
    sig_tuple = tuple(sig)
    for node in oracle["nodes"]:
        if tuple(node["signature"]) == sig_tuple:
            if won:
                node["won"] = True
            return str(node["node_id"])
    node_id = f"O{len(oracle['nodes'])}"
    oracle["nodes"].append(
        {
            "node_id": node_id,
            "depth": len(oracle["nodes"]),
            "signature": list(sig_tuple),
            "label": state_text(sig_tuple, limit=500),
            "observation": observation,
            "admissible_commands": list(admissible_commands or []),
            "won": bool(won),
        }
    )
    return node_id


def merge_success_trajectory_into_oracle(
    oracle: Dict[str, Any],
    rollout_rows: Sequence[Dict[str, Any]],
    signature_mode: str,
    traj_uid: str,
) -> bool:
    if not trajectory_success(rollout_rows):
        return False
    ensure_reference_paths(oracle)
    expert_traj = load_traj_data(normalize_trial_dir(str(oracle["trial_dir"])))
    transitions = mark_official_success_transition(
        model_transition_rows(rollout_rows, expert_traj, signature_mode=signature_mode)
    )
    kept = clean_success_transitions(transitions)
    if not kept or not signature_has_success(kept[-1]["to_signature"]):
        return False
    expert_len = len((oracle.get("reference_paths") or [{"edge_ids": oracle.get("edges", [])}])[0].get("edge_ids", []))
    max_clean_len = max(4, int(math.ceil(float(expert_len) * 1.5 + 3.0)))
    if len(kept) > max_clean_len:
        return False

    path_id = f"online_{safe_filename(traj_uid[:8])}"
    if any(ref.get("path_id") == path_id for ref in oracle.get("reference_paths", [])):
        return False

    node_ids: List[str] = []
    edge_ids: List[str] = []
    current_sig = tuple(kept[0]["from_signature"])
    first_row = kept[0]["row"]
    node_ids.append(
        find_or_add_oracle_node(
            oracle,
            current_sig,
            observation=first_row.get("obs_before"),
            admissible_commands=first_row.get("admissible_actions_before") or [],
            won=signature_has_success(current_sig),
        )
    )

    existing_edges = {
        (
            tuple(edge["from_signature"]),
            tuple(edge["to_signature"]),
            str(edge.get("walkthrough_action") or ""),
            str(edge.get("action_key") or ""),
        ): str(edge.get("edge_id", f"expert_e{idx}"))
        for idx, edge in enumerate(oracle["edges"])
    }
    for idx, transition in enumerate(kept):
        row = transition["row"]
        from_sig = current_sig
        to_sig = tuple(transition["to_signature"])
        # The cleaned path is a compressed causal path. Skipped probes/detours do
        # not introduce extra reference nodes; each kept action starts at the
        # previous kept action's resulting abstract state.

        src_id = find_or_add_oracle_node(
            oracle,
            from_sig,
            observation=row.get("obs_before"),
            admissible_commands=row.get("admissible_actions_before") or [],
            won=signature_has_success(from_sig),
        )
        dst_id = find_or_add_oracle_node(
            oracle,
            to_sig,
            observation=row.get("obs_after"),
            admissible_commands=row.get("admissible_actions_after") or [],
            won=truthy_number(row.get("won")) or signature_has_success(to_sig),
        )
        edge_key = (from_sig, to_sig, str(row.get("executed_action") or ""), str(transition.get("action_key") or ""))
        edge_id = existing_edges.get(edge_key)
        if edge_id is None:
            edge_id = f"{path_id}_e{idx}"
            roles = str(transition.get("role_text") or "")
            oracle["edges"].append(
                {
                    "oracle_step_idx": len(oracle["edges"]),
                    "edge_id": edge_id,
                    "path_source": "online_success",
                    "src": src_id,
                    "dst": dst_id,
                    "from_signature": list(from_sig),
                    "to_signature": list(to_sig),
                    "walkthrough_action": str(row.get("executed_action") or ""),
                    "abstract_action": transition.get("abstract_action"),
                    "abstract_roles": [],
                    "abstract_entities": [],
                    "role_text": roles,
                    "action_key": transition.get("action_key"),
                    "observation_before": row.get("obs_before"),
                    "observation_after": row.get("obs_after"),
                    "admissible_before": row.get("admissible_actions_before") or [],
                    "done": truthy_number(row.get("done")),
                    "won": truthy_number(row.get("won")),
                }
            )
            existing_edges[edge_key] = edge_id
        edge_ids.append(edge_id)
        if node_ids[-1] != dst_id:
            node_ids.append(dst_id)
        current_sig = to_sig

    nodes_by_id = {str(node["node_id"]): node for node in oracle["nodes"]}
    if len(edge_ids) < 1 or not signature_has_success(nodes_by_id[node_ids[-1]]["signature"]):
        return False
    oracle["reference_paths"].append(
        {
            "path_id": path_id,
            "source": "online_success",
            "traj_uid": traj_uid,
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "walkthrough": [str(t["row"].get("executed_action") or "") for t in kept],
        }
    )
    oracle["online_merged_paths"] = int(oracle.get("online_merged_paths", 0)) + 1
    return True


def oracle_from_success_trajectory(
    rollout_rows: Sequence[Dict[str, Any]],
    signature_mode: str,
    traj_uid: str,
) -> Optional[Dict[str, Any]]:
    """Create an ALFWorld oracle whose single reference path is a successful rollout.

    The task metadata is read only to parse abstract states; the dataset walkthrough
    is not used. The rollout is kept verbatim up to its first won/done step.
    """
    if not rollout_rows or not trajectory_success(rollout_rows):
        return None
    ordered_rows = sorted([dict(row) for row in rollout_rows], key=lambda row: int(row.get("traj_step_idx", 0) or 0))
    gamefile = str(ordered_rows[0].get("gamefile") or "")
    if not gamefile:
        return None
    trial_dir = trial_dir_from_gamefile(gamefile)
    try:
        task_traj = load_traj_data(trial_dir)
    except Exception:
        return None
    transitions = mark_official_success_transition(
        model_transition_rows(ordered_rows, task_traj, signature_mode=signature_mode)
    )
    kept = []
    for transition in transitions:
        kept.append(transition)
        row = transition["row"]
        if truthy_number(row.get("won")) or truthy_number(row.get("done")):
            break
    if not kept or not signature_has_success(kept[-1]["to_signature"]):
        return None

    oracle: Dict[str, Any] = {
        "task_key": task_key_from_gamefile(gamefile),
        "gamefile": portable_alfworld_path(gamefile),
        "trial_dir": portable_alfworld_path(str(trial_dir)),
        "oracle_type": "alfworld_success_trajectory",
        "source": "success_trajectory",
        "state_source": "success_trajectory",
        "signature_mode": signature_mode,
        "task_type": task_traj.get("task_type"),
        "pddl_params": task_traj.get("pddl_params", {}),
        "walkthrough": [str(t["row"].get("executed_action") or "") for t in kept],
        "reference_paths": [],
        "nodes": [],
        "edges": [],
        "goal_signature": [],
    }

    node_ids: List[str] = []
    edge_ids: List[str] = []
    first_row = kept[0]["row"]
    current_sig = tuple(kept[0]["from_signature"])
    node_ids.append(
        find_or_add_oracle_node(
            oracle,
            current_sig,
            observation=first_row.get("obs_before"),
            admissible_commands=first_row.get("admissible_actions_before") or [],
            won=signature_has_success(current_sig),
        )
    )
    path_id = f"success_{safe_filename(traj_uid[:8] or 'rollout')}"
    for idx, transition in enumerate(kept):
        row = transition["row"]
        from_sig = current_sig
        to_sig = tuple(transition["to_signature"])
        src_id = find_or_add_oracle_node(
            oracle,
            from_sig,
            observation=row.get("obs_before"),
            admissible_commands=row.get("admissible_actions_before") or [],
            won=signature_has_success(from_sig),
        )
        dst_id = find_or_add_oracle_node(
            oracle,
            to_sig,
            observation=row.get("obs_after"),
            admissible_commands=row.get("admissible_actions_after") or [],
            won=truthy_number(row.get("won")) or signature_has_success(to_sig),
        )
        edge_id = f"{path_id}_e{idx}"
        oracle["edges"].append(
            {
                "oracle_step_idx": idx,
                "edge_id": edge_id,
                "path_source": "success_trajectory",
                "src": src_id,
                "dst": dst_id,
                "from_signature": list(from_sig),
                "to_signature": list(to_sig),
                "walkthrough_action": str(row.get("executed_action") or ""),
                "abstract_action": transition.get("abstract_action"),
                "abstract_roles": [],
                "abstract_entities": [],
                "role_text": str(transition.get("role_text") or ""),
                "action_key": transition.get("action_key"),
                "observation_before": row.get("obs_before"),
                "observation_after": row.get("obs_after"),
                "admissible_before": row.get("admissible_actions_before") or [],
                "done": truthy_number(row.get("done")),
                "won": truthy_number(row.get("won")),
            }
        )
        edge_ids.append(edge_id)
        if node_ids[-1] != dst_id:
            node_ids.append(dst_id)
        current_sig = to_sig

    oracle["goal_signature"] = list(kept[-1]["to_signature"])
    oracle["reference_paths"].append(
        {
            "path_id": path_id,
            "source": "success_trajectory",
            "traj_uid": traj_uid,
            "node_ids": node_ids,
            "edge_ids": edge_ids,
            "walkthrough": oracle["walkthrough"],
        }
    )
    return oracle


def trajectory_success(rows: Sequence[Dict[str, Any]]) -> bool:
    return any(float(row.get("episode_rewards", 0.0) or 0.0) > 0 for row in rows) or any(
        float(row.get("won", 0.0) or 0.0) > 0 for row in rows
    )
