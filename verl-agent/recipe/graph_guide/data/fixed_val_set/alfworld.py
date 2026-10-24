"""Build deterministic, train-disjoint ALFWorld validation task manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _load_trajectory_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing ALFWorld trajectory metadata: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid ALFWorld trajectory metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"ALFWorld trajectory metadata must be an object: {path}")
    return payload


def _first_task_description(metadata: dict[str, Any], path: Path) -> str:
    annotations = metadata.get("turk_annotations", {}).get("anns", [])
    if isinstance(annotations, list):
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            task_desc = annotation.get("task_desc")
            if isinstance(task_desc, str) and task_desc.strip():
                return task_desc.strip()
    raise ValueError(f"No ALFWorld task description found in {path}")


def enumerate_alfworld_games(data_root: Path, split: str) -> list[dict[str, Any]]:
    data_root = data_root.resolve()
    split_root = data_root / split
    if not split_root.is_dir():
        raise ValueError(f"ALFWorld split does not exist: {split_root}")

    games: list[dict[str, Any]] = []
    for game_path in sorted(split_root.glob("*/*/game.tw-pddl")):
        metadata_path = game_path.with_name("traj_data.json")
        metadata = _load_trajectory_metadata(metadata_path)
        task_instance = str(metadata.get("task_id") or game_path.parent.name).strip()
        if not task_instance:
            raise ValueError(f"Missing ALFWorld task identity: {metadata_path}")
        task_dir = game_path.parent.parent.name
        task_type = str(metadata.get("task_type") or task_dir.split("-", 1)[0])
        games.append(
            {
                "environment": "alfworld",
                "gamefile_relative": game_path.relative_to(data_root).as_posix(),
                "task_desc": _first_task_description(metadata, metadata_path),
                "task_family": task_type,
                "task_instance": task_instance,
                "split": split,
            }
        )
    return games


def _selection_key(case: dict[str, Any]) -> str:
    relative_path = str(case["gamefile_relative"])
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()


def select_alfworld_cases(
    data_root: Path,
    count: int = 128,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count <= 0:
        raise ValueError("ALFWorld validation count must be positive")

    train_games = enumerate_alfworld_games(data_root, "train")
    valid_games = enumerate_alfworld_games(data_root, "valid_seen")
    train_instances = {case["task_instance"] for case in train_games}
    eligible = [
        case for case in valid_games if case["task_instance"] not in train_instances
    ]

    unique_eligible = {
        (case["gamefile_relative"], case["task_instance"]): case
        for case in eligible
    }
    if len(unique_eligible) < count:
        raise ValueError(
            f"ALFWorld validation requires {count} unique eligible games, "
            f"found {len(unique_eligible)}"
        )

    selected = sorted(unique_eligible.values(), key=_selection_key)[:count]
    for index, case in enumerate(selected):
        case["case_id"] = f"alfworld:{index:03d}:{case['task_instance']}"

    selected_gamefiles = {case["gamefile_relative"] for case in selected}
    selected_instances = {case["task_instance"] for case in selected}
    train_overlap = selected_instances & train_instances
    if len(selected_gamefiles) != count or len(selected_instances) != count:
        raise ValueError("ALFWorld validation selection contains duplicate tasks")
    if train_overlap:
        raise ValueError(
            "ALFWorld validation selection overlaps training instances: "
            f"{sorted(train_overlap)[:5]}"
        )

    audit = {
        "selection_rule": "sha256(gamefile_relative)",
        "train_candidate_count": len(train_games),
        "valid_seen_candidate_count": len(valid_games),
        "eligible_candidate_count": len(unique_eligible),
        "selected_count": len(selected),
        "selected_unique_gamefiles": len(selected_gamefiles),
        "selected_unique_task_instances": len(selected_instances),
        "train_overlap_count": len(train_overlap),
    }
    return selected, audit
