from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


def default_fixed_train_parent() -> Path:
    return Path(__file__).resolve().parents[2]


def _fixed_train_config(config: Any) -> Mapping[str, Any]:
    env_config = config.get("env", {})
    fixed_config = env_config.get("fixed_train_set", {})
    return fixed_config or {}


def _optional_int(value: Any, *, name: str, minimum: int) -> int | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"env.fixed_train_set.{name} must be an integer or null") from exc
    if parsed < minimum:
        raise ValueError(
            f"env.fixed_train_set.{name} must be >= {minimum}, got {parsed}"
        )
    return parsed


def _resolve_start_index(
    fixed_config: Mapping[str, Any],
    batches: list[dict[str, Any]],
    *,
    root: Path,
) -> int:
    start_step = _optional_int(
        fixed_config.get("start_step"),
        name="start_step",
        minimum=1,
    )
    start_index = _optional_int(
        fixed_config.get("start_index"),
        name="start_index",
        minimum=0,
    )
    if start_step is None and start_index is None:
        return 0

    index_from_step = None
    if start_step is not None:
        for index, batch in enumerate(batches):
            if batch.get("step") == start_step:
                index_from_step = index
                break
        if index_from_step is None:
            raise ValueError(
                f"env.fixed_train_set.start_step={start_step} was not found in "
                f"the fixed training schedule: {root}"
            )

    if start_index is not None and start_index > len(batches):
        raise ValueError(
            f"env.fixed_train_set.start_index={start_index} exceeds schedule "
            f"horizon={len(batches)}: {root}"
        )

    if start_index is not None and index_from_step is not None:
        if start_index != index_from_step:
            raise ValueError(
                "env.fixed_train_set.start_step and start_index point to "
                f"different schedule positions: start_step={start_step} maps to "
                f"start_index={index_from_step}, got start_index={start_index}"
            )
        return start_index

    return index_from_step if index_from_step is not None else start_index


def fixed_train_set_enabled(config: Any) -> bool:
    path = _fixed_train_config(config).get("path")
    return path not in (None, "")


def resolve_fixed_train_path(config: Any) -> Path | None:
    path_value = _fixed_train_config(config).get("path")
    if path_value in (None, ""):
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    if not path.is_absolute():
        path = default_fixed_train_parent() / path
    return path.resolve()


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Fixed training asset not found: {path}")


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    _require_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed fixed training manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Fixed training manifest must be an object: {path}")
    return payload


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield line_number, row


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    environment: str,
    train_batch_size: int,
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Fixed training manifest has an unexpected schema version: {root}")
    if manifest.get("environment") != environment:
        raise ValueError(
            f"Fixed training manifest environment={manifest.get('environment')!r}, "
            f"expected {environment!r}: {root}"
        )
    if int(manifest.get("train_batch_size", -1)) != int(train_batch_size):
        raise ValueError(
            f"Fixed training manifest train_batch_size={manifest.get('train_batch_size')}, "
            f"but runtime train_batch_size={train_batch_size}: {root}"
        )
    if int(manifest.get("horizon", -1)) <= 0:
        raise ValueError(f"Fixed training manifest horizon must be positive: {root}")


def _validate_batch(
    batch: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
    train_batch_size: int,
) -> None:
    step = batch.get("step")
    tasks = batch.get("tasks")
    if not isinstance(step, int) or step <= 0:
        raise ValueError(f"Fixed training batch has invalid step at {path}:{line_number}")
    if not isinstance(tasks, list):
        raise ValueError(f"Fixed training batch is missing tasks at {path}:{line_number}")
    if len(tasks) != train_batch_size:
        raise ValueError(
            f"Fixed training batch at {path}:{line_number} has {len(tasks)} tasks, "
            f"but train_batch_size={train_batch_size}"
        )
    expected_slots = list(range(train_batch_size))
    slots = [task.get("slot") if isinstance(task, dict) else None for task in tasks]
    if slots != expected_slots:
        raise ValueError(
            f"Fixed training batch at {path}:{line_number} must have slots "
            f"{expected_slots}, got {slots}"
        )


def load_fixed_train_batches(
    config: Any,
    *,
    environment: str,
    train_batch_size: int,
) -> list[dict[str, Any]] | None:
    fixed_config = _fixed_train_config(config)
    root = resolve_fixed_train_path(config)
    if root is None:
        return None
    if not root.is_dir():
        raise FileNotFoundError(f"Fixed training directory not found: {root}")

    manifest = _load_manifest(root)
    _validate_manifest(
        manifest,
        root=root,
        environment=environment,
        train_batch_size=train_batch_size,
    )

    schedule_path = root / "schedule.jsonl"
    _require_file(schedule_path)
    batches = [row for _, row in _iter_jsonl(schedule_path)]
    if len(batches) != int(manifest["horizon"]):
        raise ValueError(
            f"Fixed training schedule has {len(batches)} batches, "
            f"but manifest horizon={manifest['horizon']}: {schedule_path}"
        )
    for line_number, batch in enumerate(batches, start=1):
        _validate_batch(
            batch,
            path=schedule_path,
            line_number=line_number,
            train_batch_size=train_batch_size,
        )
    start_index = _resolve_start_index(fixed_config, batches, root=root)
    return batches[start_index:]


def build_fixed_train_case_assignments(
    batch: Mapping[str, Any],
    *,
    env_num: int,
    group_n: int,
) -> list[dict[str, Any]]:
    if env_num <= 0:
        raise ValueError("env_num must be positive")
    if group_n <= 0:
        raise ValueError("group_n must be positive")
    tasks = batch.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != env_num:
        raise ValueError(
            f"Fixed training batch must contain exactly env_num={env_num} tasks"
        )
    assignments = []
    for slot in range(env_num):
        fixed_case = dict(tasks[slot])
        fixed_case.setdefault("step", batch["step"])
        assignments.extend(dict(fixed_case) for _ in range(group_n))
    return assignments


def _resolve_alfworld_data_root(alfworld_data_root: str | Path | None) -> Path:
    if alfworld_data_root is None:
        env_value = os.environ.get("ALFWORLD_DATA")
        if not env_value:
            raise EnvironmentError(
                "ALFWORLD_DATA must be set when fixed ALFWorld training is enabled."
            )
        data_root = Path(env_value) / "json_2.1.1"
    else:
        data_root = Path(alfworld_data_root)
    return data_root.expanduser().resolve()


def load_alfworld_fixed_train_case_shards(
    config: Any,
    *,
    train_batch_size: int,
    group_n: int,
    alfworld_data_root: str | Path | None = None,
) -> list[list[dict[str, Any]]] | None:
    batches = load_fixed_train_batches(
        config,
        environment="alfworld",
        train_batch_size=train_batch_size,
    )
    if batches is None:
        return None

    data_root = _resolve_alfworld_data_root(alfworld_data_root)
    num_workers = train_batch_size * group_n
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
    missing: list[Path] = []
    for batch in batches:
        assignments = build_fixed_train_case_assignments(
            batch,
            env_num=train_batch_size,
            group_n=group_n,
        )
        for worker_index, case in enumerate(assignments):
            fixed_case = dict(case)
            relative_path = fixed_case.get("gamefile_relative")
            if not isinstance(relative_path, str) or not relative_path.strip():
                raise ValueError("Fixed ALFWorld training case is missing gamefile_relative")
            gamefile = data_root / relative_path
            fixed_case["gamefile"] = str(gamefile)
            if not gamefile.is_file():
                missing.append(gamefile)
            shards[worker_index].append(fixed_case)
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} fixed ALFWorld training gamefiles do not exist: {preview}"
        )
    return shards


def install_fixed_webshop_train_goal(env: Any, fixed_case: Mapping[str, Any]) -> str:
    case_id = str(fixed_case["case_id"])
    session_suffix = re.sub(r"[^a-zA-Z0-9]+", "-", case_id).strip("-")
    session_id = f"fixed-train-{session_suffix}"
    env.server.user_sessions[session_id] = {
        "goal": copy.deepcopy(fixed_case["goal"]),
        "done": False,
    }
    return session_id
