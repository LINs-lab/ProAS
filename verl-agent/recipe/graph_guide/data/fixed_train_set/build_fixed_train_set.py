#!/usr/bin/env python3
"""Build fixed ALFWorld/WebShop training schedules from historical rollouts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from recipe.graph_guide.data.fixed_val_set.webshop import (
    build_webshop_candidate_catalog,
    webshop_base_task_key,
    webshop_exact_fingerprint,
    webshop_exact_task_key,
)


SCHEMA_VERSION = 1
DEFAULT_HORIZON = 150
DEFAULT_TRAIN_BATCH_SIZE = 16
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
DEFAULT_FIXED_VAL_ROOT = REPO_ROOT / "recipe/graph_guide/data/fixed_val_set"
DEFAULT_WEBSHOP_DATA_ROOT = (
    REPO_ROOT / "agent_system/environments/env_package/webshop/webshop/data"
)
DEFAULT_ROLLOUT_DIRS = {
    "alfworld": {
        0: REPO_ROOT / "official_outputs/grpo_alfworld/rollout_data",
        1: REPO_ROOT / "official_outputs/grpo_alfworld_seed1/rollout_data",
        2: REPO_ROOT / "official_outputs/grpo_alfworld_seed2/rollout_data",
    },
    "webshop": {
        0: REPO_ROOT / "official_outputs/webshop/grpo_webshop/rollout_data",
        1: REPO_ROOT / "official_outputs/webshop/grpo_webshop_seed1/rollout_data",
        2: REPO_ROOT / "official_outputs/webshop/grpo_webshop_seed2/rollout_data",
    },
}
JSON_ROOT_MARKER = "json_2.1.1/"
PRICE_PATTERN = re.compile(
    r"price lower than\s+([0-9]+(?:\.[0-9]+)?)\s+dollars",
    flags=re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_source_path(path: Path, env_var: str | None = None) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        if env_var:
            return f"${{{env_var}}}/{resolved.name}"
        return resolved.name


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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


def _safe_identity(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip() or fallback
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", text).strip("-")
    return safe or fallback


def normalize_alfworld_gamefile(gamefile: str) -> str:
    text = str(gamefile or "").strip().replace("\\", "/")
    if not text:
        raise ValueError("ALFWorld rollout row is missing gamefile")
    if JSON_ROOT_MARKER in text:
        return text.split(JSON_ROOT_MARKER, 1)[1].lstrip("/")
    parts = Path(text).parts
    if "train" in parts:
        train_index = parts.index("train")
        return Path(*parts[train_index:]).as_posix()
    raise ValueError(f"Cannot normalize ALFWorld gamefile path: {gamefile}")


def _alfworld_task_family(gamefile_relative: str) -> str:
    parts = Path(gamefile_relative).parts
    if len(parts) < 2:
        raise ValueError(f"Invalid ALFWorld gamefile relative path: {gamefile_relative}")
    return parts[1].split("-", 1)[0]


def _rollout_step_from_path(path: Path) -> int:
    if not path.stem.isdigit():
        raise ValueError(f"Rollout filename must be a numeric step: {path}")
    step = int(path.stem)
    if step <= 0:
        raise ValueError(f"Rollout step must be positive: {path}")
    return step


def _source_rows_by_group(path: Path, train_batch_size: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Rollout file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for line_number, row in _iter_jsonl(path):
        if row.get("traj_step_idx") != 0:
            continue
        group_uid = row.get("episode_group_uid")
        if not isinstance(group_uid, str) or not group_uid.strip():
            raise ValueError(
                f"Rollout source row is missing episode_group_uid at {path}:{line_number}"
            )
        if group_uid in seen_groups:
            continue
        seen_groups.add(group_uid)
        rows.append(row)
        if len(rows) == train_batch_size:
            break
    if len(rows) != train_batch_size:
        raise ValueError(
            f"Expected {train_batch_size} source episode groups in {path}, "
            f"found {len(rows)}"
        )
    return rows


def _alfworld_task_from_row(row: Mapping[str, Any], *, step: int, slot: int) -> dict[str, Any]:
    gamefile_relative = normalize_alfworld_gamefile(str(row.get("gamefile") or ""))
    task_instance = str(row.get("task_instance") or Path(gamefile_relative).parent.name)
    task_desc = str(row.get("task_desc") or "").strip()
    if not task_desc:
        raise ValueError(f"ALFWorld source row is missing task_desc at step {step}")
    task_type = _alfworld_task_family(gamefile_relative)
    return {
        "case_id": f"alfworld:step{step:03d}:slot{slot:03d}:{_safe_identity(task_instance, str(slot))}",
        "environment": "alfworld",
        "episode_group_uid": row["episode_group_uid"],
        "gamefile_relative": gamefile_relative,
        "slot": slot,
        "task_desc": task_desc,
        "task_instance": task_instance,
        "task_type": task_type,
    }


def parse_webshop_price_upper(task_desc: str) -> float | None:
    match = PRICE_PATTERN.search(task_desc)
    if match is None:
        return None
    return float(match.group(1))


def build_webshop_goal_lookup(
    items_path: Path,
    attributes_path: Path,
    seeds: Iterable[int] = (0,),
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        for candidate in build_webshop_candidate_catalog(
            items_path,
            attributes_path,
            seed=seed,
        ):
            lookup.setdefault(str(candidate["base_task_key"]), candidate)
    return lookup


def _webshop_task_from_row(
    row: Mapping[str, Any],
    *,
    step: int,
    slot: int,
    webshop_goal_lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    task_desc = str(row.get("task_desc") or "").strip()
    if not task_desc:
        raise ValueError(f"WebShop source row is missing task_desc at step {step}")
    base_task_key = webshop_base_task_key(task_desc)
    candidate = webshop_goal_lookup.get(base_task_key)
    if candidate is None:
        raise ValueError(
            "Cannot recover WebShop goal payload for rollout task: "
            f"step={step} slot={slot} base_task_key={base_task_key}"
        )

    goal = copy.deepcopy(candidate.get("goal", candidate))
    if not isinstance(goal, dict):
        raise ValueError(f"Invalid WebShop catalog goal at step {step} slot {slot}")
    goal["instruction_text"] = task_desc
    price_upper = parse_webshop_price_upper(task_desc)
    if price_upper is not None:
        goal["price_upper"] = price_upper
    goal.setdefault("weight", 1.0)

    return {
        "base_task_key": base_task_key,
        "case_id": f"webshop:step{step:03d}:slot{slot:03d}:{base_task_key[:12]}",
        "environment": "webshop",
        "episode_group_uid": row["episode_group_uid"],
        "exact_fingerprint": webshop_exact_fingerprint(goal),
        "exact_task_key": webshop_exact_task_key(task_desc),
        "goal": goal,
        "slot": slot,
        "task_desc": task_desc,
    }


def extract_step_batch(
    rollout_file: Path | str,
    *,
    environment: str,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    webshop_goal_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    path = Path(rollout_file)
    step = _rollout_step_from_path(path)
    rows = _source_rows_by_group(path, train_batch_size)
    if environment == "alfworld":
        tasks = [
            _alfworld_task_from_row(row, step=step, slot=slot)
            for slot, row in enumerate(rows)
        ]
    elif environment == "webshop":
        if webshop_goal_lookup is None:
            raise ValueError("webshop_goal_lookup is required for WebShop extraction")
        tasks = [
            _webshop_task_from_row(
                row,
                step=step,
                slot=slot,
                webshop_goal_lookup=webshop_goal_lookup,
            )
            for slot, row in enumerate(rows)
        ]
    else:
        raise ValueError(f"Unsupported fixed train environment: {environment}")
    return {"step": step, "tasks": tasks}


def build_fixed_train_schedule(
    rollout_dir: Path | str,
    *,
    environment: str,
    horizon: int = DEFAULT_HORIZON,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    webshop_goal_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    rollout_root = Path(rollout_dir)
    if not rollout_root.is_dir():
        raise FileNotFoundError(f"Rollout directory does not exist: {rollout_root}")
    return [
        extract_step_batch(
            rollout_root / f"{step}.jsonl",
            environment=environment,
            train_batch_size=train_batch_size,
            webshop_goal_lookup=webshop_goal_lookup,
        )
        for step in range(1, horizon + 1)
    ]


def _read_fixed_val_rows(fixed_val_root: Path, filename: str) -> list[dict[str, Any]]:
    path = fixed_val_root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Fixed validation asset does not exist: {path}")
    return [row for _, row in _iter_jsonl(path)]


def audit_schedule_against_fixed_val(
    batches: Iterable[Mapping[str, Any]],
    fixed_val_root: Path | str,
    environment: str,
) -> dict[str, Any]:
    batch_list = list(batches)
    fixed_val_path = Path(fixed_val_root)
    train_tasks = [task for batch in batch_list for task in batch.get("tasks", [])]
    if environment == "alfworld":
        val_rows = _read_fixed_val_rows(fixed_val_path, "alfworld.jsonl")
        val_instances = {str(row.get("task_instance")) for row in val_rows}
        val_gamefiles = {str(row.get("gamefile_relative")) for row in val_rows}
        train_instances = {str(task.get("task_instance")) for task in train_tasks}
        train_gamefiles = {str(task.get("gamefile_relative")) for task in train_tasks}
        instance_overlap = sorted(train_instances & val_instances)
        gamefile_overlap = sorted(train_gamefiles & val_gamefiles)
        if instance_overlap or gamefile_overlap:
            raise ValueError(
                "alfworld fixed train schedule overlaps fixed validation: "
                f"task_instance={instance_overlap[:5]} "
                f"gamefile_relative={gamefile_overlap[:5]}"
            )
        return {
            "fixed_val_case_count": len(val_rows),
            "fixed_val_overlap_count": 0,
            "train_case_count": len(train_tasks),
            "train_unique_gamefiles": len(train_gamefiles),
            "train_unique_task_instances": len(train_instances),
        }
    if environment == "webshop":
        val_rows = _read_fixed_val_rows(fixed_val_path, "webshop.jsonl")
        val_base_keys = {
            str(row.get("base_task_key") or webshop_base_task_key(row.get("task_desc", "")))
            for row in val_rows
        }
        train_base_keys = {str(task.get("base_task_key")) for task in train_tasks}
        overlap = sorted(train_base_keys & val_base_keys)
        if overlap:
            raise ValueError(
                "webshop fixed train schedule overlaps fixed validation: "
                f"base_task_key={overlap[:5]}"
            )
        return {
            "fixed_val_case_count": len(val_rows),
            "fixed_val_overlap_count": 0,
            "train_case_count": len(train_tasks),
            "train_unique_base_tasks": len(train_base_keys),
            "train_unique_exact_tasks": len(
                {str(task.get("exact_task_key")) for task in train_tasks}
            ),
        }
    raise ValueError(f"Unsupported fixed train environment: {environment}")


def write_schedule(
    output_dir: Path | str,
    batches: list[dict[str, Any]],
    *,
    environment: str,
    seed: int,
    source_rollout_dir: Path | str,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    fixed_val_root: Path | str = DEFAULT_FIXED_VAL_ROOT,
    extra_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("Cannot write an empty fixed train schedule")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    schedule_path = output_path / "schedule.jsonl"
    _write_jsonl(schedule_path, batches)
    audit = audit_schedule_against_fixed_val(batches, fixed_val_root, environment)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment,
        "seed": seed,
        "horizon": len(batches),
        "train_batch_size": train_batch_size,
        "audits": {
            "fixed_val_overlap": audit,
        },
        "sources": {
            "source_algorithm": "grpo",
            "source_rollout_dir": _portable_source_path(Path(source_rollout_dir)),
            "fixed_val_root": _portable_source_path(Path(fixed_val_root)),
            **dict(extra_sources or {}),
        },
        "output_sha256": {
            "schedule.jsonl": sha256_file(schedule_path),
        },
    }
    _write_json(output_path / "manifest.json", manifest)
    return manifest


def _readme_content() -> str:
    return """# Fixed Train Set

This directory stores path-addressable fixed training schedules for ALFWorld and
WebShop. The runtime switch is intentionally small:

```bash
env.fixed_train_set.path=recipe/graph_guide/data/fixed_train_set/seed0/alfworld
```

If `env.fixed_train_set.path` is unset, `null`, or empty, training keeps the
original online sampler.

## Layout

- `seed0/alfworld`, `seed1/alfworld`, `seed2/alfworld`
- `seed0/webshop`, `seed1/webshop`, `seed2/webshop`

Each environment directory contains:

- `schedule.jsonl`: one JSON object per RL step, with 16 task slots.
- `manifest.json`: schema, source rollout path, horizon, hashes, and the
  fixed-validation overlap audit result.

The schedules are generated from the GRPO official rollout traces and preserve
the first-seen `episode_group_uid` order within each step. This keeps duplicate
task identities in distinct rollout groups when they occurred in the original
training batch.

## Regenerate

```bash
cd "$(git rev-parse --show-toplevel)/verl-agent"
PYTHONPATH=. python recipe/graph_guide/data/fixed_train_set/build_fixed_train_set.py --horizon 150
```

The overlap audit is performed during generation only. Runtime loading does not
repeat this audit.
"""


def write_readme(output_root: Path | str) -> None:
    path = Path(output_root) / "README.md"
    path.write_text(_readme_content(), encoding="utf-8")


def build_all_fixed_train_sets(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fixed_val_root: Path = DEFAULT_FIXED_VAL_ROOT,
    horizon: int = DEFAULT_HORIZON,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    webshop_items: Path = DEFAULT_WEBSHOP_DATA_ROOT / "items_shuffle_1000.json",
    webshop_attributes: Path = DEFAULT_WEBSHOP_DATA_ROOT / "items_ins_v2_1000.json",
    rollout_dirs: Mapping[str, Mapping[int, Path]] = DEFAULT_ROLLOUT_DIRS,
) -> dict[str, Any]:
    output_root = Path(output_root)
    fixed_val_root = Path(fixed_val_root)
    write_readme(output_root)

    webshop_goal_lookup = build_webshop_goal_lookup(
        webshop_items,
        webshop_attributes,
        seeds=range(16),
    )
    manifests: dict[str, Any] = {}
    for seed in (0, 1, 2):
        for environment in ("alfworld", "webshop"):
            goal_lookup = webshop_goal_lookup if environment == "webshop" else None
            rollout_dir = Path(rollout_dirs[environment][seed])
            batches = build_fixed_train_schedule(
                rollout_dir,
                environment=environment,
                horizon=horizon,
                train_batch_size=train_batch_size,
                webshop_goal_lookup=goal_lookup,
            )
            manifest = write_schedule(
                output_root / f"seed{seed}" / environment,
                batches,
                environment=environment,
                seed=seed,
                source_rollout_dir=rollout_dir,
                train_batch_size=train_batch_size,
                fixed_val_root=fixed_val_root,
                extra_sources=(
                    {
                        "webshop_items": _portable_source_path(
                            Path(webshop_items), "WEBSHOP_DATA_DIR"
                        ),
                        "webshop_items_sha256": sha256_file(Path(webshop_items)),
                        "webshop_attributes": _portable_source_path(
                            Path(webshop_attributes), "WEBSHOP_DATA_DIR"
                        ),
                        "webshop_attributes_sha256": sha256_file(Path(webshop_attributes)),
                        "webshop_catalog_seed_range": [0, 15],
                    }
                    if environment == "webshop"
                    else None
                ),
            )
            manifests[f"seed{seed}/{environment}"] = manifest
    return manifests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fixed-val-root", type=Path, default=DEFAULT_FIXED_VAL_ROOT)
    parser.add_argument(
        "--webshop-items",
        type=Path,
        default=DEFAULT_WEBSHOP_DATA_ROOT / "items_shuffle_1000.json",
    )
    parser.add_argument(
        "--webshop-attributes",
        type=Path,
        default=DEFAULT_WEBSHOP_DATA_ROOT / "items_ins_v2_1000.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifests = build_all_fixed_train_sets(
        output_root=args.output_root,
        fixed_val_root=args.fixed_val_root,
        horizon=args.horizon,
        train_batch_size=args.train_batch_size,
        webshop_items=args.webshop_items,
        webshop_attributes=args.webshop_attributes,
    )
    print(json.dumps(manifests, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
