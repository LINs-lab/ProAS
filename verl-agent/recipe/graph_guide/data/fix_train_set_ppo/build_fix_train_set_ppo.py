#!/usr/bin/env python3
"""Build fixed 128-task PPO schedules that mirror the online samplers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import random
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from recipe.graph_guide.data.fixed_val_set.webshop import (
    webshop_base_task_key,
    webshop_exact_fingerprint,
    webshop_exact_task_key,
)


SCHEMA_VERSION = 1
DEFAULT_HORIZON = 150
DEFAULT_TRAIN_BATCH_SIZE = 128
SEEDS = (0, 1, 2)
TASK_TYPES = {
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
}
PRICE_RANGE = [10.0 * index for index in range(1, 100)]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
DEFAULT_FIXED_VAL_ROOT = REPO_ROOT / "recipe/graph_guide/data/fixed_val_set"
DEFAULT_WEBSHOP_DATA_ROOT = (
    REPO_ROOT / "agent_system/environments/env_package/webshop/webshop/data"
)


def _load_json(path: Path, expected_type: type) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, expected_type):
        raise ValueError(f"Expected {expected_type.__name__} in {path}")
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_task_description(metadata: Mapping[str, Any], path: Path) -> str:
    annotations = metadata.get("turk_annotations", {}).get("anns", [])
    for annotation in annotations:
        task_desc = annotation.get("task_desc") if isinstance(annotation, dict) else None
        if isinstance(task_desc, str) and task_desc.strip():
            return task_desc.strip()
    raise ValueError(f"No ALFWorld task description found in {path}")


def collect_alfworld_train_games(data_root: Path) -> list[dict[str, Any]]:
    """Reproduce AlfredTWEnv.collect_game_files, including its os.walk order."""
    train_root = data_root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"ALFWorld train split not found: {train_root}")

    games = []
    for root, _dirs, files in os.walk(train_root, topdown=False):
        if "traj_data.json" not in files or "movable" in root or "Sliced" in root:
            continue
        root_path = Path(root)
        metadata_path = root_path / "traj_data.json"
        gamefile = root_path / "game.tw-pddl"
        metadata = _load_json(metadata_path, dict)
        if metadata.get("task_type") not in TASK_TYPES or not gamefile.is_file():
            continue
        game_data = _load_json(gamefile, dict)
        if not game_data.get("solvable", False):
            continue
        games.append(
            {
                "gamefile_relative": gamefile.relative_to(data_root).as_posix(),
                "task_desc": _first_task_description(metadata, metadata_path),
                "task_instance": str(metadata.get("task_id") or root_path.name),
                "task_type": str(metadata["task_type"]),
            }
        )
    if not games:
        raise ValueError(f"No eligible ALFWorld games found under {train_root}")
    return games


def build_alfworld_schedule(
    games: list[dict[str, Any]],
    *,
    seed: int,
    horizon: int,
    train_batch_size: int,
) -> list[dict[str, Any]]:
    worker_sequences = []
    for slot in range(train_batch_size):
        order = np.arange(len(games))
        np.random.RandomState(seed + slot).shuffle(order)
        worker_sequences.append(order[:horizon])

    batches = []
    for step in range(1, horizon + 1):
        tasks = []
        for slot, sequence in enumerate(worker_sequences):
            source = games[int(sequence[step - 1])]
            tasks.append(
                {
                    "case_id": f"alfworld:seed{seed}:step{step:03d}:slot{slot:03d}",
                    "environment": "alfworld",
                    "slot": slot,
                    **source,
                }
            )
        batches.append({"step": step, "tasks": tasks})
    return batches


def _parse_prices(raw: Any) -> list[float]:
    if not raw:
        return [100.0]
    if isinstance(raw, list):
        return [float(value) for value in raw][:2]
    return [
        float(Decimal(re.sub(r"[^\d.]", "", value)))
        for value in str(raw).split("$")[1:]
    ][:2]


def _webshop_products(items_path: Path) -> list[dict[str, Any]]:
    products = _load_json(items_path, list)
    eligible = []
    seen_asins: set[str] = set()
    for product in products:
        asin = str(product.get("asin") or "")
        if not asin or asin == "nan" or len(asin) > 10 or asin in seen_asins:
            continue
        seen_asins.add(asin)
        options = {}
        for raw_name, raw_values in (product.get("customization_options") or {}).items():
            if raw_values is None:
                continue
            options[str(raw_name).lower()] = [
                str(item["value"]).strip().replace("/", " | ").lower()
                for item in raw_values
            ]
        eligible.append(
            {
                **product,
                "options": options,
                "pricing_values": _parse_prices(product.get("pricing")),
                "query": str(product.get("query") or "").lower().strip(),
            }
        )
    return eligible


def build_webshop_worker_catalog(
    products: list[dict[str, Any]],
    attributes: Mapping[str, Any],
    *,
    worker_seed: int,
) -> list[dict[str, Any]]:
    """Reproduce WebAgentTextEnv's synthetic goal generation and shuffle."""
    generation_rng = random.Random(worker_seed)
    prices = {}
    for product in products:
        values = product["pricing_values"]
        prices[product["asin"]] = (
            values[0] if len(values) == 1 else generation_rng.uniform(*values[:2])
        )

    goals = []
    attribute_counts: Counter[str] = Counter()
    for product in products:
        asin = product["asin"]
        attribute_record = attributes.get(asin) or {}
        instruction = attribute_record.get("instruction")
        if instruction is None:
            continue
        instruction_attributes = list(
            attribute_record.get("instruction_attributes") or []
        )
        price_range = [price for price in PRICE_RANGE if price > prices[asin]][:4]
        if len(price_range) >= 2:
            _, price_upper = sorted(generation_rng.sample(price_range, 2))
            price_text = f", and price lower than {price_upper:.2f} dollars"
        else:
            price_upper = 1_000_000.0
            price_text = ""

        option_names = sorted(product["options"])
        combinations = itertools.product(
            *(product["options"][name] for name in option_names)
        )
        for combination in combinations:
            goal_options = dict(zip(option_names, combination, strict=True))
            option_text = ", and ".join(
                f"{name}: {value}" for name, value in goal_options.items()
            )
            instruction_text = str(instruction)
            if option_text:
                instruction_text += f" with {option_text}"
            instruction_text += price_text
            for attribute in instruction_attributes:
                attribute_counts[str(attribute)] += 1
            goals.append(
                {
                    "asin": asin,
                    "attributes": instruction_attributes,
                    "category": product.get("category"),
                    "goal_options": goal_options,
                    "instruction_text": instruction_text,
                    "name": product.get("name") or product.get("Title"),
                    "price_upper": price_upper,
                    "product_category": product.get("product_category"),
                    "query": product.get("query"),
                }
            )

    for goal in goals:
        attrs = goal["attributes"]
        goal["weight"] = (
            sum(1.0 / attribute_counts[str(attr)] for attr in attrs) / len(attrs)
            if attrs
            else 1.0
        )

    # SimServer resets Python's global RNG to the worker seed before shuffling.
    random.Random(worker_seed).shuffle(goals)
    return goals


def _webshop_task(goal: Mapping[str, Any], *, seed: int, step: int, slot: int) -> dict[str, Any]:
    task_desc = str(goal["instruction_text"])
    return {
        "base_task_key": webshop_base_task_key(task_desc),
        "case_id": f"webshop:seed{seed}:step{step:03d}:slot{slot:03d}",
        "environment": "webshop",
        "exact_fingerprint": webshop_exact_fingerprint(goal),
        "exact_task_key": webshop_exact_task_key(task_desc),
        "goal": copy.deepcopy(dict(goal)),
        "slot": slot,
        "task_desc": task_desc,
    }


def build_webshop_schedule(
    products: list[dict[str, Any]],
    attributes: Mapping[str, Any],
    *,
    seed: int,
    horizon: int,
    train_batch_size: int,
    val_base_keys: set[str],
) -> tuple[list[dict[str, Any]], int, int, int]:
    probe_catalog = build_webshop_worker_catalog(
        products, attributes, worker_seed=seed
    )
    goal_indices = np.arange(500, len(probe_catalog))
    rng = np.random.RandomState(seed)
    selected_by_step = [
        [int(index) for index in rng.choice(goal_indices, train_batch_size, replace=False)]
        for _ in range(horizon)
    ]
    used_by_step = [set(indices) for indices in selected_by_step]
    tasks_by_step: list[list[dict[str, Any] | None]] = [
        [None] * train_batch_size for _ in range(horizon)
    ]
    used_base_keys_by_step: list[set[str]] = [set() for _ in range(horizon)]
    fixed_val_replacements = 0
    duplicate_replacements = 0

    for slot in range(train_batch_size):
        catalog = build_webshop_worker_catalog(
            products, attributes, worker_seed=seed + slot
        )
        if len(catalog) != len(probe_catalog):
            raise ValueError("WebShop worker catalogs have inconsistent lengths")
        for step_index in range(horizon):
            selected_index = selected_by_step[step_index][slot]
            goal = catalog[selected_index]
            base_task_key = webshop_base_task_key(goal["instruction_text"])
            replacement_reason = None
            if base_task_key in val_base_keys:
                replacement_reason = "fixed_val"
            elif base_task_key in used_base_keys_by_step[step_index]:
                replacement_reason = "duplicate"
            if replacement_reason is not None:
                replacement = (selected_index + 1) % len(catalog)
                while (
                    replacement < 500
                    or replacement in used_by_step[step_index]
                    or (
                        webshop_base_task_key(
                            catalog[replacement]["instruction_text"]
                        )
                        in val_base_keys
                    )
                    or (
                        webshop_base_task_key(
                            catalog[replacement]["instruction_text"]
                        )
                        in used_base_keys_by_step[step_index]
                    )
                ):
                    replacement = (replacement + 1) % len(catalog)
                used_by_step[step_index].remove(selected_index)
                used_by_step[step_index].add(replacement)
                selected_by_step[step_index][slot] = replacement
                goal = catalog[replacement]
                if replacement_reason == "fixed_val":
                    fixed_val_replacements += 1
                else:
                    duplicate_replacements += 1
            used_base_keys_by_step[step_index].add(
                webshop_base_task_key(goal["instruction_text"])
            )
            tasks_by_step[step_index][slot] = _webshop_task(
                goal,
                seed=seed,
                step=step_index + 1,
                slot=slot,
            )

    batches = []
    for step_index, tasks in enumerate(tasks_by_step):
        if any(task is None for task in tasks):
            raise AssertionError("WebShop schedule contains an unassigned task")
        batches.append({"step": step_index + 1, "tasks": tasks})
    return (
        batches,
        len(probe_catalog),
        fixed_val_replacements,
        duplicate_replacements,
    )


def audit_schedule(
    batches: list[dict[str, Any]],
    *,
    environment: str,
    fixed_val_root: Path,
    train_batch_size: int,
) -> dict[str, Any]:
    if any(len(batch["tasks"]) != train_batch_size for batch in batches):
        raise ValueError(f"{environment} schedule has an invalid batch size")
    tasks = [task for batch in batches for task in batch["tasks"]]
    if environment == "alfworld":
        val_rows = list(_iter_jsonl(fixed_val_root / "alfworld.jsonl"))
        val_instances = {str(row["task_instance"]) for row in val_rows}
        val_gamefiles = {str(row["gamefile_relative"]) for row in val_rows}
        train_instances = {str(task["task_instance"]) for task in tasks}
        train_gamefiles = {str(task["gamefile_relative"]) for task in tasks}
        overlap = (train_instances & val_instances) | (train_gamefiles & val_gamefiles)
        unique_key = "train_unique_gamefiles"
        unique_count = len(train_gamefiles)
    else:
        val_rows = list(_iter_jsonl(fixed_val_root / "webshop.jsonl"))
        val_keys = {str(row["base_task_key"]) for row in val_rows}
        train_keys = {str(task["base_task_key"]) for task in tasks}
        unique_per_step = [
            len({str(task["base_task_key"]) for task in batch["tasks"]})
            for batch in batches
        ]
        if any(count != train_batch_size for count in unique_per_step):
            raise ValueError(
                "webshop fixed PPO schedule must contain a distinct base task "
                "for every slot"
            )
        overlap = train_keys & val_keys
        unique_key = "train_unique_base_tasks"
        unique_count = len(train_keys)
    if overlap:
        raise ValueError(
            f"{environment} fixed PPO schedule overlaps fixed validation: "
            f"{sorted(overlap)[:5]}"
        )
    audit = {
        "fixed_val_case_count": len(val_rows),
        "fixed_val_overlap_count": 0,
        "train_case_count": len(tasks),
        unique_key: unique_count,
    }
    if environment == "webshop":
        audit["unique_base_tasks_per_step"] = train_batch_size
    return audit


def write_schedule(
    output_dir: Path,
    batches: list[dict[str, Any]],
    *,
    environment: str,
    seed: int,
    train_batch_size: int,
    fixed_val_root: Path,
    sources: Mapping[str, Any],
    sampler_audit: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = output_dir / "schedule.jsonl"
    _write_jsonl(schedule_path, batches)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment,
        "seed": seed,
        "horizon": len(batches),
        "train_batch_size": train_batch_size,
        "audits": {
            "fixed_val_overlap": audit_schedule(
                batches,
                environment=environment,
                fixed_val_root=fixed_val_root,
                train_batch_size=train_batch_size,
            ),
            "sampler": dict(sampler_audit),
        },
        "sources": dict(sources),
        "output_sha256": {"schedule.jsonl": _sha256_file(schedule_path)},
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_all(
    *,
    output_root: Path,
    fixed_val_root: Path,
    alfworld_data_root: Path,
    webshop_items: Path,
    webshop_attributes: Path,
    horizon: int,
    train_batch_size: int,
) -> dict[str, Any]:
    if horizon <= 0 or horizon > 3553:
        raise ValueError("horizon must be between 1 and the ALFWorld pool size")
    if train_batch_size != 128:
        raise ValueError("PPO-correct schedules are defined for train_batch_size=128")

    output_root.mkdir(parents=True, exist_ok=True)
    alfworld_games = collect_alfworld_train_games(alfworld_data_root)
    products = _webshop_products(webshop_items)
    attributes = _load_json(webshop_attributes, dict)
    val_base_keys = {
        str(row["base_task_key"])
        for row in _iter_jsonl(fixed_val_root / "webshop.jsonl")
    }

    manifests = {}
    for seed in SEEDS:
        alfworld_batches = build_alfworld_schedule(
            alfworld_games,
            seed=seed,
            horizon=horizon,
            train_batch_size=train_batch_size,
        )
        manifests[f"seed{seed}/alfworld"] = write_schedule(
            output_root / f"seed{seed}" / "alfworld",
            alfworld_batches,
            environment="alfworld",
            seed=seed,
            train_batch_size=train_batch_size,
            fixed_val_root=fixed_val_root,
            sources={
                "alfworld_data_root": "$ALFWORLD_DATA/json_2.1.1",
                "sampler": "TextworldBatchGymEnv seeded shuffled_cycle per worker",
            },
            sampler_audit={
                "eligible_game_count": len(alfworld_games),
                "worker_seed_rule": "seed + slot",
            },
        )

        (
            webshop_batches,
            catalog_size,
            fixed_val_replacements,
            duplicate_replacements,
        ) = build_webshop_schedule(
            products,
            attributes,
            seed=seed,
            horizon=horizon,
            train_batch_size=train_batch_size,
            val_base_keys=val_base_keys,
        )
        manifests[f"seed{seed}/webshop"] = write_schedule(
            output_root / f"seed{seed}" / "webshop",
            webshop_batches,
            environment="webshop",
            seed=seed,
            train_batch_size=train_batch_size,
            fixed_val_root=fixed_val_root,
            sources={
                "sampler": "WebshopMultiProcessEnv RandomState(seed) index sampling",
                "webshop_attributes": (
                    "agent_system/environments/env_package/webshop/webshop/data/"
                    "items_ins_v2_1000.json"
                ),
                "webshop_attributes_sha256": _sha256_file(webshop_attributes),
                "webshop_items": (
                    "agent_system/environments/env_package/webshop/webshop/data/"
                    "items_shuffle_1000.json"
                ),
                "webshop_items_sha256": _sha256_file(webshop_items),
            },
            sampler_audit={
                "catalog_size_per_worker": catalog_size,
                "goal_index_range": [500, catalog_size],
                "fixed_val_replacements": fixed_val_replacements,
                "duplicate_task_replacements": duplicate_replacements,
                "replacement_rule": (
                    "next eligible unused goal index with a new base task identity"
                ),
                "worker_seed_rule": "seed + slot",
            },
        )
    return manifests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fixed-val-root", type=Path, default=DEFAULT_FIXED_VAL_ROOT)
    parser.add_argument(
        "--alfworld-data-root",
        type=Path,
        default=Path(os.environ.get("ALFWORLD_DATA", "/mnt/public/dingbowen/data_zoo/alfworld"))
        / "json_2.1.1",
    )
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
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifests = build_all(
        output_root=args.output_root.resolve(),
        fixed_val_root=args.fixed_val_root.resolve(),
        alfworld_data_root=args.alfworld_data_root.resolve(),
        webshop_items=args.webshop_items.resolve(),
        webshop_attributes=args.webshop_attributes.resolve(),
        horizon=args.horizon,
        train_batch_size=args.train_batch_size,
    )
    print(json.dumps(manifests, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
