#!/usr/bin/env python3
"""Build deterministic, leak-free ALFWorld and WebShop validation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from recipe.graph_guide.data.fixed_val_set.alfworld import select_alfworld_cases
from recipe.graph_guide.data.fixed_val_set.webshop import (
    build_webshop_candidate_catalog,
    canonical_webshop_split,
    load_or_scan_training_exposure,
)


SCHEMA_VERSION = 2
DEFAULT_EXPECTED_COUNT = 128
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ALFWORLD_DATA_ROOT = (
    Path(os.environ.get("ALFWORLD_DATA", Path.home() / ".cache/alfworld"))
    / "json_2.1.1"
)
DEFAULT_WEBSHOP_DATA_ROOT = (
    REPO_ROOT / "agent_system/environments/env_package/webshop/webshop/data"
)
DEFAULT_WEBSHOP_ROLLOUT_ROOTS = (
    REPO_ROOT / "official_outputs/webshop/grpo_webshop",
    REPO_ROOT / "official_outputs/webshop/grpo_webshop_seed1",
    REPO_ROOT / "official_outputs/webshop/grpo_webshop_seed2",
)
DEFAULT_WEBSHOP_CANONICAL_SEED = 42
DEFAULT_WEBSHOP_EVAL_START = 500
DEFAULT_WEBSHOP_EVAL_STOP = 1500
DEFAULT_WEBSHOP_TRAIN_START = 1500
GENERATED_FILENAMES = (
    "README.md",
    "alfworld.jsonl",
    "alfworld_gamefiles.txt",
    "webshop.jsonl",
    "webshop_goal_fingerprints.txt",
    "webshop_training_exposure.json",
)
LEGACY_FILENAMES = (
    "webshop_goal_indices.txt",
    "webshop_worker_assignments.tsv",
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


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def discover_rollout_files(roots: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for raw_root in roots:
        root = raw_root.resolve()
        if not root.is_dir():
            raise ValueError(f"WebShop rollout root does not exist: {root}")
        files.update(root.glob("**/rollout_data/*.jsonl"))
    if not files:
        raise ValueError("No WebShop training rollout files were found")
    return sorted(path.resolve() for path in files)


def _selection_key(candidate: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        str(candidate["base_task_key"]).encode("utf-8")
    ).hexdigest()


def _select_webshop_cases(
    candidates: list[dict[str, Any]],
    historical_base_keys: set[str],
    canonical_train_keys: set[str],
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unseen = [
        candidate
        for candidate in candidates
        if candidate["base_task_key"] not in historical_base_keys
        and candidate["base_task_key"] not in canonical_train_keys
    ]
    if len(unseen) < count:
        raise ValueError(
            f"WebShop validation requires {count} unseen unique tasks, "
            f"found {len(unseen)}"
        )

    selected_candidates = sorted(unseen, key=_selection_key)[:count]
    selected_keys = {
        str(candidate["base_task_key"]) for candidate in selected_candidates
    }
    selected_exact = {
        str(candidate["exact_fingerprint"]) for candidate in selected_candidates
    }
    historical_overlap = selected_keys & historical_base_keys
    canonical_train_overlap = selected_keys & canonical_train_keys
    if len(selected_keys) != count or len(selected_exact) != count:
        raise ValueError("WebShop validation selection contains duplicate tasks")
    if historical_overlap or canonical_train_overlap:
        raise ValueError("WebShop validation selection overlaps training tasks")

    rows = []
    for index, candidate in enumerate(selected_candidates):
        goal = candidate["goal"]
        rows.append(
            {
                "base_task_key": candidate["base_task_key"],
                "case_id": f"webshop:{index:03d}:{candidate['base_task_key'][:12]}",
                "environment": "webshop",
                "exact_fingerprint": candidate["exact_fingerprint"],
                "goal": goal,
                "source_type": candidate["source_type"],
                "task_desc": goal["instruction_text"],
            }
        )

    audit = {
        "candidate_count": len(candidates),
        "historically_exposed_candidate_count": len(candidates) - len(unseen),
        "unseen_candidate_count": len(unseen),
        "selected_count": len(rows),
        "selected_unique_base_tasks": len(selected_keys),
        "selected_unique_exact_fingerprints": len(selected_exact),
        "historical_overlap_count": len(historical_overlap),
        "canonical_train_task_count": len(canonical_train_keys),
        "canonical_train_overlap_count": len(canonical_train_overlap),
        "selection_rule": "sha256(base_task_key)",
        "source_type_counts": {
            source_type: sum(
                row["source_type"] == source_type for row in rows
            )
            for source_type in sorted({row["source_type"] for row in rows})
        },
    }
    return rows, audit


def _readme_content() -> str:
    return """# 固定验证集

本目录保存 ALFWorld 和 WebShop 各 128 条固定、唯一且与训练任务隔离的验证任务。
这些文件是任务 manifest，不包含模型 rollout。

## 身份与隔离

- ALFWorld 来自官方 `valid_seen`，按 `game.tw-pddl` 相对路径稳定选择，并验证
  task instance 不属于官方 `train`。
- WebShop 使用 seed 42 的 canonical catalog，并从原始 benchmark 的
  `eval=[500,1500)` 选择任务；`train=[1500,len(goals))` 单独参与零重叠审计。
  历史训练 exposure 只扫描 seed0/1/2 的 GRPO rollout 基准。重叠按去除随机
  价格阈值后的 base task key 判断，因此同一目标换一个价格上限仍被视为已经见过。

## 文件

- `alfworld.jsonl` / `alfworld_gamefiles.txt`：ALFWorld 任务元数据和加载顺序。
- `webshop.jsonl`：WebShop 完整 goal payload。
- `webshop_goal_fingerprints.txt`：WebShop exact fingerprint 加载顺序。
- `webshop_training_exposure.json`：历史训练 rollout exposure cache。
- `manifest.json`：数据来源、计数、零重叠审计和输出 SHA256。

## 生成与审计

```bash
cd "$(git rev-parse --show-toplevel)/verl-agent"
PYTHONPATH=. python recipe/graph_guide/data/fixed_val_set/build_fixed_val_set.py
PYTHONPATH=. python recipe/graph_guide/data/fixed_val_set/build_fixed_val_set.py --audit-only
PYTHONPATH=. pytest -q recipe/graph_guide/data/fixed_val_set/tests
```

ALFWorld 验证环境按 `alfworld_gamefiles.txt` 指定的顺序 reset。WebShop 验证环境
注入 `webshop.jsonl` 中的完整 `goal`，不会把 fingerprint 当作 worker-local goal
index。公共配置默认不启用；需要固定评测时显式设置
`env.fixed_val_set.enabled=true`，并保持 `data.val_batch_size=128`。
"""


def build_fixed_val_set(
    alfworld_data_root: Path,
    webshop_items: Path,
    webshop_attributes: Path,
    webshop_rollout_paths: Iterable[Path],
    output_dir: Path,
    expected_count: int = DEFAULT_EXPECTED_COUNT,
    webshop_canonical_seed: int = DEFAULT_WEBSHOP_CANONICAL_SEED,
    webshop_split_start: int = DEFAULT_WEBSHOP_EVAL_START,
    webshop_split_stop: int = DEFAULT_WEBSHOP_EVAL_STOP,
    webshop_train_start: int = DEFAULT_WEBSHOP_TRAIN_START,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    alfworld_cases, alfworld_audit = select_alfworld_cases(
        alfworld_data_root,
        count=expected_count,
    )
    catalog = build_webshop_candidate_catalog(
        webshop_items,
        webshop_attributes,
        seed=webshop_canonical_seed,
    )
    candidates = canonical_webshop_split(
        catalog,
        seed=webshop_canonical_seed,
        start=webshop_split_start,
        stop=webshop_split_stop,
    )
    canonical_train = (
        canonical_webshop_split(
            catalog,
            seed=webshop_canonical_seed,
            start=webshop_train_start,
            stop=len(catalog),
        )
        if webshop_train_start < len(catalog)
        else []
    )
    canonical_train_keys = {
        str(candidate["base_task_key"]) for candidate in canonical_train
    }

    rollout_paths = list(webshop_rollout_paths)
    exposure_cache = output_dir / "webshop_training_exposure.json"
    exposure = load_or_scan_training_exposure(rollout_paths, exposure_cache)
    historical_base_keys = set(exposure["base_task_keys"])
    webshop_cases, webshop_audit = _select_webshop_cases(
        candidates,
        historical_base_keys,
        canonical_train_keys,
        expected_count,
    )
    webshop_audit["canonical_seed"] = webshop_canonical_seed
    webshop_audit["canonical_split"] = "eval"
    webshop_audit["canonical_split_range"] = [
        webshop_split_start,
        webshop_split_stop,
    ]
    webshop_audit["official_train_range"] = [webshop_train_start, len(catalog)]
    webshop_audit["raw_catalog_count"] = len(catalog)

    if len(alfworld_cases) != expected_count or len(webshop_cases) != expected_count:
        raise ValueError("Fixed validation build produced an unexpected task count")

    with tempfile.TemporaryDirectory(
        prefix=".fixed_val_set.",
        dir=output_dir.parent,
    ) as temporary_dir:
        staging = Path(temporary_dir)
        _write_jsonl(staging / "alfworld.jsonl", alfworld_cases)
        _write_text(
            staging / "alfworld_gamefiles.txt",
            "".join(f"{case['gamefile_relative']}\n" for case in alfworld_cases),
        )
        _write_jsonl(staging / "webshop.jsonl", webshop_cases)
        _write_text(
            staging / "webshop_goal_fingerprints.txt",
            "".join(f"{case['exact_fingerprint']}\n" for case in webshop_cases),
        )
        persisted_exposure = {
            key: value for key, value in exposure.items() if key != "cache_hit"
        }
        _write_json(
            staging / "webshop_training_exposure.json",
            persisted_exposure,
        )
        _write_text(staging / "README.md", _readme_content())

        output_hashes = {
            filename: sha256_file(staging / filename)
            for filename in GENERATED_FILENAMES
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "counts": {
                "alfworld": len(alfworld_cases),
                "webshop": len(webshop_cases),
            },
            "unique_tasks": {
                "alfworld": len(
                    {case["task_instance"] for case in alfworld_cases}
                ),
                "webshop": len(
                    {case["base_task_key"] for case in webshop_cases}
                ),
            },
            "audits": {
                "alfworld": alfworld_audit,
                "webshop": webshop_audit,
            },
            "sources": {
                "alfworld_data_root": _portable_source_path(
                    alfworld_data_root, "ALFWORLD_DATA"
                ),
                "webshop_items": _portable_source_path(
                    webshop_items, "WEBSHOP_DATA_DIR"
                ),
                "webshop_items_sha256": sha256_file(Path(webshop_items)),
                "webshop_attributes": _portable_source_path(
                    webshop_attributes, "WEBSHOP_DATA_DIR"
                ),
                "webshop_attributes_sha256": sha256_file(Path(webshop_attributes)),
                "webshop_rollout_file_count": exposure["rollout_file_count"],
                "webshop_rollout_total_bytes": exposure["rollout_total_bytes"],
                "webshop_rollout_inventory_signature": exposure[
                    "inventory_signature"
                ],
            },
            "output_sha256": output_hashes,
        }
        _write_json(staging / "manifest.json", manifest)

        for filename in (*GENERATED_FILENAMES, "manifest.json"):
            shutil.move(str(staging / filename), output_dir / filename)

    for legacy_filename in LEGACY_FILENAMES:
        legacy_path = output_dir / legacy_filename
        if legacy_path.exists():
            legacy_path.unlink()
    return manifest


def audit_fixed_val_set(
    output_dir: Path,
    expected_count: int = DEFAULT_EXPECTED_COUNT,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Fixed validation manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Fixed validation manifest has an unexpected schema version")
    if manifest.get("counts") != {
        "alfworld": expected_count,
        "webshop": expected_count,
    }:
        raise ValueError("Fixed validation manifest has an unexpected task count")
    if manifest.get("unique_tasks") != {
        "alfworld": expected_count,
        "webshop": expected_count,
    }:
        raise ValueError("Fixed validation manifest contains duplicate tasks")
    for environment in ("alfworld", "webshop"):
        audit = manifest["audits"][environment]
        overlap_fields = [key for key in audit if key.endswith("overlap_count")]
        if any(audit[key] != 0 for key in overlap_fields):
            raise ValueError(f"{environment} validation tasks overlap training data")
    for filename, expected_hash in manifest["output_sha256"].items():
        path = output_dir / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Fixed validation asset hash mismatch: {path}")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alfworld-data-root", type=Path, default=DEFAULT_ALFWORLD_DATA_ROOT)
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
    parser.add_argument("--webshop-rollout-root", action="append", type=Path)
    parser.add_argument(
        "--webshop-canonical-seed",
        type=int,
        default=DEFAULT_WEBSHOP_CANONICAL_SEED,
    )
    parser.add_argument(
        "--webshop-split-start", type=int, default=DEFAULT_WEBSHOP_EVAL_START
    )
    parser.add_argument(
        "--webshop-split-stop", type=int, default=DEFAULT_WEBSHOP_EVAL_STOP
    )
    parser.add_argument(
        "--webshop-train-start", type=int, default=DEFAULT_WEBSHOP_TRAIN_START
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.audit_only:
        manifest = audit_fixed_val_set(args.output_dir, args.expected_count)
    else:
        rollout_roots = args.webshop_rollout_root or DEFAULT_WEBSHOP_ROLLOUT_ROOTS
        rollout_paths = discover_rollout_files(rollout_roots)
        manifest = build_fixed_val_set(
            alfworld_data_root=args.alfworld_data_root,
            webshop_items=args.webshop_items,
            webshop_attributes=args.webshop_attributes,
            webshop_rollout_paths=rollout_paths,
            output_dir=args.output_dir,
            expected_count=args.expected_count,
            webshop_canonical_seed=args.webshop_canonical_seed,
            webshop_split_start=args.webshop_split_start,
            webshop_split_stop=args.webshop_split_stop,
            webshop_train_start=args.webshop_train_start,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
