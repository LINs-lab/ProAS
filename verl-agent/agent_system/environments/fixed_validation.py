from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


def default_fixed_val_root() -> Path:
    return Path(__file__).resolve().parents[2] / "recipe/graph_guide/data/fixed_val_set"


def _fixed_val_config(config: Any) -> Mapping[str, Any]:
    env_config = config.env
    fixed_config = env_config.get("fixed_val_set", {})
    return fixed_config or {}


def fixed_validation_enabled(config: Any) -> bool:
    return bool(_fixed_val_config(config).get("enabled", False))


def resolve_fixed_val_root(config: Any) -> Path:
    configured_root = _fixed_val_config(config).get("root")
    if configured_root in (None, ""):
        return default_fixed_val_root()

    root = Path(os.path.expandvars(os.path.expanduser(str(configured_root))))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    return root.resolve()


def _validate_batch_size(config: Any, task_count: int, environment: str) -> None:
    val_batch_size = int(config.data.val_batch_size)
    if val_batch_size != task_count:
        raise ValueError(
            f"Fixed {environment} validation contains {task_count} tasks, but "
            f"data.val_batch_size={val_batch_size}. Set val_batch_size={task_count} "
            "to evaluate the complete fixed set, or disable env.fixed_val_set.enabled "
            "for a special-purpose inference run."
        )


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Fixed validation asset not found: {path}")


def _load_manifest(config: Any) -> dict[str, Any]:
    manifest_path = resolve_fixed_val_root(config) / "manifest.json"
    _require_file(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed fixed validation manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Fixed validation manifest must be an object: {manifest_path}")
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_alfworld_fixed_game_files(
    config: Any,
    *,
    alfworld_data_root: str | Path | None = None,
) -> list[str] | None:
    if not fixed_validation_enabled(config):
        return None

    asset_path = resolve_fixed_val_root(config) / "alfworld_gamefiles.txt"
    _require_file(asset_path)
    relative_paths = [
        line.strip()
        for line in asset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not relative_paths:
        raise ValueError(f"Fixed ALFWorld validation asset is empty: {asset_path}")
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError(f"Fixed ALFWorld validation contains duplicate gamefiles: {asset_path}")

    _validate_batch_size(config, len(relative_paths), "ALFWorld")

    if alfworld_data_root is None:
        alfworld_root = os.environ.get("ALFWORLD_DATA")
        if not alfworld_root:
            raise EnvironmentError(
                "ALFWORLD_DATA must be set when fixed ALFWorld validation is enabled."
            )
        data_root = Path(alfworld_root) / "json_2.1.1"
    else:
        data_root = Path(alfworld_data_root)
    data_root = data_root.expanduser().resolve()

    gamefiles = [data_root / relative_path for relative_path in relative_paths]
    missing = [path for path in gamefiles if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} fixed ALFWorld gamefiles do not exist under {data_root}: {preview}"
        )
    return [str(path) for path in gamefiles]


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


def load_webshop_fixed_cases(config: Any) -> list[dict[str, Any]] | None:
    if not fixed_validation_enabled(config):
        return None

    asset_path = resolve_fixed_val_root(config) / "webshop.jsonl"
    _require_file(asset_path)
    numbered_cases = list(_iter_jsonl(asset_path))
    cases = [row for _, row in numbered_cases]
    if not cases:
        raise ValueError(f"Fixed WebShop validation asset is empty: {asset_path}")

    case_ids: set[str] = set()
    fingerprints: set[str] = set()
    required_goal_fields = {
        "asin",
        "instruction_text",
        "attributes",
        "goal_options",
        "price_upper",
    }
    for line_number, case in numbered_cases:
        case_id = case.get("case_id")
        fingerprint = case.get("exact_fingerprint")
        goal = case.get("goal")
        if case.get("environment") != "webshop":
            raise ValueError(f"Invalid WebShop environment tag at {asset_path}:{line_number}")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Missing WebShop case_id at {asset_path}:{line_number}")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(
                f"Missing WebShop exact_fingerprint at {asset_path}:{line_number}"
            )
        if case_id in case_ids or fingerprint in fingerprints:
            raise ValueError(
                f"Fixed WebShop validation contains duplicate identity at "
                f"{asset_path}:{line_number}"
            )
        if not isinstance(goal, dict):
            raise ValueError(f"Missing WebShop goal payload at {asset_path}:{line_number}")
        missing_fields = sorted(required_goal_fields - goal.keys())
        if missing_fields:
            raise ValueError(
                f"WebShop goal at {asset_path}:{line_number} is missing fields: "
                f"{', '.join(missing_fields)}"
            )
        case_ids.add(case_id)
        fingerprints.add(fingerprint)

    _validate_batch_size(config, len(cases), "WebShop")
    return cases


def load_webshop_fixed_seed(config: Any) -> int | None:
    if not fixed_validation_enabled(config):
        return None

    manifest_path = resolve_fixed_val_root(config) / "manifest.json"
    try:
        manifest = _load_manifest(config)
        seed = manifest["audits"]["webshop"]["canonical_seed"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Fixed validation manifest does not define audits.webshop.canonical_seed: "
            f"{manifest_path}"
        ) from exc
    if not isinstance(seed, int):
        raise ValueError(f"WebShop canonical_seed must be an integer: {manifest_path}")
    return seed


def validate_webshop_fixed_data_files(
    config: Any,
    items_path: str | Path,
    attributes_path: str | Path,
) -> None:
    if not fixed_validation_enabled(config):
        return

    manifest = _load_manifest(config)
    try:
        expected_hashes = {
            "items": manifest["sources"]["webshop_items_sha256"],
            "attributes": manifest["sources"]["webshop_attributes_sha256"],
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Fixed validation manifest does not define WebShop source hashes."
        ) from exc

    for label, path_value in (("items", items_path), ("attributes", attributes_path)):
        path = Path(path_value)
        _require_file(path)
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hashes[label]:
            raise ValueError(
                f"WebShop {label} file hash does not match the fixed validation manifest: "
                f"{path}"
            )


def build_fixed_case_assignments(
    cases: Sequence[dict[str, Any]],
    *,
    env_num: int,
    group_n: int,
) -> list[dict[str, Any]]:
    if env_num <= 0:
        raise ValueError("env_num must be positive")
    if group_n <= 0:
        raise ValueError("group_n must be positive")
    if len(cases) != env_num:
        raise ValueError(
            f"Fixed validation has {len(cases)} cases but env_num={env_num}; "
            "each task slot must receive exactly one fixed case."
        )
    return [case for case in cases for _ in range(group_n)]


def install_fixed_webshop_goal(env: Any, fixed_case: Mapping[str, Any]) -> str:
    case_id = str(fixed_case["case_id"])
    session_suffix = re.sub(r"[^a-zA-Z0-9]+", "-", case_id).strip("-")
    session_id = f"fixed-val-{session_suffix}"
    env.server.user_sessions[session_id] = {
        "goal": copy.deepcopy(fixed_case["goal"]),
        "done": False,
    }
    return session_id


def select_webshop_worker_seed(
    base_seed: int,
    worker_index: int,
    group_n: int,
    *,
    fixed_seed: int | None,
) -> int:
    if fixed_seed is not None:
        return fixed_seed
    return base_seed + (worker_index // group_n)
