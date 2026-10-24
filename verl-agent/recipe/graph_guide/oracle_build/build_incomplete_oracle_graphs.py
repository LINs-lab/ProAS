from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from copy import deepcopy
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.graph_guide.oracle_build.webshop_reference_graph import (  # noqa: E402
    TargetSpec,
    build_oracle_graph,
    parse_price_upper,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def portable_repo_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def stable_score(parts: Sequence[Any], seed: int) -> str:
    text = "\n".join([str(seed), *[str(part) for part in parts]])
    return sha1(text.encode("utf-8")).hexdigest()


def choose_global_drop_set(items: Iterable[Tuple[str, str]], fraction: float, seed: int) -> Set[Tuple[str, str]]:
    unique_items = sorted(set(items))
    if not unique_items or fraction <= 0:
        return set()
    drop_count = min(len(unique_items), max(0, int(math.floor(len(unique_items) * fraction + 0.5))))
    ranked = sorted(unique_items, key=lambda item: stable_score(item, seed))
    return set(ranked[:drop_count])


def choose_local_drop_ids(ids: Sequence[str], fraction: float, seed: int, task_key: str) -> Set[str]:
    if not ids or fraction <= 0:
        return set()
    drop_count = min(len(ids), max(0, int(math.floor(len(ids) * fraction + 0.5))))
    ranked = sorted(ids, key=lambda item: stable_score([task_key, item], seed))
    return set(ranked[:drop_count])


def resolve_oracle_index(root: Path) -> Tuple[Path, Path]:
    root = root.resolve()
    candidates = [
        root / "oracle_graphs" / "index.json",
        root / "index.json",
    ]
    for index_path in candidates:
        if index_path.exists():
            graph_root = index_path.parent.parent if index_path.parent.name == "oracle_graphs" else index_path.parent
            return graph_root, index_path
    raise FileNotFoundError(f"cannot find oracle_graphs/index.json or index.json under {root}")


def graph_path_from_entry(graph_root: Path, entry: Mapping[str, Any]) -> Path:
    rel = entry.get("path")
    if not rel:
        raise KeyError("oracle index entry does not contain a path")
    return graph_root / str(rel)


def sorted_nodes(oracle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return sorted(
        [dict(node) for node in oracle.get("nodes", [])],
        key=lambda node: (int(node.get("depth", 0)), str(node.get("node_id", ""))),
    )


def make_alfworld_incomplete_oracle(
    oracle: Mapping[str, Any],
    drop_fraction: float,
    seed: int = 0,
    drop_node_ids: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = deepcopy(dict(oracle))
    nodes = sorted_nodes(out)
    task_key = str(out.get("task_key") or "")
    if len(nodes) <= 2:
        drop_ids: Set[str] = set()
    else:
        eligible = [str(node["node_id"]) for node in nodes[1:-1]]
        drop_ids = set(drop_node_ids) if drop_node_ids is not None else choose_local_drop_ids(eligible, drop_fraction, seed, task_key)
    kept_ids = {str(node["node_id"]) for node in nodes if str(node["node_id"]) not in drop_ids}

    kept_nodes = [node for node in nodes if str(node["node_id"]) in kept_ids]
    for depth, node in enumerate(kept_nodes):
        node["depth"] = depth

    original_edges = [dict(edge) for edge in out.get("edges", [])]
    kept_edges = [
        edge
        for edge in original_edges
        if str(edge.get("src")) in kept_ids and str(edge.get("dst")) in kept_ids
    ]
    kept_edge_ids = set()
    for idx, edge in enumerate(kept_edges):
        edge["oracle_step_idx"] = idx
        edge_id = str(edge.get("edge_id") or f"expert_e{idx}")
        edge["edge_id"] = edge_id
        kept_edge_ids.add(edge_id)

    out["nodes"] = kept_nodes
    out["edges"] = kept_edges
    if kept_nodes:
        out["goal_signature"] = list(kept_nodes[-1].get("signature") or [])

    if out.get("reference_paths"):
        new_refs = []
        for ref in out.get("reference_paths") or []:
            new_ref = deepcopy(ref)
            new_ref["node_ids"] = [str(node_id) for node_id in ref.get("node_ids", []) if str(node_id) in kept_ids]
            new_ref["edge_ids"] = [str(edge_id) for edge_id in ref.get("edge_ids", []) if str(edge_id) in kept_edge_ids]
            if new_ref["node_ids"]:
                new_refs.append(new_ref)
        out["reference_paths"] = new_refs

    out["oracle_perturbation"] = {
        "kind": "incomplete",
        "environment": "alfworld",
        "drop_fraction": drop_fraction,
        "seed": seed,
        "dropped_node_ids": sorted(drop_ids),
        "dropped_units": len(drop_ids),
        "original_nodes": len(nodes),
        "remaining_nodes": len(kept_nodes),
        "original_edges": len(original_edges),
        "remaining_edges": len(kept_edges),
    }
    stats = {
        "eligible_units": max(0, len(nodes) - 2),
        "dropped_units": len(drop_ids),
        "remaining_units": max(0, len(kept_nodes) - 2),
        "original_nodes": len(nodes),
        "remaining_nodes": len(kept_nodes),
        "original_edges": len(original_edges),
        "remaining_edges": len(kept_edges),
    }
    return out, stats


def make_webshop_incomplete_oracle(
    oracle: Mapping[str, Any],
    drop_fraction: float,
    seed: int = 0,
    drop_option_keys: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    task_key = str(oracle.get("task_key") or "")
    required_options = dict(oracle.get("required_options") or {})
    eligible = sorted(str(key) for key in required_options)
    if drop_option_keys is None:
        drop_keys = choose_local_drop_ids(eligible, drop_fraction, seed, task_key)
    else:
        drop_keys = set(drop_option_keys)
    kept_options = {
        key: value
        for key, value in required_options.items()
        if str(key) not in drop_keys
    }

    spec = TargetSpec(
        task_desc=str(oracle.get("task_desc") or ""),
        task_key=task_key,
        target_asins=list(oracle.get("target_asins") or []),
        required_options=kept_options,
        price_upper=parse_price_upper(str(oracle.get("task_desc") or "")) if oracle.get("task_desc") else oracle.get("price_upper"),
        source_traj_uids=list(oracle.get("source_traj_uids") or []),
        human_records=deepcopy(dict(oracle.get("human_records") or {})),
        products=deepcopy(dict(oracle.get("products") or {})),
    )
    out = build_oracle_graph(spec)
    out["oracle_perturbation"] = {
        "kind": "incomplete",
        "environment": "webshop",
        "drop_fraction": drop_fraction,
        "seed": seed,
        "respect_oracle_required_options": True,
        "dropped_required_options": {
            key: required_options[key]
            for key in sorted(drop_keys)
            if key in required_options
        },
        "dropped_units": len(drop_keys),
        "original_required_options": len(required_options),
        "remaining_required_options": len(kept_options),
    }
    stats = {
        "eligible_units": len(required_options),
        "dropped_units": len(drop_keys),
        "remaining_units": len(kept_options),
        "original_required_options": len(required_options),
        "remaining_required_options": len(kept_options),
        "original_edges": len(oracle.get("edges", []) or []),
        "remaining_edges": len(out.get("edges", []) or []),
    }
    return out, stats


def collect_global_units(env: str, graph_root: Path, index: Mapping[str, Any]) -> List[Tuple[str, str]]:
    units: List[Tuple[str, str]] = []
    for task_key, entry in sorted((index.get("graphs") or {}).items()):
        if entry.get("error"):
            continue
        graph = read_json(graph_path_from_entry(graph_root, entry))
        if env == "alfworld":
            nodes = sorted_nodes(graph)
            units.extend((str(task_key), str(node["node_id"])) for node in nodes[1:-1])
        elif env == "webshop":
            required = graph.get("required_options") or {}
            units.extend((str(task_key), str(key)) for key in sorted(required))
        else:
            raise ValueError(f"unsupported env={env!r}")
    return units


def build_incomplete_oracles(
    env: str,
    input_root: Path,
    output_root: Path,
    drop_fraction: float,
    seed: int = 0,
    overwrite: bool = False,
) -> Dict[str, Any]:
    graph_root, index_path = resolve_oracle_index(input_root)
    index = read_json(index_path)
    output_root = output_root.resolve()
    output_graph_dir = output_root / "oracle_graphs"
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_graph_dir.mkdir(parents=True, exist_ok=True)

    global_drop_set = choose_global_drop_set(
        collect_global_units(env, graph_root, index),
        fraction=drop_fraction,
        seed=seed,
    )

    new_index = deepcopy(index)
    new_index["oracle_perturbation"] = {
        "kind": "incomplete",
        "environment": env,
        "drop_fraction": drop_fraction,
        "seed": seed,
        "source_index": portable_repo_path(index_path),
    }
    summary: Dict[str, Any] = {
        "environment": env,
        "drop_fraction": drop_fraction,
        "seed": seed,
        "source_index": portable_repo_path(index_path),
        "output_index": portable_repo_path(output_graph_dir / "index.json"),
        "num_graphs": 0,
        "eligible_units": 0,
        "dropped_units": 0,
        "remaining_units": 0,
        "graphs": {},
    }

    for task_key, entry in sorted((index.get("graphs") or {}).items()):
        if entry.get("error"):
            continue
        oracle = read_json(graph_path_from_entry(graph_root, entry))
        if env == "alfworld":
            per_graph_drop = {unit for key, unit in global_drop_set if key == str(task_key)}
            new_oracle, stats = make_alfworld_incomplete_oracle(
                oracle,
                drop_fraction=drop_fraction,
                seed=seed,
                drop_node_ids=per_graph_drop,
            )
        elif env == "webshop":
            per_graph_drop = {unit for key, unit in global_drop_set if key == str(task_key)}
            new_oracle, stats = make_webshop_incomplete_oracle(
                oracle,
                drop_fraction=drop_fraction,
                seed=seed,
                drop_option_keys=per_graph_drop,
            )
        else:
            raise ValueError(f"unsupported env={env!r}")

        rel_path = Path(str(entry["path"]))
        out_path = output_root / rel_path
        write_json(out_path, new_oracle)
        new_entry = deepcopy(entry)
        new_entry["path"] = str(rel_path)
        new_entry["num_edges"] = len(new_oracle.get("edges", []) or [])
        new_entry["num_nodes"] = len(new_oracle.get("nodes", []) or [])
        new_entry["oracle_perturbation"] = {
            "kind": "incomplete",
            "drop_fraction": drop_fraction,
            "seed": seed,
            "dropped_units": stats["dropped_units"],
            "eligible_units": stats["eligible_units"],
        }
        new_index["graphs"][task_key] = new_entry

        summary["num_graphs"] += 1
        summary["eligible_units"] += int(stats["eligible_units"])
        summary["dropped_units"] += int(stats["dropped_units"])
        summary["remaining_units"] += int(stats["remaining_units"])
        summary["graphs"][task_key] = stats

    summary["realized_drop_fraction"] = (
        float(summary["dropped_units"]) / float(summary["eligible_units"])
        if summary["eligible_units"]
        else 0.0
    )
    write_json(output_graph_dir / "index.json", new_index)
    write_json(output_root / "incomplete_oracle_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build incomplete ASFD oracle graphs by dropping progress units.")
    parser.add_argument("--env", choices=["alfworld", "webshop"], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--drop-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.drop_fraction) <= 1.0:
        raise ValueError("--drop-fraction must be in [0, 1]")
    summary = build_incomplete_oracles(
        env=args.env,
        input_root=args.input_root,
        output_root=args.output_root,
        drop_fraction=float(args.drop_fraction),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
    )
    print(
        "[incomplete-oracle] "
        f"env={summary['environment']} graphs={summary['num_graphs']} "
        f"dropped={summary['dropped_units']}/{summary['eligible_units']} "
        f"realized={summary['realized_drop_fraction']:.3f} "
        f"index={summary['output_index']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
