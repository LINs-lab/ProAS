from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.graph_guide.oracle_build import webshop_reference_graph as ref_graph


DEFAULT_ATTR = ref_graph.DATA_DIR / "items_ins_v2_1000.json"


def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def compact_product(product: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "asin": ref_graph.product_asin(product),
        "name": product.get("name") or product.get("Title") or product.get("title"),
        "Title": product.get("Title") or product.get("name") or product.get("title"),
        "category": product.get("category"),
        "query": product.get("query"),
        "product_category": product.get("product_category"),
        "price": ref_graph.product_price(product),
        "pricing": product.get("pricing"),
        "Attributes": product.get("Attributes") or product.get("attributes") or [],
    }


def option_map_from_human_record(record: Mapping[str, Any]) -> Dict[str, str]:
    raw = record.get("instruction_options") or record.get("options") or {}
    if isinstance(raw, dict):
        return {
            ref_graph.norm_text(key): ref_graph.norm_text(value)
            for key, value in raw.items()
            if ref_graph.norm_text(value)
        }
    out: Dict[str, str] = {}
    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            text = str(item or "").strip()
            if not text:
                continue
            if ":" in text:
                key, value = text.split(":", 1)
                out[ref_graph.norm_text(key)] = ref_graph.norm_text(value)
            else:
                out[f"option_{idx}"] = ref_graph.norm_text(text)
    return out


def product_customization_options(product: Mapping[str, Any]) -> Dict[str, list[str]]:
    raw = product.get("customization_options") or {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, list[str]] = {}
    for option_name, option_contents in sorted(raw.items()):
        if not option_contents:
            continue
        values = []
        for item in option_contents:
            if not isinstance(item, Mapping):
                continue
            value = str(item.get("value") or "").strip().replace("/", " | ").lower()
            if value:
                values.append(ref_graph.norm_text(value))
        if values:
            out[ref_graph.norm_text(option_name)] = sorted(dict.fromkeys(values))
    return out


def merge_spec(dst: Dict[str, ref_graph.TargetSpec], spec: ref_graph.TargetSpec) -> None:
    existing = dst.get(spec.task_key)
    if existing is None:
        dst[spec.task_key] = spec
        return
    for asin in spec.target_asins:
        if asin not in existing.target_asins:
            existing.target_asins.append(asin)
    for key, value in spec.required_options.items():
        existing.required_options.setdefault(key, value)
    for uid in spec.source_traj_uids:
        if uid not in existing.source_traj_uids:
            existing.source_traj_uids.append(uid)
    existing.human_records.update(spec.human_records)
    existing.products.update(spec.products)
    if existing.price_upper is None:
        existing.price_upper = spec.price_upper


def specs_from_dataset(
    items_path: Path,
    human_records: Mapping[str, list[Dict[str, Any]]],
    attr_records: Mapping[str, Any],
    goal_mode: str,
    limit_products: Optional[int],
    max_goals: Optional[int],
) -> Iterable[ref_graph.TargetSpec]:
    emitted = 0
    for product_idx, product in enumerate(ref_graph.iter_json_array(items_path), start=1):
        if limit_products is not None and product_idx > limit_products:
            break
        asin = ref_graph.product_asin(product)
        if not asin:
            continue
        product_summary = compact_product(product)
        if goal_mode in {"human", "both"}:
            for record_idx, record in enumerate(human_records.get(asin, [])):
                desc = str(record.get("instruction") or "").strip().rstrip(".")
                if not desc:
                    continue
                product_for_spec = dict(product_summary)
                product_for_spec["Attributes"] = (
                    record.get("attributes")
                    or record.get("instruction_attributes")
                    or product_summary.get("Attributes")
                    or []
                )
                spec = ref_graph.TargetSpec(
                    task_desc=desc,
                    task_key=ref_graph.task_key(desc),
                    target_asins=[asin],
                    required_options=option_map_from_human_record(record),
                    price_upper=None,
                    source_traj_uids=[f"human:{asin}:{record_idx}"],
                    human_records={asin: [dict(record)]},
                    products={asin: product_for_spec},
                )
                yield spec
                emitted += 1
                if max_goals is not None and emitted >= max_goals:
                    return
        if goal_mode in {"synthetic", "both"} and asin in attr_records:
            attr = attr_records.get(asin) or {}
            instruction = str(attr.get("instruction") or "").strip()
            if not instruction:
                continue
            product_for_spec = dict(product_summary)
            product_for_spec["Attributes"] = attr.get("attributes") or product_summary.get("Attributes") or []
            option_values = product_customization_options(product)
            option_names = sorted(option_values)
            combinations = itertools.product(*(option_values[name] for name in option_names))
            for combo_idx, combination in enumerate(combinations):
                required_options = {
                    option_names[idx]: value
                    for idx, value in enumerate(combination)
                }
                option_text = ", and ".join(f"{key}: {value}" for key, value in required_options.items())
                desc = f"{instruction} with {option_text}" if option_text else instruction
                spec = ref_graph.TargetSpec(
                    task_desc=desc,
                    task_key=ref_graph.task_key(desc),
                    target_asins=[asin],
                    required_options=required_options,
                    price_upper=None,
                    source_traj_uids=[f"synthetic:{asin}:{combo_idx}"],
                    human_records={asin: list(human_records.get(asin, []))},
                    products={asin: product_for_spec},
                )
                yield spec
                emitted += 1
                if max_goals is not None and emitted >= max_goals:
                    return
        if product_idx % 10000 == 0:
            print(f"[dataset] scanned {product_idx:,} products; emitted {emitted:,} specs", flush=True)


def specs_from_rollouts(
    rollout_paths: Iterable[Path],
    human_records: Mapping[str, list[Dict[str, Any]]],
    items_path: Path,
) -> Dict[str, ref_graph.TargetSpec]:
    grouped_by_file = {path: ref_graph.group_trajectories(path) for path in rollout_paths if path.exists()}
    if not grouped_by_file:
        return {}
    return ref_graph.build_target_specs(grouped_by_file, human_records, items_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WebShop target-level oracle graphs.")
    parser.add_argument("--rollout", action="append", type=Path, default=None)
    parser.add_argument("--human-ins", type=Path, default=ref_graph.DEFAULT_HUMAN_INS)
    parser.add_argument("--items", type=Path, default=ref_graph.DEFAULT_ITEMS)
    parser.add_argument("--attr", type=Path, default=DEFAULT_ATTR)
    parser.add_argument(
        "--goal-mode",
        choices=["human", "synthetic", "both", "rollout_only"],
        default="human",
        help="Which dataset goals to materialize. rollout_only uses only successful rollout purchases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recipe/graph_guide/oracle_build/results/oracle_graphs_webshop_train"),
    )
    parser.add_argument("--limit-products", type=int, default=None)
    parser.add_argument("--max-goals", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    print(f"[human] loading {args.human_ins}", flush=True)
    human_records = ref_graph.load_human_records(args.human_ins)
    attr_records = load_json_dict(args.attr)
    specs: Dict[str, ref_graph.TargetSpec] = {}

    if args.goal_mode != "rollout_only":
        print(
            f"[dataset] items={args.items} attr={args.attr} goal_mode={args.goal_mode}",
            flush=True,
        )
        for spec in specs_from_dataset(
            args.items,
            human_records,
            attr_records,
            args.goal_mode,
            args.limit_products,
            args.max_goals,
        ):
            merge_spec(specs, spec)

    if args.rollout:
        print(f"[rollout] adding successful purchases from {len(args.rollout)} files", flush=True)
        for spec in specs_from_rollouts(args.rollout, human_records, args.items).values():
            merge_spec(specs, spec)

    print(f"[oracle] writing {len(specs):,} task specs to {output_dir}", flush=True)
    index = ref_graph.write_oracle_index(specs, output_dir)
    built = sum(1 for item in index.get("graphs", {}).values() if "path" in item)
    failed = sum(1 for item in index.get("graphs", {}).values() if "error" in item)
    print(
        json.dumps(
            {
                "index": str(output_dir / "oracle_graphs" / "index.json"),
                "requested": len(specs),
                "built": built,
                "failed": failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
