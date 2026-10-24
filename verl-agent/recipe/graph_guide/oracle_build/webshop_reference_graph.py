from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
WEBSHOP_DIR = REPO_ROOT / "agent_system" / "environments" / "env_package" / "webshop" / "webshop"
DATA_DIR = WEBSHOP_DIR / "data"

DEFAULT_HUMAN_INS = DATA_DIR / "items_human_ins.json"
DEFAULT_ITEMS = DATA_DIR / "items_shuffle.json"


def safe_filename(value: str, limit: int = 160) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return (text or "item")[:limit]


def signature_hash(signature: Sequence[Any]) -> str:
    parts = [str(item) for item in signature]
    return sha1("\n".join(sorted(parts)).encode("utf-8")).hexdigest()[:16]


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            yield row


def truthy_number(value: Any) -> bool:
    try:
        return float(value or 0.0) > 0.0
    except Exception:
        return bool(value)


def group_trajectories(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[str(row.get("traj_uid") or f"line_{row['_line_no']}")].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item.get("traj_step_idx", 0) or 0))
    return dict(grouped)


def trajectory_success(rows: Sequence[Dict[str, Any]]) -> bool:
    return any(truthy_number(row.get("episode_rewards")) for row in rows) or any(truthy_number(row.get("won")) for row in rows)


def norm_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^a-z0-9$.'&|+:/ -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def task_stem(task_desc: Any) -> str:
    text = norm_text(task_desc)
    text = re.sub(r"^find me\s+", "", text)
    text = re.sub(r",?\s*and price lower than [0-9.]+ dollars\.?$", "", text)
    return text.strip(" .,")


def task_key(task_desc: Any) -> str:
    return sha1(task_stem(task_desc).encode("utf-8")).hexdigest()[:16]


def parse_price_upper(task_desc: Any) -> Optional[float]:
    match = re.search(r"price lower than\s+([0-9.]+)\s+dollars", str(task_desc or ""), flags=re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_task_options(task_desc: Any) -> Dict[str, str]:
    text = str(task_desc or "")
    text = re.sub(r",?\s*and price lower than [0-9.]+ dollars\.?$", "", text, flags=re.I)
    out: Dict[str, str] = {}
    for match in re.finditer(r"(?:with|and)\s+([^:,]+):\s*([^,]+?)(?=(?:,\s*and\s+[^:,]+:)|$)", text, flags=re.I):
        key = norm_text(match.group(1))
        if " with " in key:
            key = key.rsplit(" with ", 1)[-1]
        if " and " in key:
            key = key.rsplit(" and ", 1)[-1]
        value = norm_text(match.group(2))
        if key and value:
            out[key] = value
    return out


def parse_task_constraint_terms(task_desc: Any) -> List[str]:
    text = str(task_desc or "")
    text = re.sub(r"^find me\s+", "", text, flags=re.I).strip()
    text = re.sub(r",?\s*and price lower than [0-9.]+ dollars\.?$", "", text, flags=re.I)
    option_start = re.search(r"\b(?:with|and)\s+[^:,]+:\s*", text, flags=re.I)
    if option_start:
        text = text[: option_start.start()]
    pieces: List[str] = []
    for chunk in re.split(r",|\s+with\s+|\s+for\s+", text, flags=re.I):
        term = norm_text(chunk)
        term = term.strip(" .,:;")
        if len(term) < 3:
            continue
        if term in {"and", "with", "for"}:
            continue
        pieces.append(term)
    deduped: List[str] = []
    for term in pieces:
        if term not in deduped:
            deduped.append(term)
    return deduped[:12]


def parse_action(raw: Any) -> Tuple[str, str]:
    text = re.sub(r"\s+", " ", "" if raw is None else str(raw)).strip().strip("'\"").lower()
    match = re.match(r"^(search|click)\[(.*)\]$", text, flags=re.S)
    if not match:
        return "malformed", norm_text(text)
    return match.group(1), norm_text(match.group(2))


def action_from_output(row: Mapping[str, Any]) -> str:
    raw = row.get("raw_action_text")
    if raw:
        return str(raw)
    output = str(row.get("output") or "")
    match = re.search(r"<action>\s*(.*?)\s*</action>", output, flags=re.I | re.S)
    return match.group(1).strip() if match else output[-80:]


def has_cjk(value: Any) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(value or "")))


def split_sep(obs: Any) -> List[str]:
    text = "" if obs is None else str(obs)
    return [part.strip().strip("'") for part in text.split("[SEP]")]


def parse_purchased(obs: Any) -> Tuple[Optional[str], Dict[str, str], Optional[float]]:
    parts = split_sep(obs)
    lowered = [norm_text(part) for part in parts]
    asin: Optional[str] = None
    options: Dict[str, str] = {}
    score: Optional[float] = None
    for idx, part in enumerate(lowered):
        if asin is None and part == "asin" and idx + 1 < len(parts):
            cand = norm_text(parts[idx + 1]).upper()
            if re.fullmatch(r"B[0-9A-Z]{9}", cand):
                asin = cand
        if not options and part == "options" and idx + 1 < len(parts):
            raw = parts[idx + 1].strip()
            try:
                parsed = json.loads(raw)
            except Exception:
                try:
                    parsed = ast.literal_eval(raw)
                except Exception:
                    parsed = None
            if isinstance(parsed, dict):
                options = {norm_text(k): norm_text(v) for k, v in parsed.items()}
        if score is None and part == "your score (min 0.0, max 1.0)" and idx + 1 < len(parts):
            try:
                score = float(parts[idx + 1])
            except ValueError:
                score = None
    return asin, options, score


def extract_asins(text: Any) -> List[str]:
    return sorted(set(re.findall(r"\bB[0-9A-Z]{9}\b", str(text or "").upper())))


def load_human_records(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for asin, records in raw.items():
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, dict):
                out[str(asin).upper()].append(record)
    return dict(out)


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with path.open(encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk and not buffer.strip():
                break
            buffer += chunk
            if not started:
                start = buffer.find("[")
                if start < 0:
                    if not chunk:
                        break
                    buffer = buffer[-16:]
                    continue
                buffer = buffer[start + 1 :]
                started = True
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return
                try:
                    obj, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict):
                    yield obj
                buffer = buffer[end:]
            if not chunk:
                break


def product_asin(product: Mapping[str, Any]) -> str:
    return str(product.get("asin") or product.get("product_information", {}).get("ASIN") or "").upper()


def load_products_for_asins(path: Path, asins: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    wanted = {asin.upper() for asin in asins if asin}
    if not wanted or not path.exists():
        return {}
    found: Dict[str, Dict[str, Any]] = {}
    for idx, product in enumerate(iter_json_array(path), start=1):
        asin = product_asin(product)
        if asin in wanted:
            found[asin] = product
            if len(found) == len(wanted):
                break
        if idx % 100000 == 0:
            print(f"[items] scanned {idx:,}; found {len(found)}/{len(wanted)}", flush=True)
    return found


def product_price(product: Mapping[str, Any]) -> Optional[float]:
    product_info = product.get("product_information")
    if not isinstance(product_info, Mapping):
        product_info = {}
    candidates = [
        product.get("price"),
        product.get("Price"),
        product.get("pricing"),
        product_info.get("Price"),
    ]
    for value in candidates:
        if value is None:
            continue
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", str(value))
        if match:
            return float(match.group(0))
    return None


def product_name(product: Mapping[str, Any]) -> str:
    return str(product.get("name") or product.get("Title") or product.get("title") or "")


def product_attrs(product: Mapping[str, Any]) -> List[str]:
    attrs = product.get("Attributes") or product.get("attributes") or []
    if isinstance(attrs, dict):
        return [norm_text(f"{k}: {v}") for k, v in attrs.items()]
    if isinstance(attrs, list):
        return [norm_text(item) for item in attrs if str(item).strip()]
    return []


def product_constraint_text(product: Mapping[str, Any]) -> str:
    fields = [
        product_name(product),
        product.get("category"),
        product.get("query"),
        product.get("product_category"),
        " ".join(product_attrs(product)),
    ]
    return norm_text(" ".join(str(field or "") for field in fields))


def term_supported_by_product(term: str, product_text: str) -> bool:
    normalized = norm_text(term)
    if not normalized:
        return True
    if normalized in product_text:
        return True
    tokens = [token for token in re.split(r"\s+", normalized) if len(token) > 2 and token not in {"and", "the", "for", "with"}]
    return bool(tokens) and all(token in product_text for token in tokens)


def human_record_constraint_text(records: Sequence[Mapping[str, Any]]) -> str:
    pieces: List[str] = []
    for record in records:
        for key in ("instruction", "attributes", "options", "instruction_attributes", "instruction_options"):
            value = record.get(key)
            if isinstance(value, list):
                pieces.extend(str(item) for item in value)
            elif value:
                pieces.append(str(value))
    return norm_text(" ".join(pieces))


@dataclass
class TargetSpec:
    task_desc: str
    task_key: str
    target_asins: List[str] = field(default_factory=list)
    required_options: Dict[str, str] = field(default_factory=dict)
    price_upper: Optional[float] = None
    source_traj_uids: List[str] = field(default_factory=list)
    human_records: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    products: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def option_values(self) -> List[str]:
        values = [norm_text(v) for v in self.required_options.values() if norm_text(v)]
        return sorted(set(values))


def option_dict_covers(selected: Mapping[str, str], required: Mapping[str, str]) -> bool:
    if not required:
        return True
    selected_values = {norm_text(value) for value in selected.values()}
    selected_pairs = {f"{norm_text(k)}:{norm_text(v)}" for k, v in selected.items()}
    for key, value in required.items():
        nv = norm_text(value)
        if nv in selected_values:
            continue
        if f"{norm_text(key)}:{nv}" in selected_pairs:
            continue
        return False
    return True


def build_target_specs(
    grouped_by_file: Mapping[Path, Dict[str, List[Dict[str, Any]]]],
    human_records: Mapping[str, List[Dict[str, Any]]],
    products_path: Path,
) -> Dict[str, TargetSpec]:
    specs: Dict[str, TargetSpec] = {}
    for grouped in grouped_by_file.values():
        for uid, rows in grouped.items():
            if not rows:
                continue
            desc = str(rows[0].get("task_desc") or "")
            key = task_key(desc)
            specs.setdefault(
                key,
                TargetSpec(
                    task_desc=desc,
                    task_key=key,
                    required_options=parse_task_options(desc),
                    price_upper=parse_price_upper(desc),
                ),
            )
            if not trajectory_success(rows):
                continue
            for row in rows:
                asin, options, score = parse_purchased(row.get("obs_after"))
                if asin and (truthy_number(row.get("won")) or truthy_number(row.get("episode_rewards")) or (score or 0) >= 1.0):
                    spec = specs[key]
                    if asin not in spec.target_asins:
                        spec.target_asins.append(asin)
                    spec.source_traj_uids.append(uid)
                    if options:
                        spec.required_options.update(options)
    target_asins = sorted({asin for spec in specs.values() for asin in spec.target_asins})
    products = load_products_for_asins(products_path, target_asins)
    for spec in specs.values():
        for asin in list(spec.target_asins):
            if asin in human_records:
                spec.human_records[asin] = list(human_records[asin])
            if asin in products:
                spec.products[asin] = products[asin]
    return specs


def oracle_signature(stage: str, spec: TargetSpec) -> List[str]:
    pieces = [f"stage:{stage}"]
    if spec.target_asins:
        pieces.append("target_asin:" + "|".join(spec.target_asins))
    for key, value in sorted(spec.required_options.items()):
        pieces.append(f"required_option:{key}={value}")
    if spec.price_upper is not None:
        pieces.append(f"price_upper:{spec.price_upper:g}")
    return pieces


def build_constraint_checks(spec: TargetSpec) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    if spec.price_upper is not None:
        target_results = []
        for asin in spec.target_asins:
            product = spec.products.get(asin, {})
            price = product_price(product) if product else None
            target_results.append(
                {
                    "asin": asin,
                    "observed_price": price,
                    "threshold": spec.price_upper,
                    "satisfied": None if price is None else price < spec.price_upper,
                }
            )
        checks.append(
            {
                "constraint_id": f"C{len(checks)}",
                "name": "price_constraint_satisfied",
                "kind": "price",
                "depends_on": ["G2"],
                "gates": "G3",
                "spit_units": 0,
                "requested": f"price lower than {spec.price_upper:g} dollars",
                "completion": "target product price is below the requested upper bound",
                "target_results": target_results,
                "status": constraint_status(target_results),
            }
        )
    terms = parse_task_constraint_terms(spec.task_desc)
    if terms:
        target_results = []
        for asin in spec.target_asins:
            product = spec.products.get(asin, {})
            product_text = " ".join(
                [
                    product_constraint_text(product) if product else "",
                    human_record_constraint_text(spec.human_records.get(asin, [])),
                ]
            )
            matched = [term for term in terms if term_supported_by_product(term, product_text)]
            missing = [term for term in terms if term not in matched]
            target_results.append(
                {
                    "asin": asin,
                    "matched_terms": matched,
                    "missing_terms": missing,
                    "satisfied": None if not product_text.strip() else not missing,
                }
            )
        checks.append(
            {
                "constraint_id": f"C{len(checks)}",
                "name": "attribute_constraint_satisfied",
                "kind": "attribute_or_category",
                "depends_on": ["G2"],
                "gates": "G3",
                "spit_units": 0,
                "requested_terms": terms,
                "completion": "target product metadata/title/category should support the non-option request terms",
                "target_results": target_results,
                "status": constraint_status(target_results),
            }
        )
    return checks


def constraint_status(target_results: Sequence[Mapping[str, Any]]) -> str:
    values = [item.get("satisfied") for item in target_results]
    if not values or all(value is None for value in values):
        return "unknown"
    if any(value is True for value in values):
        return "satisfied_for_at_least_one_target"
    if any(item.get("matched_terms") for item in target_results):
        return "partial_metadata_match"
    return "not_verified"


def build_oracle_graph(spec: TargetSpec) -> Dict[str, Any]:
    option_steps = [
        {
            "substep_id": f"option:{key}",
            "label": f"{key} = {value}",
            "kind": "required_option",
            "option_key": key,
            "option_value": value,
            "order": "unordered_within_subgoal",
            "completion": "selected option value appears in current selected-options set",
        }
        for key, value in sorted(spec.required_options.items())
    ]
    constraint_checks = build_constraint_checks(spec)
    ordered_subgoals = [
        {
            "subgoal_id": "G0",
            "name": "target_visible",
            "order_index": 0,
            "depends_on": [],
            "ordered": True,
            "completion": "target asin appears in search results or current page",
            "unordered_substeps": [
                {
                    "substep_id": "target_asin_seen",
                    "label": "target asin observed",
                    "kind": "target_coverage",
                    "completion": "any target asin appears in observation text",
                }
            ],
            "spit_units": 1,
        },
        {
            "subgoal_id": "G1",
            "name": "target_product_opened",
            "order_index": 1,
            "depends_on": ["G0"],
            "ordered": True,
            "completion": "current page is one of the target asins",
            "unordered_substeps": [
                {
                    "substep_id": "open_target_asin",
                    "label": "click target asin",
                    "kind": "navigation",
                    "completion": "click[target asin] or current_asin is target",
                }
            ],
            "spit_units": 1,
        },
        {
            "subgoal_id": "G2",
            "name": "required_options_configured",
            "order_index": 2,
            "depends_on": ["G1"],
            "ordered": True,
            "completion": "all required option substeps are covered, order-insensitive",
            "unordered_substeps": option_steps,
            "spit_units": len(option_steps),
        },
        {
            "subgoal_id": "G3",
            "name": "target_purchased",
            "order_index": 3,
            "depends_on": [constraint_checks[-1]["constraint_id"] if constraint_checks else "G2"],
            "ordered": True,
            "completion": "click[buy now] purchases a target asin with required options covered",
            "unordered_substeps": [
                {
                    "substep_id": "buy_target",
                    "label": "buy target",
                    "kind": "terminal_action",
                    "completion": "purchased asin is target and purchased options cover all required options",
                }
            ],
            "spit_units": 1,
        },
    ]
    total_spit_units = 1 + 1 + len(option_steps) + 1
    nodes = []
    for subgoal in ordered_subgoals:
        signature = oracle_signature(subgoal["name"], spec)
        nodes.append(
            {
                "node_id": subgoal["subgoal_id"],
                "depth": subgoal["order_index"],
                "stage": subgoal["name"],
                "node_type": "ordered_subgoal",
                "signature": signature,
                "signature_hash": signature_hash(signature),
                "label": subgoal["name"],
            }
        )
    substep_nodes = []
    for subgoal in ordered_subgoals:
        for idx, substep in enumerate(subgoal["unordered_substeps"]):
            if subgoal["subgoal_id"] != "G2":
                continue
            node_id = f"{subgoal['subgoal_id']}_U{idx}"
            signature = oracle_signature(substep["substep_id"], spec)
            substep_nodes.append(
                {
                    "node_id": node_id,
                    "depth": subgoal["order_index"],
                    "stage": substep["substep_id"],
                    "node_type": "unordered_substep",
                    "parent_subgoal": subgoal["subgoal_id"],
                    "signature": signature,
                    "signature_hash": signature_hash(signature),
                    "label": substep["label"],
                }
            )
    edges = [
        ("G0", "G1", "ordered_dependency", "target must be visible before opening the product"),
        ("G1", "G2", "ordered_dependency", "target product page must be open before option coverage counts"),
    ]
    if constraint_checks:
        previous = "G2"
        for check in constraint_checks:
            edges.append(
                (
                    previous,
                    check["constraint_id"],
                    "constraint_dependency",
                    f"{check['name']} gates target purchase validity",
                )
            )
            previous = check["constraint_id"]
        edges.append((previous, "G3", "ordered_dependency", "validated target/options can be purchased"))
    else:
        edges.append(("G2", "G3", "ordered_dependency", "required options must be covered before buying"))
    constraint_nodes = []
    for check in constraint_checks:
        signature = oracle_signature(check["name"], spec)
        constraint_nodes.append(
            {
                "node_id": check["constraint_id"],
                "depth": 3,
                "stage": check["name"],
                "node_type": "case_constraint",
                "signature": signature,
                "signature_hash": signature_hash(signature),
                "label": check["constraint_id"],
                "status": check.get("status"),
            }
        )
    node_by_id = {str(node["node_id"]): node for node in nodes + constraint_nodes}
    edge_payload = [
        {
            "oracle_step_idx": idx,
            "edge_id": f"ordered_e{idx}",
            "src": src,
            "dst": dst,
            "action_kind": action_kind,
            "walkthrough_action": description,
            "from_signature": node_by_id[src]["signature"],
            "to_signature": node_by_id[dst]["signature"],
            "from_signature_hash": node_by_id[src]["signature_hash"],
            "to_signature_hash": node_by_id[dst]["signature_hash"],
        }
        for idx, (src, dst, action_kind, description) in enumerate(edges)
    ]
    for subnode in substep_nodes:
        edge_payload.append(
            {
                "oracle_step_idx": len(edge_payload),
                "edge_id": f"unordered_{subnode['node_id']}",
                "src": "G2",
                "dst": subnode["node_id"],
                "action_kind": "unordered_substep_membership",
                "walkthrough_action": "option substeps under G2 are order-insensitive",
                "from_signature": oracle_signature("required_options_configured", spec),
                "to_signature": subnode["signature"],
                "from_signature_hash": signature_hash(oracle_signature("required_options_configured", spec)),
                "to_signature_hash": subnode["signature_hash"],
            }
        )
    return {
        "task_key": spec.task_key,
        "task_desc": spec.task_desc,
        "oracle_type": "webshop_hierarchical_target_level",
        "target_asins": spec.target_asins,
        "required_options": spec.required_options,
        "required_option_values": spec.option_values(),
        "price_upper": spec.price_upper,
        "ordered_subgoals": ordered_subgoals,
        "constraint_checks": constraint_checks,
        "spit_rubric": {
            "type": "hierarchical_ordered_subgoals_with_unordered_option_bits",
            "total_units": total_spit_units,
            "units": [
                {"unit": "target_visible", "weight": 1, "ordered": True},
                {"unit": "target_product_opened", "weight": 1, "ordered": True},
                {
                    "unit": "required_options_configured",
                    "weight": len(option_steps),
                    "ordered": True,
                    "internal_order": "unordered",
                    "substeps": [step["substep_id"] for step in option_steps],
                },
                {"unit": "target_purchased", "weight": 1, "ordered": True},
            ],
            "formula": "SPIT = completed_ordered_units_plus_completed_unordered_option_bits / total_units",
        },
        "source_traj_uids": sorted(set(spec.source_traj_uids)),
        "products": {
            asin: {
                "asin": asin,
                "name": product_name(product),
                "category": product.get("category"),
                "query": product.get("query"),
                "product_category": product.get("product_category"),
                "price": product_price(product),
                "attributes": product_attrs(product),
            }
            for asin, product in spec.products.items()
        },
        "human_records": spec.human_records,
        "nodes": nodes + substep_nodes + constraint_nodes,
        "edges": edge_payload,
        "goal_signature": nodes[-1]["signature"],
    }


def target_spec_from_oracle(oracle: Mapping[str, Any], task_desc: Optional[str] = None) -> TargetSpec:
    desc = str(task_desc or oracle.get("task_desc") or "")
    parsed_options = parse_task_options(desc)
    required_options = dict(oracle.get("required_options") or {})
    perturbation = oracle.get("oracle_perturbation") or {}
    respect_oracle_options = bool(perturbation.get("respect_oracle_required_options"))
    if parsed_options and not respect_oracle_options:
        required_options = parsed_options
    return TargetSpec(
        task_desc=desc or str(oracle.get("task_desc") or ""),
        task_key=str(oracle.get("task_key") or task_key(desc)),
        target_asins=list(oracle.get("target_asins") or []),
        required_options=required_options,
        price_upper=parse_price_upper(desc) if desc else oracle.get("price_upper"),
        source_traj_uids=list(oracle.get("source_traj_uids") or []),
        human_records=deepcopy(dict(oracle.get("human_records") or {})),
        products=deepcopy(dict(oracle.get("products") or {})),
    )


@dataclass
class RolloutState:
    searched: bool = False
    target_seen: bool = False
    current_asin: Optional[str] = None
    selected_options: Dict[str, str] = field(default_factory=dict)
    purchased_asin: Optional[str] = None
    purchased_options: Dict[str, str] = field(default_factory=dict)
    purchase_score: Optional[float] = None


def option_hit_keys(state: RolloutState, spec: TargetSpec, use_purchased: bool = True) -> List[str]:
    selected = dict(state.selected_options)
    if use_purchased:
        selected.update(state.purchased_options)
    hits = []
    selected_values = {norm_text(value) for value in selected.values()}
    selected_pairs = {f"{norm_text(key)}:{norm_text(value)}" for key, value in selected.items()}
    for key, value in sorted(spec.required_options.items()):
        nv = norm_text(value)
        if nv in selected_values or f"{norm_text(key)}:{nv}" in selected_pairs:
            hits.append(key)
    return hits


def progress_detail(state: RolloutState, spec: TargetSpec) -> Dict[str, Any]:
    targets = set(spec.target_asins)
    target_visible = bool(state.target_seen or state.current_asin in targets or state.purchased_asin in targets)
    product_opened = bool(state.current_asin in targets or state.purchased_asin in targets)
    option_hits = option_hit_keys(state, spec) if product_opened else []
    options_complete = len(option_hits) == len(spec.required_options)
    target_purchased = bool(state.purchased_asin in targets and option_dict_covers(state.purchased_options, spec.required_options))
    # Ordered dependencies: terminal credit implies prior ordered subgoals are satisfied.
    if target_purchased:
        target_visible = True
        product_opened = True
        option_hits = sorted(spec.required_options)
        options_complete = True
    units = int(target_visible) + int(product_opened) + len(option_hits) + int(target_purchased)
    total = 1 + 1 + len(spec.required_options) + 1
    completed_subgoals = []
    if target_visible:
        completed_subgoals.append("G0:target_visible")
    if product_opened:
        completed_subgoals.append("G1:target_product_opened")
    if options_complete:
        completed_subgoals.append("G2:required_options_configured")
    if target_purchased:
        completed_subgoals.append("G3:target_purchased")
    return {
        "target_visible": target_visible,
        "product_opened": product_opened,
        "option_hits": option_hits,
        "options_complete": options_complete,
        "target_purchased": target_purchased,
        "completed_subgoals": completed_subgoals,
        "units": units,
        "total": total,
        "spit": units / float(total),
    }


def progress_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    subgoals = []
    substeps = []
    if not before["target_visible"] and after["target_visible"]:
        subgoals.append("G0:target_visible")
        substeps.append("target_asin_seen")
    if not before["product_opened"] and after["product_opened"]:
        subgoals.append("G1:target_product_opened")
        substeps.append("open_target_asin")
    before_options = set(before["option_hits"])
    after_options = set(after["option_hits"])
    for key in sorted(after_options - before_options):
        subgoals.append("G2:required_options_configured")
        substeps.append(f"option:{key}")
    if not before["target_purchased"] and after["target_purchased"]:
        subgoals.append("G3:target_purchased")
        substeps.append("buy_target")
    return sorted(set(subgoals)), substeps


def spit_step_regrets(annotations: Sequence[Dict[str, Any]]) -> List[float]:
    n = len(annotations)
    regrets: List[float] = []
    for idx in range(n):
        tail = annotations[idx:]
        total = sum(1.0 - float(ann.get("spit", 0.0) or 0.0) for ann in tail)
        regrets.append(total / float(len(tail)) if tail else 0.0)
    return regrets


def attach_step_regret_fields(annotations: List[Dict[str, Any]]) -> None:
    regrets = spit_step_regrets(annotations)
    hits = [int(ann.get("hit", 0) or 0) for ann in annotations]
    misses = [1 - hit for hit in hits]
    turn_future_miss = sum(misses)
    dis_indices = [idx for idx, ann in enumerate(annotations) if int(ann.get("reasonable", 0) or 0) == 0]
    mean_regret = sum(regrets[idx] for idx in dis_indices) / len(dis_indices) if dis_indices else 0.0
    dis_set = set(dis_indices)
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
        elif mean_regret <= 0:
            ann["normalized_step_regret"] = 1.0 / float(len(dis_indices)) if dis_indices else 0.0
        else:
            ann["normalized_step_regret"] = float(regrets[idx]) / mean_regret


def state_stage(state: RolloutState, spec: TargetSpec) -> int:
    detail = progress_detail(state, spec)
    if detail["target_purchased"]:
        return 4
    if detail["options_complete"] and detail["product_opened"]:
        return 3
    if detail["product_opened"]:
        return 2
    if detail["target_visible"]:
        return 1
    return 0


def state_signature(state: RolloutState, spec: TargetSpec) -> List[str]:
    stage_names = ["start_search", "target_in_results", "target_product_page", "required_options_selected", "target_purchased"]
    stage = state_stage(state, spec)
    detail = progress_detail(state, spec)
    pieces = [f"stage:{stage_names[stage]}"]
    pieces.append(f"spit_units:{detail['units']}/{detail['total']}")
    if state.searched:
        pieces.append("searched")
    if state.target_seen:
        pieces.append("target_seen")
    if state.current_asin:
        pieces.append(f"viewing:{state.current_asin}")
    for key, value in sorted(state.selected_options.items()):
        pieces.append(f"selected:{key}={value}")
    for key in detail["option_hits"]:
        pieces.append(f"covered_option:{key}")
    if state.purchased_asin:
        pieces.append(f"purchased:{state.purchased_asin}")
    return pieces


def apply_row_to_state(state: RolloutState, row: Mapping[str, Any], spec: TargetSpec) -> RolloutState:
    next_state = RolloutState(
        searched=state.searched,
        target_seen=state.target_seen,
        current_asin=state.current_asin,
        selected_options=dict(state.selected_options),
        purchased_asin=state.purchased_asin,
        purchased_options=dict(state.purchased_options),
        purchase_score=state.purchase_score,
    )
    action = action_from_output(row)
    action_type, arg = parse_action(action)
    if action_type == "search":
        next_state.searched = True
    targets = set(spec.target_asins)
    obs_after = row.get("obs_after")
    if targets & set(extract_asins(obs_after)):
        next_state.target_seen = True
    if action_type == "click":
        upper_arg = arg.upper()
        if upper_arg in targets:
            next_state.current_asin = upper_arg
            next_state.target_seen = True
        elif re.fullmatch(r"B[0-9A-Z]{9}", upper_arg):
            next_state.current_asin = upper_arg
        else:
            for key, value in spec.required_options.items():
                if norm_text(arg) == norm_text(value):
                    next_state.selected_options[key] = norm_text(value)
    purchased_asin, purchased_options, purchase_score = parse_purchased(obs_after)
    if purchased_asin:
        next_state.purchased_asin = purchased_asin
        next_state.purchased_options = purchased_options
        next_state.purchase_score = purchase_score
    return next_state


def classify_step(row: Mapping[str, Any], before: RolloutState, after: RolloutState, spec: TargetSpec) -> Dict[str, Any]:
    action = action_from_output(row)
    action_type, arg = parse_action(action)
    targets = set(spec.target_asins)
    before_detail = progress_detail(before, spec)
    after_detail = progress_detail(after, spec)
    matched_subgoals, matched_substeps = progress_delta(before_detail, after_detail)
    reasonable = 0
    reason = "no_target_progress"
    if has_cjk(action):
        reason = "non_english_or_malformed"
    elif action_type == "malformed":
        reason = "malformed_action"
    elif after_detail["units"] > before_detail["units"]:
        reasonable = 1
        reason = "target_progress"
        if matched_substeps and all(step.startswith("option:") for step in matched_substeps):
            reason = "unordered_option_progress"
    elif action_type == "search" and arg and before_detail["units"] == 0:
        reasonable = 1
        reason = "initial_search_no_spit_until_target_seen"
    elif action_type == "click" and arg.upper() in targets:
        reason = "repeat_target_product_open" if before_detail["product_opened"] else "open_target_product"
        reasonable = int(not before_detail["product_opened"])
    elif action_type == "click" and norm_text(arg) in spec.option_values():
        option_key = next((key for key, value in spec.required_options.items() if norm_text(value) == norm_text(arg)), "")
        if not before_detail["product_opened"]:
            reason = "option_before_target_product"
        elif option_key in before_detail["option_hits"]:
            reason = "repeat_required_option"
        else:
            reasonable = 1
            reason = "unordered_option_progress"
            matched_subgoals = ["G2:required_options_configured"]
            matched_substeps = [f"option:{option_key}"] if option_key else []
    elif action_type == "click" and re.fullmatch(r"B[0-9A-Z]{9}", arg.upper()) and arg.upper() not in targets:
        reason = "wrong_product_click"
    elif action_type == "click" and arg == "buy now":
        reason = "buy_wrong_target_or_options"
    elif action_type == "click" and arg in {"description", "features", "reviews", "next >", "< prev", "< prev>", "back to search"}:
        reason = "navigation_or_probe_without_target_progress"
    return {
        "reasonable": reasonable,
        "reason": reason,
        "hit": int(after_detail["units"] > before_detail["units"]),
        "matched_subgoals": matched_subgoals,
        "matched_substeps": matched_substeps,
        "progress_before": before_detail["units"],
        "progress_after": after_detail["units"],
        "progress_delta": after_detail["units"] - before_detail["units"],
        "completed_subgoals_before": before_detail["completed_subgoals"],
        "completed_subgoals_after": after_detail["completed_subgoals"],
        "covered_options_before": before_detail["option_hits"],
        "covered_options_after": after_detail["option_hits"],
        "spit_num": after_detail["units"],
        "spit_den": after_detail["total"],
        "spit": after_detail["spit"],
        "spit_before": before_detail["spit"],
        "delta_spit_num": after_detail["units"] - before_detail["units"],
        "delta_spit": after_detail["spit"] - before_detail["spit"],
        "spit_damage": 1.0 - after_detail["spit"],
        "turn_success_damage": 1.0 - after_detail["spit"],
    }


def annotate_trajectory(
    rows: Sequence[Dict[str, Any]],
    oracle: Mapping[str, Any],
    signature_mode: str = "target",
) -> Dict[str, Any]:
    task_desc = str(rows[0].get("task_desc") or oracle.get("task_desc") or "") if rows else str(oracle.get("task_desc") or "")
    spec = target_spec_from_oracle(oracle, task_desc=task_desc)
    state = RolloutState()
    annotations: List[Dict[str, Any]] = []
    for row in rows:
        before = state
        after = apply_row_to_state(before, row, spec)
        label = classify_step(row, before, after, spec)
        from_signature = state_signature(before, spec)
        to_signature = state_signature(after, spec)
        ann = {
            "traj_step_idx": int(row.get("traj_step_idx", 0) or 0),
            "task_key": spec.task_key,
            "task_desc": spec.task_desc,
            "action": action_from_output(row),
            "from_signature": from_signature,
            "to_signature": to_signature,
            "from_signature_hash": signature_hash(from_signature),
            "to_signature_hash": signature_hash(to_signature),
            "signature_mode": signature_mode,
            "purchased_asin": after.purchased_asin,
            "purchased_options": dict(after.purchased_options),
            "obs_before": row.get("obs_before"),
            "obs_after": row.get("obs_after"),
            "score": row.get("score"),
            "step_reward": row.get("step_reward"),
            "done": row.get("done"),
            "won": row.get("won"),
            **label,
        }
        annotations.append(ann)
        state = after
    attach_step_regret_fields(annotations)
    reason_counts = Counter(str(ann["reason"]) for ann in annotations)
    reasonable_counts = Counter(str(ann["reasonable"]) for ann in annotations)
    selected_reference_path = {
        "path_id": "webshop_partial_order_v1",
        "oracle_type": oracle.get("oracle_type", "webshop_hierarchical_target_level"),
        "signature_mode": signature_mode,
        "total_units": annotations[-1]["spit_den"] if annotations else 0,
    }
    return {
        "annotations": annotations,
        "final_spit": annotations[-1]["spit"] if annotations else 0.0,
        "reason_counts": dict(reason_counts),
        "reasonable_counts": dict(reasonable_counts),
        "selected_reference_path": selected_reference_path,
        "reference_path": selected_reference_path["path_id"],
    }


def merge_success_trajectory_into_oracle(
    oracle: Dict[str, Any],
    rollout_rows: Sequence[Dict[str, Any]],
    signature_mode: str = "target",
    traj_uid: str = "",
) -> bool:
    """Clean WebShop online update: merge only terminal purchase facts, not the raw action trail."""
    if not rollout_rows or not trajectory_success(rollout_rows):
        return False
    task_desc = str(rollout_rows[0].get("task_desc") or oracle.get("task_desc") or "")
    spec = target_spec_from_oracle(oracle, task_desc=task_desc)
    changed = False
    for row in rollout_rows:
        asin, options, score = parse_purchased(row.get("obs_after"))
        if not asin:
            continue
        terminal_success = (
            truthy_number(row.get("won"))
            or truthy_number(row.get("episode_rewards"))
            or (score is not None and score >= 1.0)
        )
        if not terminal_success:
            continue
        if asin not in spec.target_asins:
            spec.target_asins.append(asin)
            changed = True
        if options:
            for key, value in options.items():
                if key not in spec.required_options or spec.required_options[key] != value:
                    spec.required_options[key] = value
                    changed = True
        if traj_uid and traj_uid not in spec.source_traj_uids:
            spec.source_traj_uids.append(traj_uid)
            changed = True
    if not changed:
        return False
    rebuilt = build_oracle_graph(spec)
    oracle.clear()
    oracle.update(rebuilt)
    return True


def write_oracle_index(specs: Mapping[str, TargetSpec], output_dir: Path) -> Dict[str, Any]:
    graph_dir = output_dir / "oracle_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    index = {"oracle_type": "webshop_hierarchical_target_level", "graphs": {}}
    for key, spec in sorted(specs.items()):
        if not spec.target_asins:
            index["graphs"][key] = {"task_desc": spec.task_desc, "error": "no_successful_rollout_target"}
            continue
        oracle = build_oracle_graph(spec)
        path = graph_dir / f"{safe_filename(key + '_' + task_stem(spec.task_desc), 120)}.json"
        json_dump(path, oracle)
        index["graphs"][key] = {
            "task_desc": spec.task_desc,
            "target_asins": spec.target_asins,
            "path": str(path.relative_to(output_dir)),
            "num_edges": len(oracle["edges"]),
        }
    json_dump(graph_dir / "index.json", index)
    return index
