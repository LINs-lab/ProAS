"""Build stable WebShop goals and audit historical training exposure."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


PRICE_SUFFIX = re.compile(
    r",?\s*and price lower than [0-9.]+ dollars\.?$",
    flags=re.IGNORECASE,
)
SOURCE_STEP_PATTERN = re.compile(rb'"traj_step_idx"\s*:\s*0(?:\s*[,}])')
PRICE_RANGE = [10.0 * index for index in range(1, 100)]


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower().replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9$.'&|+:/ -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _task_text(task_desc: str, include_price: bool) -> str:
    text = normalize_text(task_desc)
    if not include_price:
        text = PRICE_SUFFIX.sub("", text).strip(" ., ")
    text = re.sub(r"^find me\s+", "", text)
    return text.strip(" ., ")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def webshop_base_task_key(task_desc: str) -> str:
    return _sha256_text(_task_text(task_desc, include_price=False))


def webshop_exact_task_key(task_desc: str) -> str:
    return _sha256_text(_task_text(task_desc, include_price=True))


def _normalize_options(raw: Any) -> dict[str, str]:
    if isinstance(raw, Mapping):
        return {
            normalize_text(key): normalize_text(value)
            for key, value in raw.items()
            if normalize_text(key) and normalize_text(value)
        }
    if not isinstance(raw, list):
        return {}

    options: dict[str, str] = {}
    for index, item in enumerate(raw):
        text = str(item or "").strip()
        if not text:
            continue
        if ":" in text:
            key, value = text.split(":", 1)
        else:
            key, value = f"option_{index}", text
        if normalize_text(key) and normalize_text(value):
            options[normalize_text(key)] = normalize_text(value)
    return options


def canonical_goal_payload(goal: Mapping[str, Any]) -> dict[str, Any]:
    attributes = sorted(
        {
            normalize_text(attribute)
            for attribute in goal.get("attributes", []) or []
            if normalize_text(attribute)
        }
    )
    return {
        "asin": normalize_text(goal.get("asin")),
        "attributes": attributes,
        "category": normalize_text(goal.get("category")),
        "goal_options": dict(sorted(_normalize_options(goal.get("goal_options")).items())),
        "instruction_text": normalize_text(goal.get("instruction_text")),
        "name": normalize_text(goal.get("name")),
        "price_upper": float(goal.get("price_upper", 1_000_000.0)),
        "product_category": normalize_text(goal.get("product_category")),
        "query": normalize_text(goal.get("query")),
    }


def webshop_exact_fingerprint(goal: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        canonical_goal_payload(goal),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256_text(encoded)


def _load_json(path: Path, expected_type: type) -> Any:
    if not path.is_file():
        raise ValueError(f"WebShop data file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid WebShop JSON data: {path}") from exc
    if not isinstance(payload, expected_type):
        raise ValueError(
            f"WebShop data must be {expected_type.__name__}: {path}"
        )
    return payload


def _product_options(product: Mapping[str, Any]) -> dict[str, list[str]]:
    raw_options = product.get("customization_options") or {}
    if not isinstance(raw_options, Mapping):
        return {}
    options: dict[str, list[str]] = {}
    for raw_name, raw_values in raw_options.items():
        if not isinstance(raw_values, list):
            continue
        values = []
        for item in raw_values:
            if not isinstance(item, Mapping):
                continue
            value = normalize_text(str(item.get("value") or "").replace("/", " | "))
            if value:
                values.append(value)
        if values:
            options[normalize_text(raw_name)] = list(dict.fromkeys(values))
    return dict(sorted(options.items()))


def _goal_from_product(
    product: Mapping[str, Any],
    instruction: str,
    attributes: Iterable[Any],
    goal_options: Mapping[str, str],
    price_upper: float = 1_000_000.0,
) -> dict[str, Any]:
    return {
        "asin": str(product.get("asin") or "").strip(),
        "attributes": [str(value) for value in attributes if str(value).strip()],
        "category": product.get("category"),
        "goal_options": dict(goal_options),
        "instruction_text": instruction.strip().rstrip("."),
        "name": product.get("name") or product.get("Title"),
        "price_upper": float(price_upper),
        "product_category": product.get("product_category"),
        "query": product.get("query"),
        "weight": 1.0,
    }


def _candidate(goal: dict[str, Any], source_type: str) -> dict[str, Any]:
    instruction = str(goal["instruction_text"])
    return {
        "base_task_key": webshop_base_task_key(instruction),
        "exact_fingerprint": webshop_exact_fingerprint(goal),
        "goal": goal,
        "source_type": source_type,
    }


def _synthetic_candidates(
    products: list[dict[str, Any]],
    attributes: Mapping[str, Any],
    price_uppers: Mapping[str, float] | None = None,
) -> Iterable[dict[str, Any]]:
    for product in products:
        asin = str(product.get("asin") or "")
        attribute_record = attributes.get(asin) or {}
        if not isinstance(attribute_record, Mapping):
            continue
        instruction = str(attribute_record.get("instruction") or "").strip()
        if not instruction:
            continue
        price_upper = (
            float(price_uppers[asin])
            if price_uppers is not None and asin in price_uppers
            else 1_000_000.0
        )
        price_text = (
            f", and price lower than {price_upper:.2f} dollars"
            if price_upper < 1_000_000.0
            else ""
        )
        option_values = _product_options(product)
        option_names = list(option_values)
        combinations = itertools.product(
            *(option_values[name] for name in option_names)
        )
        for combination in combinations:
            goal_options = dict(zip(option_names, combination, strict=True))
            option_text = ", and ".join(
                f"{name}: {value}" for name, value in goal_options.items()
            )
            task_stem = (
                f"{instruction.rstrip('.')} with {option_text}"
                if option_text
                else instruction.rstrip(".")
            )
            task_desc = f"{task_stem}{price_text}"
            goal = _goal_from_product(
                product,
                task_desc,
                attribute_record.get("instruction_attributes") or [],
                goal_options,
                price_upper=price_upper,
            )
            yield _candidate(goal, "synthetic")


def _human_candidates(
    products: list[dict[str, Any]],
    human_instructions: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    for product in products:
        asin = str(product.get("asin") or "")
        records = human_instructions.get(asin) or []
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            instruction = str(record.get("instruction") or "").strip()
            if not instruction:
                continue
            goal = _goal_from_product(
                product,
                instruction,
                record.get("instruction_attributes") or record.get("attributes") or [],
                _normalize_options(
                    record.get("instruction_options") or record.get("options")
                ),
            )
            yield _candidate(goal, "human")


def build_webshop_candidates(
    items_path: Path,
    attributes_path: Path,
    human_instructions_path: Path,
) -> list[dict[str, Any]]:
    products = _load_json(items_path, list)
    attributes = _load_json(attributes_path, dict)
    human_instructions = _load_json(human_instructions_path, dict)

    unique: dict[str, dict[str, Any]] = {}
    for candidate in itertools.chain(
        _synthetic_candidates(products, attributes),
        _human_candidates(products, human_instructions),
    ):
        unique.setdefault(candidate["base_task_key"], candidate)
    return list(unique.values())


def _eligible_products(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = []
    seen_asins: set[str] = set()
    for product in products:
        asin = str(product.get("asin") or "")
        if not asin or asin == "nan" or len(asin) > 10 or asin in seen_asins:
            continue
        seen_asins.add(asin)
        eligible.append(product)
    return eligible


def _pricing_values(product: Mapping[str, Any]) -> list[float]:
    pricing = product.get("pricing")
    if not pricing:
        return [100.0]
    if isinstance(pricing, list):
        values = [float(value) for value in pricing]
    else:
        values = [
            float(value)
            for value in re.findall(r"\$\s*([0-9]+(?:\.[0-9]+)?)", str(pricing))
        ]
    return values[:2] or [100.0]


def _generate_product_prices(
    products: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, float]:
    prices = {}
    for product in products:
        asin = str(product["asin"])
        pricing = _pricing_values(product)
        if len(pricing) == 1:
            prices[asin] = pricing[0]
        else:
            prices[asin] = rng.uniform(*pricing[:2])
    return prices


def build_webshop_candidate_catalog(
    items_path: Path,
    attributes_path: Path,
    seed: int = 42,
) -> list[dict[str, Any]]:
    products = _eligible_products(_load_json(items_path, list))
    attributes = _load_json(attributes_path, dict)
    rng = random.Random(seed)
    product_prices = _generate_product_prices(products, rng)
    price_uppers: dict[str, float] = {}
    for product in products:
        asin = str(product.get("asin") or "")
        price_range = [
            price for price in PRICE_RANGE if price > product_prices[asin]
        ][:4]
        if len(price_range) >= 2:
            _, price_upper = sorted(rng.sample(price_range, 2))
            price_uppers[asin] = price_upper
        else:
            price_uppers[asin] = 1_000_000.0
    return list(_synthetic_candidates(products, attributes, price_uppers))


def canonical_webshop_split(
    candidates: Iterable[dict[str, Any]],
    seed: int,
    start: int,
    stop: int,
) -> list[dict[str, Any]]:
    if start < 0 or stop <= start:
        raise ValueError("WebShop canonical split bounds are invalid")
    catalog = list(candidates)
    random.Random(seed).shuffle(catalog)
    if stop > len(catalog):
        raise ValueError(
            f"WebShop canonical split [{start}, {stop}) exceeds "
            f"catalog size {len(catalog)}"
        )
    unique: dict[str, dict[str, Any]] = {}
    for candidate in catalog[start:stop]:
        unique.setdefault(candidate["base_task_key"], candidate)
    return list(unique.values())


def rollout_inventory(paths: Iterable[Path]) -> dict[str, Any]:
    entries = []
    for raw_path in sorted({Path(path).resolve() for path in paths}):
        if not raw_path.is_file():
            raise ValueError(f"WebShop rollout file does not exist: {raw_path}")
        stat = raw_path.stat()
        entries.append(
            {
                "path": str(raw_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "entries": entries,
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "signature": _sha256_text(encoded),
    }


def scan_training_exposure(paths: Iterable[Path]) -> dict[str, Any]:
    inventory = rollout_inventory(paths)
    base_task_keys: set[str] = set()
    source_trajectories: set[tuple[str, str]] = set()

    for entry in inventory["entries"]:
        path = Path(entry["path"])
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                if not SOURCE_STEP_PATTERN.search(line):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed WebShop source trajectory at "
                        f"{path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict) or row.get("traj_step_idx") != 0:
                    raise ValueError(
                        f"Malformed WebShop source trajectory at {path}:{line_number}"
                    )
                traj_uid = row.get("traj_uid")
                task_desc = row.get("task_desc")
                if not isinstance(traj_uid, str) or not traj_uid.strip():
                    raise ValueError(
                        f"Malformed WebShop source trajectory at {path}:{line_number}: "
                        "missing traj_uid"
                    )
                if not isinstance(task_desc, str) or not task_desc.strip():
                    raise ValueError(
                        f"Malformed WebShop source trajectory at {path}:{line_number}: "
                        "missing task_desc"
                    )
                identity = (str(path), traj_uid)
                if identity in source_trajectories:
                    continue
                source_trajectories.add(identity)
                base_task_keys.add(webshop_base_task_key(task_desc))

    return {
        "schema_version": 1,
        "inventory_signature": inventory["signature"],
        "rollout_file_count": inventory["file_count"],
        "rollout_total_bytes": inventory["total_bytes"],
        "source_trajectory_count": len(source_trajectories),
        "unique_base_task_count": len(base_task_keys),
        "base_task_keys": sorted(base_task_keys),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_or_scan_training_exposure(
    paths: Iterable[Path],
    cache_path: Path,
) -> dict[str, Any]:
    path_list = list(paths)
    inventory = rollout_inventory(path_list)
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("inventory_signature") == inventory["signature"]
        ):
            result = dict(cached)
            result["cache_hit"] = True
            return result

    result = scan_training_exposure(path_list)
    _write_json_atomic(cache_path, result)
    result["cache_hit"] = False
    return result
