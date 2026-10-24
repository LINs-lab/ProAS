from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.graph_guide.oracle_build import webshop_reference_graph as ref_graph  # noqa: E402


DEFAULT_TEACHER_MODEL = (
    REPO_ROOT
    / "official_outputs"
    / "7B"
    / "webshop"
    / "grpo_webshop"
    / "grpo_webshop"
    / "global_step_140"
    / "actor"
)
SOURCE = "teacher_7b_grpo_success_target"


def portable_repo_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(payload)
    return rows


def group_trajectories(rows: Iterable[Mapping[str, Any]]) -> Dict[str, list[Dict[str, Any]]]:
    grouped: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for line_number, row in enumerate(rows, start=1):
        uid = str(row.get("traj_uid") or "")
        if not uid:
            raise ValueError(f"Missing traj_uid in inference row {line_number}")
        grouped[uid].append(dict(row))
    for trajectory in grouped.values():
        trajectory.sort(key=lambda item: int(item.get("traj_step_idx") or 0))
    return dict(grouped)


def load_schedule_cases(path: Path) -> list[Dict[str, Any]]:
    cases = []
    for batch in read_jsonl(path):
        tasks = batch.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError(f"Schedule batch is missing tasks: {path}")
        cases.extend(dict(task) for task in tasks)
    return cases


def schedule_audit(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    schedule_cases: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected_case_ids = Counter(str(case.get("case_id") or "") for case in schedule_cases)
    inferred_case_ids = Counter()
    inferred_task_descs = Counter()
    for trajectory in grouped.values():
        if not trajectory:
            continue
        first = trajectory[0]
        case_id = str(first.get("fixed_train_case_id") or "")
        if case_id:
            inferred_case_ids[case_id] += 1
        inferred_task_descs[str(first.get("task_desc") or "")] += 1

    expected_task_descs = Counter(str(case.get("task_desc") or "") for case in schedule_cases)
    case_id_metadata_available = bool(inferred_case_ids)
    expected = expected_case_ids if case_id_metadata_available else expected_task_descs
    inferred = inferred_case_ids if case_id_metadata_available else inferred_task_descs
    missing = expected - inferred
    unexpected = inferred - expected
    return {
        "schedule_entries": len(schedule_cases),
        "inferred_trajectories": len(grouped),
        "case_id_metadata_available": case_id_metadata_available,
        "schedule_multiset_match": not missing and not unexpected,
        "missing_entries": sum(missing.values()),
        "unexpected_entries": sum(unexpected.values()),
        "missing_preview": list(missing.items())[:10],
        "unexpected_preview": list(unexpected.items())[:10],
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare_output(output_root: Path, overwrite: bool) -> Path:
    index_path = output_root / "oracle_graphs" / "index.json"
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    if index_path.exists():
        raise FileExistsError(f"Oracle index already exists: {index_path}")
    graph_dir = output_root / "oracle_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    return graph_dir


def build_oracles(
    *,
    input_path: Path,
    output_root: Path,
    successful_output: Path,
    schedule_path: Path,
    teacher_model: str,
    seed: int,
    items_path: Path = ref_graph.DEFAULT_ITEMS,
    human_ins_path: Path = ref_graph.DEFAULT_HUMAN_INS,
    overwrite: bool = False,
    allow_incomplete_schedule: bool = False,
) -> Dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    successful_output = successful_output.expanduser().resolve()
    schedule_path = schedule_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not schedule_path.is_file():
        raise FileNotFoundError(schedule_path)
    if successful_output.exists() and not overwrite:
        raise FileExistsError(f"Successful trajectory file already exists: {successful_output}")

    rows = read_jsonl(input_path)
    grouped = group_trajectories(rows)
    audit = schedule_audit(grouped, load_schedule_cases(schedule_path))
    if not audit["schedule_multiset_match"] and not allow_incomplete_schedule:
        raise ValueError(
            "Teacher inference does not exactly cover the fixed training schedule: "
            f"missing={audit['missing_entries']} unexpected={audit['unexpected_entries']}"
        )

    human_records = ref_graph.load_human_records(human_ins_path)
    specs = ref_graph.build_target_specs({input_path: grouped}, human_records, items_path)
    specs = {key: spec for key, spec in specs.items() if spec.target_asins}
    successful_uids = {
        uid
        for spec in specs.values()
        for uid in spec.source_traj_uids
    }
    successful_rows = [row for row in rows if str(row.get("traj_uid") or "") in successful_uids]

    graph_dir = prepare_output(output_root, overwrite=overwrite)
    index: Dict[str, Any] = {
        "env_adapter": "webshop",
        "source": SOURCE,
        "signature_mode": "target",
        "input_path": portable_repo_path(input_path),
        "successful_trajectory_path": portable_repo_path(successful_output),
        "schedule_path": portable_repo_path(schedule_path),
        "teacher_model": portable_repo_path(teacher_model),
        "seed": seed,
        "graphs": {},
    }
    for task_key, spec in sorted(specs.items()):
        oracle = ref_graph.build_oracle_graph(spec)
        oracle["oracle_type"] = "webshop_teacher_distilled_target_level"
        oracle["source"] = SOURCE
        oracle["state_source"] = SOURCE
        oracle["signature_mode"] = "target"
        oracle["teacher_model"] = portable_repo_path(teacher_model)
        oracle["teacher_traj_uids"] = list(spec.source_traj_uids)
        oracle["distillation_input"] = portable_repo_path(input_path)
        oracle["fixed_train_seed"] = seed
        filename = f"{ref_graph.safe_filename(task_key)}.json"
        graph_path = graph_dir / filename
        write_json_atomic(graph_path, oracle)
        index["graphs"][task_key] = {
            "path": str(Path("oracle_graphs") / filename),
            "oracle_type": oracle["oracle_type"],
            "source": SOURCE,
            "num_nodes": len(oracle.get("nodes", [])),
            "num_edges": len(oracle.get("edges", [])),
            "num_teacher_trajectories": len(spec.source_traj_uids),
            "target_asins": list(spec.target_asins),
            "task_desc": spec.task_desc,
        }

    write_jsonl_atomic(successful_output, successful_rows)
    write_json_atomic(graph_dir / "index.json", index)
    summary = {
        "input_path": portable_repo_path(input_path),
        "successful_trajectory_path": portable_repo_path(successful_output),
        "output_root": portable_repo_path(output_root),
        "schedule_path": portable_repo_path(schedule_path),
        "teacher_model": portable_repo_path(teacher_model),
        "seed": seed,
        "total_trajectories": len(grouped),
        "successful_purchase_trajectories": len(successful_uids),
        "failed_or_unparsed_trajectories": len(grouped) - len(successful_uids),
        "successful_rows": len(successful_rows),
        "oracles_built": len(index["graphs"]),
        **audit,
    }
    write_json_atomic(output_root / "build_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build seed-aligned WebShop target-level oracle graphs from 7B rollouts."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successful-output", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--teacher-model", default=str(DEFAULT_TEACHER_MODEL))
    parser.add_argument("--items", type=Path, default=ref_graph.DEFAULT_ITEMS)
    parser.add_argument("--human-ins", type=Path, default=ref_graph.DEFAULT_HUMAN_INS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-incomplete-schedule", action="store_true")
    args = parser.parse_args()
    summary = build_oracles(
        input_path=args.input,
        output_root=args.output,
        successful_output=args.successful_output,
        schedule_path=args.schedule,
        teacher_model=args.teacher_model,
        seed=args.seed,
        items_path=args.items,
        human_ins_path=args.human_ins,
        overwrite=args.overwrite,
        allow_incomplete_schedule=args.allow_incomplete_schedule,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
