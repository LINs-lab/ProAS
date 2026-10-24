from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.graph_guide.oracle_build import alfworld_reference_graph as ref_graph  # noqa: E402


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = MODULE_DIR.parent / "outputs" / "7B_new" / "seed0" / "alfworld" / "GRPO_7B_infer" / "val_data" / "0.jsonl"
DEFAULT_OUTPUT = MODULE_DIR / "results" / "oracle_graphs_textworld_train" / "oracle_graphs_from_7b"
DEFAULT_TEACHER_MODEL = MODULE_DIR.parent / "models" / "GRPO-7B-ALFWorld"
SOURCE = "teacher_7b_grpo_success_raw"

ROW_KEYS = {
    "traj_uid",
    "traj_step_idx",
    "episode_group_uid",
    "episode_lengths",
    "episode_rewards",
    "task_desc",
    "task_instance",
    "task_type",
    "gamefile",
    "obs_before",
    "obs_after",
    "raw_action_text",
    "executed_action",
    "executed_action_type",
    "executed_action_args",
    "admissible_actions_before",
    "admissible_actions_after",
    "step_reward",
    "is_action_valid",
    "done",
    "won",
    "goal_condition_success_rate",
}


def portable_repo_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def iter_trajectory_rows(input_path: Path) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    """Yield (traj_uid, rows) per trajectory; rows of one trajectory must be contiguous in the JSONL."""
    current_uid: Optional[str] = None
    current_rows: List[Dict[str, Any]] = []
    completed_uids = set()

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            row = {key: payload.get(key) for key in ROW_KEYS if key in payload}
            traj_uid = str(row.get("traj_uid") or "")
            if not traj_uid:
                raise ValueError(f"Missing traj_uid at {input_path}:{line_number}")

            if current_uid is None:
                current_uid = traj_uid
            elif traj_uid != current_uid:
                completed_uids.add(current_uid)
                yield current_uid, sorted(current_rows, key=lambda item: int(item.get("traj_step_idx") or 0))
                if traj_uid in completed_uids:
                    raise ValueError(f"Trajectory {traj_uid!r} is not contiguous in {input_path}")
                current_uid = traj_uid
                current_rows = []
            current_rows.append(row)

    if current_uid is not None:
        yield current_uid, sorted(current_rows, key=lambda item: int(item.get("traj_step_idx") or 0))


def trajectory_won(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(ref_graph.truthy_number(row.get("won")) for row in rows)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def prepare_output(output_root: Path, overwrite: bool) -> Path:
    index_path = output_root / "oracle_graphs" / "index.json"
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    if index_path.exists():
        raise FileExistsError(f"Oracle index already exists: {index_path}; pass --overwrite to rebuild")
    graph_dir = output_root / "oracle_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    return graph_dir


def build_oracles(
    *,
    input_path: Path,
    output_root: Path,
    signature_mode: str = "entity",
    teacher_model: str = str(DEFAULT_TEACHER_MODEL),
    overwrite: bool = False,
) -> Dict[str, Any]:
    input_path = Path(input_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    graph_dir = prepare_output(output_root, overwrite=overwrite)
    index: Dict[str, Any] = {
        "env_adapter": "alfworld",
        "source": SOURCE,
        "signature_mode": signature_mode,
        "input_path": portable_repo_path(input_path),
        "teacher_model": portable_repo_path(teacher_model),
        "graphs": {},
    }
    summary = {
        "input_path": portable_repo_path(input_path),
        "teacher_model": portable_repo_path(teacher_model),
        "total_trajectories": 0,
        "successful_trajectories": 0,
        "failed_trajectories": 0,
        "oracles_built": 0,
        "oracles_failed": 0,
    }

    for traj_uid, rows in iter_trajectory_rows(input_path):
        summary["total_trajectories"] += 1
        if not trajectory_won(rows):
            summary["failed_trajectories"] += 1
            continue
        summary["successful_trajectories"] += 1

        oracle = ref_graph.oracle_from_success_trajectory(rows, signature_mode=signature_mode, traj_uid=traj_uid)
        if oracle is None:
            summary["oracles_failed"] += 1
            continue
        task_key = str(oracle["task_key"])
        if task_key in index["graphs"]:
            raise ValueError(f"Duplicate successful trajectory for task_key={task_key}")
        oracle["oracle_type"] = "alfworld_teacher_distilled"
        oracle["source"] = SOURCE
        oracle["state_source"] = SOURCE
        oracle["teacher_model"] = portable_repo_path(teacher_model)
        oracle["teacher_traj_uid"] = traj_uid
        oracle["distillation_input"] = portable_repo_path(input_path)

        filename = f"{ref_graph.safe_filename(task_key)}.json"
        write_json_atomic(graph_dir / filename, oracle)
        index["graphs"][task_key] = {
            "path": str(Path("oracle_graphs") / filename),
            "gamefile": oracle["gamefile"],
            "oracle_type": oracle["oracle_type"],
            "source": SOURCE,
            "num_nodes": len(oracle["nodes"]),
            "num_edges": len(oracle["edges"]),
            "num_reference_paths": len(oracle["reference_paths"]),
        }
        summary["oracles_built"] += 1
        if summary["oracles_built"] % 100 == 0:
            print(f"[teacher-oracle] successful={summary['successful_trajectories']} built={summary['oracles_built']}")

    write_json_atomic(graph_dir / "index.json", index)
    write_json_atomic(output_root / "build_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ALFWorld oracle graphs from the successful trajectories of 7B teacher rollouts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signature-mode", choices=["entity", "semantic"], default="entity")
    parser.add_argument("--teacher-model", default=str(DEFAULT_TEACHER_MODEL))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = build_oracles(
        input_path=args.input,
        output_root=args.output,
        signature_mode=args.signature_mode,
        teacher_model=args.teacher_model,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
