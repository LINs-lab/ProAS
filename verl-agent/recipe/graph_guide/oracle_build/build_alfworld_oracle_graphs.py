from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.graph_guide.oracle_build import alfworld_reference_graph as ref_graph  # noqa: E402


def complete_gamefiles(data_root: Path, split: str) -> List[str]:
    split_dir = data_root / "json_2.1.1" / split
    if not split_dir.exists():
        raise FileNotFoundError(f"ALFWorld split directory not found: {split_dir}")
    gamefiles = []
    for game_path in sorted(split_dir.glob("**/game.tw-pddl")):
        if (game_path.parent / "traj_data.json").exists():
            gamefiles.append(str(game_path))
    return gamefiles


def load_existing_index(index_path: Path) -> Dict[str, Any]:
    if not index_path.exists():
        return {"graphs": {}}
    return json.loads(index_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline ALFWorld oracle graphs for a full dataset split.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get("ALFWORLD_DATA", Path.home() / ".cache/alfworld")
        ),
        help="ALFWorld data root containing json_2.1.1/.",
    )
    parser.add_argument("--split", default="train", help="Dataset split under json_2.1.1.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recipe/graph_guide/oracle_build/results/oracle_graphs_textworld_train"),
        help="Output directory; oracle_graphs/index.json is written below it.",
    )
    parser.add_argument("--signature-mode", choices=["entity", "semantic"], default="entity")
    parser.add_argument(
        "--state-source",
        choices=["abstract_replay_from_walkthrough", "textworld_replay"],
        default="textworld_replay",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Skip task keys already present in output-dir/oracle_graphs/index.json.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    all_gamefiles = complete_gamefiles(args.data_root.resolve(), args.split)
    if args.limit is not None:
        all_gamefiles = all_gamefiles[: args.limit]

    if args.missing_only:
        existing = load_existing_index(output_dir / "oracle_graphs" / "index.json")
        existing_keys = {key for key, item in existing.get("graphs", {}).items() if "path" in item}
        gamefiles = [
            gamefile
            for gamefile in all_gamefiles
            if ref_graph.task_key_from_gamefile(gamefile) not in existing_keys
        ]
        print(f"[oracle-build] total={len(all_gamefiles)} existing={len(existing_keys)} missing={len(gamefiles)}")
    else:
        gamefiles = all_gamefiles
        print(f"[oracle-build] total={len(gamefiles)}")

    if not gamefiles:
        print("[oracle-build] nothing to build")
        return

    ref_graph.build_oracle_index(
        gamefiles,
        output_dir=output_dir,
        signature_mode=args.signature_mode,
        state_source=args.state_source,
    )
    index = load_existing_index(output_dir / "oracle_graphs" / "index.json")
    built = sum(1 for item in index.get("graphs", {}).values() if "path" in item)
    failed = sum(1 for item in index.get("graphs", {}).values() if "error" in item)
    print(f"[oracle-build] wrote={output_dir / 'oracle_graphs' / 'index.json'} built={built} failed={failed}")


if __name__ == "__main__":
    main()
