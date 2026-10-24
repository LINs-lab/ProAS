"""Prepare local text placeholders for the fixed PPO task schedules.

Tasks are supplied by env.fixed_train_set, not by these parquet rows. Keeping
exactly one batch in each split gives one training step per epoch, independent
of the shared placeholder files used by other launchers. No download is needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


def prepare_split(path: Path, split: str, size: int) -> None:
    if size <= 0:
        raise ValueError("Placeholder split size must be positive")
    table = pa.Table.from_pylist([
        {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": split, "index": index},
        }
        for index in range(size)
    ])
    if path.is_file():
        metadata = pq.read_metadata(path)
        if metadata.num_rows == size and pq.read_schema(path).equals(table.schema):
            print(f"Using {path} ({size} rows)")
            return

    path.parent.mkdir(parents=True, exist_ok=True)
    # Launchers may start concurrently; publish only a complete parquet file.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        pq.write_table(table, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Prepared {path} ({size} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-data-size", type=int, default=128)
    parser.add_argument("--val-data-size", type=int, default=128)
    args = parser.parse_args()
    if args.train_data_size <= 0 or args.val_data_size <= 0:
        parser.error("data sizes must be positive")
    data_dir = args.data_dir.expanduser()
    prepare_split(data_dir / "train.parquet", "train", args.train_data_size)
    prepare_split(data_dir / "test.parquet", "test", args.val_data_size)


if __name__ == "__main__":
    main()
