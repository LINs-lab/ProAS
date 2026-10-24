#!/usr/bin/env python3
"""Average per-seed summary CSV files without speedup columns."""

from __future__ import annotations

import argparse
import csv
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path


REPO_GRAPH_GUIDE = Path(
    "/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent/recipe/graph_guide"
)
RESULTS_ROOT = REPO_GRAPH_GUIDE / "results"
DEFAULT_SEEDS = ["seed0", "seed1", "seed2"]
DEFAULT_OUT_DIR = RESULTS_ROOT / "avg_summary"
DEFAULT_STD_OUT_DIR = RESULTS_ROOT / "std_summary"
DEFAULT_COMBINE_OUT_DIR = RESULTS_ROOT / "combine_summary"


class AggregationError(RuntimeError):
    """Raised when seed summary files cannot be strictly aggregated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average seed summary CSVs and omit hit-step/speedup columns.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help=f"Root directory containing seed*/summary. Default: {RESULTS_ROOT}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for averaged summary CSVs. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--std-out-dir",
        type=Path,
        default=DEFAULT_STD_OUT_DIR,
        help=f"Directory for sample-std CSVs. Default: {DEFAULT_STD_OUT_DIR}",
    )
    parser.add_argument(
        "--combine-out-dir",
        type=Path,
        default=DEFAULT_COMBINE_OUT_DIR,
        help=f"Directory for combined CSV/Markdown/LaTeX tables. Default: {DEFAULT_COMBINE_OUT_DIR}",
    )
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        help="Seed directory name to average. Can be repeated. Default: seed0/seed1/seed2.",
    )
    return parser.parse_args()


def parse_decimal(value: str | None) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def mean(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    return sum(values) / Decimal(len(values))


def sample_std(values: list[Decimal]) -> Decimal:
    if len(values) < 2:
        raise AggregationError("sample standard deviation requires at least two values")
    avg = mean(values)
    assert avg is not None
    variance = sum((value - avg) ** 2 for value in values) / Decimal(len(values) - 1)
    return variance.sqrt()


def format_average(column: str, value: Decimal | None) -> str:
    if value is None:
        return ""
    if column == "TestScore":
        return str(value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)).rstrip("0").rstrip(".")
    return str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def is_speedup_column(column: str) -> bool:
    return column.startswith("Hit Step@") or column.startswith("Speedup@")


def read_summary(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_markdown(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def latex_cell(value: str) -> str:
    if " +/- " in value:
        mean_value, std_value = value.split(" +/- ", 1)
        return f"${mean_value} \\pm {std_value}$"
    return latex_escape(value)


def write_latex(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" + "r" * (len(header) - 1)
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(column) for column in header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(latex_cell(cell) for cell in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_seed_tables(paths: list[Path]) -> tuple[list[str], list[str], list[list[dict[str, str]]]]:
    if not paths:
        raise AggregationError("no summary files provided")
    if len(paths) != 3:
        raise AggregationError(f"expected exactly 3 seed summary files, got {len(paths)}")

    headers, first_rows = read_summary(paths[0])
    if "Method" not in headers:
        raise AggregationError(f"{paths[0]} is missing Method column")
    method_order = [row.get("Method", "") for row in first_rows]

    all_rows = [first_rows]
    for path in paths[1:]:
        current_headers, current_rows = read_summary(path)
        if current_headers != headers:
            raise AggregationError(f"header mismatch in {path}")
        current_method_order = [row.get("Method", "") for row in current_rows]
        if current_method_order != method_order:
            raise AggregationError(f"inconsistent method order in {path}")
        all_rows.append(current_rows)

    return headers, method_order, all_rows


def require_values(
    paths: list[Path],
    rows_by_seed: list[dict[str, dict[str, str]]],
    method: str,
    column: str,
) -> list[Decimal]:
    values: list[Decimal] = []
    for path, rows in zip(paths, rows_by_seed):
        value = rows[method].get(column)
        parsed = parse_decimal(value)
        if parsed is None:
            raise AggregationError(f"{path}: {method} has missing or invalid {column}")
        values.append(parsed)
    return values


def aggregate_file(paths: list[Path], avg_path: Path, std_path: Path, combine_dir: Path) -> None:
    headers, method_order, seed_rows = validate_seed_tables(paths)
    output_headers = [column for column in headers if not is_speedup_column(column)]

    rows_by_seed: list[dict[str, dict[str, str]]] = []
    for rows in seed_rows:
        rows_by_seed.append({row["Method"]: row for row in rows})

    avg_rows: list[list[str]] = []
    std_rows: list[list[str]] = []
    combined_rows: list[list[str]] = []
    for method in method_order:
        avg_row: list[str] = []
        std_row: list[str] = []
        combined_row: list[str] = []
        for column in output_headers:
            if column == "Method":
                avg_row.append(method)
                std_row.append(method)
                combined_row.append(method)
                continue
            values = require_values(paths, rows_by_seed, method, column)
            avg_value = format_average(column, mean(values))
            std_value = format_average(column, sample_std(values))
            avg_row.append(avg_value)
            std_row.append(std_value)
            combined_row.append(f"{avg_value} +/- {std_value}")
        avg_rows.append(avg_row)
        std_rows.append(std_row)
        combined_rows.append(combined_row)

    combine_csv_path = combine_dir / avg_path.name
    write_csv(avg_path, output_headers, avg_rows)
    write_csv(std_path, output_headers, std_rows)
    write_csv(combine_csv_path, output_headers, combined_rows)
    write_markdown(combine_csv_path.with_suffix(".md"), output_headers, combined_rows)
    write_latex(combine_csv_path.with_suffix(".tex"), output_headers, combined_rows)

def main() -> int:
    args = parse_args()
    seeds = args.seed or DEFAULT_SEEDS
    summary_dirs = [args.results_root / seed / "summary" for seed in seeds]
    missing_dirs = [path for path in summary_dirs if not path.exists()]
    if missing_dirs:
        for path in missing_dirs:
            print(f"missing summary dir: {path}")
        return 2

    filenames = sorted(path.name for path in summary_dirs[0].glob("*.csv"))
    for summary_dir in summary_dirs[1:]:
        current = sorted(path.name for path in summary_dir.glob("*.csv"))
        if current != filenames:
            print(f"summary file set differs in {summary_dir}")
            print(f"expected: {filenames}")
            print(f"actual: {current}")
            return 2

    try:
        for filename in filenames:
            paths = [summary_dir / filename for summary_dir in summary_dirs]
            avg_path = args.out_dir / filename
            std_path = args.std_out_dir / filename
            aggregate_file(paths, avg_path, std_path, args.combine_out_dir)
            print(f"wrote {avg_path}")
            print(f"wrote {std_path}")
            print(f"wrote {args.combine_out_dir / filename}")
            print(f"wrote {(args.combine_out_dir / filename).with_suffix('.md')}")
            print(f"wrote {(args.combine_out_dir / filename).with_suffix('.tex')}")
    except AggregationError as exc:
        print(f"aggregation error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
