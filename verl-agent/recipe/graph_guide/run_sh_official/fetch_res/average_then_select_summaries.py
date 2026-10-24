#!/usr/bin/env python3
"""Average three seed curves pointwise, then select the best mean-curve step."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

from summarize_experiment_results import (
    ABLATION_ALFWORLD_METHODS,
    ABLATION_WESHOP_METHODS,
    ALFWORLD_COLUMNS,
    ALFWORLD_METHODS,
    WESHOP_COLUMNS,
    WESHOP_METHODS,
    MethodSpec,
)


REPO_GRAPH_GUIDE = Path(
    "/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent/recipe/graph_guide"
)
RESULTS_ROOT = REPO_GRAPH_GUIDE / "results"
DEFAULT_OUT_DIR = RESULTS_ROOT / "combine_summary_avg_then_select"
DEFAULT_SEEDS = ("seed0", "seed1", "seed2")
WIDE_CSV_NAME = "tensorboard_scalars_wide.csv"
MAX_STEP = 150


class AggregationError(RuntimeError):
    """Raised when raw seed curves cannot be strictly aggregated."""


@dataclass(frozen=True)
class MetricStats:
    mean: Decimal
    std: Decimal


@dataclass(frozen=True)
class ColumnSpec:
    label: str
    source: str
    scale: Decimal = Decimal("1")
    decimals: int = 1
    strip_trailing_zeros: bool = False


@dataclass(frozen=True)
class TableJob:
    filename: str
    scope: str
    model: str
    environment: str
    methods: tuple[MethodSpec, ...]
    output_columns: tuple[ColumnSpec, ...]
    success_col: str
    tie_break_col: str

    @property
    def metric_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(column.source for column in self.output_columns))


ALFWORLD_OUTPUT_COLUMNS = (
    ColumnSpec("Pick", ALFWORLD_COLUMNS["Pick"], Decimal("100"), 1),
    ColumnSpec("Look", ALFWORLD_COLUMNS["Look"], Decimal("100"), 1),
    ColumnSpec("Clean", ALFWORLD_COLUMNS["Clean"], Decimal("100"), 1),
    ColumnSpec("Heat", ALFWORLD_COLUMNS["Heat"], Decimal("100"), 1),
    ColumnSpec("Cool", ALFWORLD_COLUMNS["Cool"], Decimal("100"), 1),
    ColumnSpec("Pick2", ALFWORLD_COLUMNS["Pick2"], Decimal("100"), 1),
    ColumnSpec("ALL", ALFWORLD_COLUMNS["ALL"], Decimal("100"), 1),
    ColumnSpec(
        "TestScore",
        ALFWORLD_COLUMNS["TestScore"],
        Decimal("1"),
        3,
        strip_trailing_zeros=True,
    ),
)

WESHOP_OUTPUT_COLUMNS = (
    ColumnSpec("Score.", WESHOP_COLUMNS["Score."], Decimal("100"), 1),
    ColumnSpec("Succ.", WESHOP_COLUMNS["Succ."], Decimal("100"), 1),
)

TABLE_JOBS = (
    TableJob(
        "main_1_5B_alfworld.csv",
        "main",
        "1_5B",
        "alfworld",
        tuple(ALFWORLD_METHODS),
        ALFWORLD_OUTPUT_COLUMNS,
        ALFWORLD_COLUMNS["ALL"],
        ALFWORLD_COLUMNS["TestScore"],
    ),
    TableJob(
        "main_1_5B_weshop.csv",
        "main",
        "1_5B",
        "weshop",
        tuple(WESHOP_METHODS),
        WESHOP_OUTPUT_COLUMNS,
        WESHOP_COLUMNS["Succ."],
        WESHOP_COLUMNS["Score."],
    ),
    TableJob(
        "main_7B_alfworld.csv",
        "main",
        "7B",
        "alfworld",
        tuple(ALFWORLD_METHODS),
        ALFWORLD_OUTPUT_COLUMNS,
        ALFWORLD_COLUMNS["ALL"],
        ALFWORLD_COLUMNS["TestScore"],
    ),
    TableJob(
        "main_7B_weshop.csv",
        "main",
        "7B",
        "weshop",
        tuple(WESHOP_METHODS),
        WESHOP_OUTPUT_COLUMNS,
        WESHOP_COLUMNS["Succ."],
        WESHOP_COLUMNS["Score."],
    ),
    TableJob(
        "ablation_1_5B_alfworld.csv",
        "ablation",
        "1_5B",
        "alfworld",
        tuple(ABLATION_ALFWORLD_METHODS),
        ALFWORLD_OUTPUT_COLUMNS,
        ALFWORLD_COLUMNS["ALL"],
        ALFWORLD_COLUMNS["TestScore"],
    ),
    TableJob(
        "ablation_1_5B_weshop.csv",
        "ablation",
        "1_5B",
        "weshop",
        tuple(ABLATION_WESHOP_METHODS),
        WESHOP_OUTPUT_COLUMNS,
        WESHOP_COLUMNS["Succ."],
        WESHOP_COLUMNS["Score."],
    ),
)


def parse_decimal(value: str | None) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def parse_step(value: str | None, path: Path) -> int:
    parsed = parse_decimal(value)
    if parsed is None or parsed != parsed.to_integral_value():
        raise AggregationError(f"{path}: invalid non-integral step {value!r}")
    return int(parsed)


def mean(values: list[Decimal]) -> Decimal:
    if not values:
        raise AggregationError("cannot compute a mean without values")
    return sum(values) / Decimal(len(values))


def sample_std(values: list[Decimal]) -> Decimal:
    if len(values) < 2:
        raise AggregationError("sample standard deviation requires at least two values")
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / Decimal(len(values) - 1)
    return variance.sqrt()


def read_wide_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise AggregationError(f"missing seed curve: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def aggregate_seed_csvs(
    paths: list[Path],
    metric_columns: list[str] | tuple[str, ...],
    anchor_col: str,
    max_step: int,
) -> dict[int, dict[str, MetricStats]]:
    if len(paths) != 3:
        raise AggregationError(f"expected exactly 3 seed curves, got {len(paths)}")
    if len({path.resolve() for path in paths}) != 3:
        raise AggregationError("expected 3 distinct seed curve paths")

    headers_and_rows = [read_wide_csv(path) for path in paths]
    expected_header = headers_and_rows[0][0]
    required_columns = ["step", *metric_columns]
    missing_columns = [column for column in required_columns if column not in expected_header]
    if missing_columns:
        raise AggregationError(f"{paths[0]}: missing columns {', '.join(missing_columns)}")

    rows_by_seed: list[dict[int, dict[str, str]]] = []
    validation_steps_by_seed: list[list[int]] = []
    for path, (header, rows) in zip(paths, headers_and_rows):
        if header != expected_header:
            raise AggregationError(f"header mismatch in {path}")

        rows_by_step: dict[int, dict[str, str]] = {}
        for row in rows:
            step = parse_step(row.get("step"), path)
            if step in rows_by_step:
                raise AggregationError(f"{path}: duplicate step {step}")
            rows_by_step[step] = row
        rows_by_seed.append(rows_by_step)
        validation_steps_by_seed.append(
            sorted(
                step
                for step, row in rows_by_step.items()
                if 0 <= step <= max_step and parse_decimal(row.get(anchor_col)) is not None
            )
        )

    expected_steps = validation_steps_by_seed[0]
    if not expected_steps:
        raise AggregationError(f"{paths[0]}: no validation steps through step {max_step}")
    for path, current_steps in zip(paths[1:], validation_steps_by_seed[1:]):
        if current_steps != expected_steps:
            raise AggregationError(f"validation step grid differs in {path}")

    aggregated: dict[int, dict[str, MetricStats]] = {}
    for step in expected_steps:
        step_stats: dict[str, MetricStats] = {}
        for column in metric_columns:
            values: list[Decimal] = []
            for path, rows_by_step in zip(paths, rows_by_seed):
                value = parse_decimal(rows_by_step[step].get(column))
                if value is None:
                    raise AggregationError(
                        f"{path}: missing or invalid {column} at validation step {step}"
                    )
                values.append(value)
            step_stats[column] = MetricStats(mean(values), sample_std(values))
        aggregated[step] = step_stats
    return aggregated


def select_best_step(
    stats_by_step: dict[int, dict[str, MetricStats]],
    success_col: str,
    tie_break_col: str,
) -> int:
    if not stats_by_step:
        raise AggregationError("no aggregated validation steps to select from")
    return max(
        stats_by_step,
        key=lambda step: (
            stats_by_step[step][success_col].mean,
            stats_by_step[step][tie_break_col].mean,
            -step,
        ),
    )


def format_decimal(value: Decimal, column: ColumnSpec) -> str:
    scaled = value * column.scale
    quantum = Decimal("1").scaleb(-column.decimals)
    rounded = scaled.quantize(quantum, rounding=ROUND_HALF_UP)
    formatted = f"{rounded:.{column.decimals}f}"
    if column.strip_trailing_zeros:
        return formatted.rstrip("0").rstrip(".")
    return formatted


def format_metric_stats(stats: MetricStats, column: ColumnSpec) -> str:
    return f"{format_decimal(stats.mean, column)} ± {format_decimal(stats.std, column)}"


def build_method_row(
    method: str,
    paths: list[Path],
    output_columns: list[ColumnSpec] | tuple[ColumnSpec, ...],
    success_col: str,
    tie_break_col: str,
    max_step: int,
    required_step: int | None = None,
) -> list[str]:
    metric_columns = tuple(dict.fromkeys(column.source for column in output_columns))
    stats_by_step = aggregate_seed_csvs(paths, metric_columns, success_col, max_step)
    if required_step is None:
        selected_step = select_best_step(stats_by_step, success_col, tie_break_col)
    else:
        selected_step = required_step
        if selected_step not in stats_by_step:
            raise AggregationError(
                f"{method}: required step {selected_step} is not a shared validation step"
            )

    return [
        method,
        str(selected_step),
        *[
            format_metric_stats(stats_by_step[selected_step][column.source], column)
            for column in output_columns
        ],
    ]


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_markdown(path: Path, header: list[str], rows: list[list[str]]) -> None:
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
    if " ± " in value:
        mean_value, std_value = value.split(" ± ", 1)
        return f"${mean_value} \\pm {std_value}$"
    return latex_escape(value)


def write_latex(path: Path, header: list[str], rows: list[list[str]]) -> None:
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


def write_table_set(stem: Path, header: list[str], rows: list[list[str]]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    write_csv(stem.with_suffix(".csv"), header, rows)
    write_markdown(stem.with_suffix(".md"), header, rows)
    write_latex(stem.with_suffix(".tex"), header, rows)


def seed_paths(
    results_root: Path,
    seeds: tuple[str, ...],
    job: TableJob,
    relative_dir: str,
    scope: str | None = None,
) -> list[Path]:
    selected_scope = scope or job.scope
    return [
        results_root
        / seed
        / selected_scope
        / job.model
        / job.environment
        / relative_dir
        / WIDE_CSV_NAME
        for seed in seeds
    ]


def generate_job(
    results_root: Path,
    out_dir: Path,
    seeds: tuple[str, ...],
    job: TableJob,
    max_step: int,
) -> None:
    header = ["Method", "Best Step", *[column.label for column in job.output_columns]]
    base_paths = seed_paths(results_root, seeds, job, "grpo_exp/grpo", scope="main")
    rows = [
        build_method_row(
            "Base Model",
            base_paths,
            job.output_columns,
            job.success_col,
            job.tie_break_col,
            max_step,
            required_step=0,
        )
    ]
    for method in job.methods:
        rows.append(
            build_method_row(
                method.method,
                seed_paths(results_root, seeds, job, method.relative_dir),
                job.output_columns,
                job.success_col,
                job.tie_break_col,
                max_step,
            )
        )

    write_table_set(out_dir / Path(job.filename).stem, header, rows)


def write_readme(out_dir: Path, seeds: tuple[str, ...], max_step: int) -> None:
    seed_text = ", ".join(f"`{seed}`" for seed in seeds)
    content = f"""# Average-Then-Select Mean-Std Summaries

These tables aggregate raw validation curves over {seed_text} before selecting
the reported checkpoint.

For each method, the pipeline computes the per-step arithmetic mean and sample
standard deviation (`ddof=1`) for all metrics, using shared validation steps in
`[0, {max_step}]`. It then selects one step by maximizing mean success rate,
breaking ties with mean task/test score and then the earliest step. Every metric
is reported at that selected step. `Base Model` is fixed at step 0.

- CSV and Markdown metric cells use `avg ± std`.
- LaTeX metric cells use `$avg \\pm std$`.
- `Best Step` is a single integer selected from the cross-seed mean curve.
- Percentage metrics use one decimal place; ALFWorld `TestScore` uses up to
  three decimal places.
- Standard deviations are sample standard deviations over the three env seeds.

This differs from `results/combine_summary`, which selects each seed's best
checkpoint first and only then averages the three selected results.
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(content, encoding="utf-8")


def generate_all(
    results_root: Path,
    out_dir: Path,
    seeds: tuple[str, ...],
    max_step: int,
) -> None:
    if len(seeds) != 3:
        raise AggregationError(f"expected exactly 3 seed names, got {len(seeds)}")
    if len(set(seeds)) != 3:
        raise AggregationError("expected 3 distinct seed names")
    for job in TABLE_JOBS:
        generate_job(results_root, out_dir, seeds, job, max_step)
        print(f"wrote {out_dir / job.filename}")
        print(f"wrote {(out_dir / job.filename).with_suffix('.md')}")
        print(f"wrote {(out_dir / job.filename).with_suffix('.tex')}")
    write_readme(out_dir, seeds, max_step)
    print(f"wrote {out_dir / 'README.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average three raw seed curves pointwise, then select the best mean step."
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument("--max-step", type=int, default=MAX_STEP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    try:
        generate_all(args.results_root, args.out_dir, seeds, args.max_step)
    except AggregationError as exc:
        print(f"aggregation error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
