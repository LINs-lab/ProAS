#!/usr/bin/env python3
"""Summarize exported TensorBoard CSV metrics into paper-style result tables."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Callable


REPO_GRAPH_GUIDE = Path(
    "/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent/recipe/graph_guide"
)
MAIN_ROOT = REPO_GRAPH_GUIDE / "results" / "main"
ABLATION_ROOT = REPO_GRAPH_GUIDE / "results" / "ablation"
SUMMARY_ROOT = REPO_GRAPH_GUIDE / "results" / "summary"

MAX_STEP = 150
WIDE_CSV_NAME = "tensorboard_scalars_wide.csv"
ALFWORLD_SPEEDUP_THRESHOLD = Decimal("0.70")
WESHOP_SPEEDUP_THRESHOLD = Decimal("0.60")


ALFWORLD_COLUMNS = {
    "Look": "val/look_at_obj_in_light_success_rate",
    "Pick": "val/pick_and_place_success_rate",
    "Clean": "val/pick_clean_then_place_in_recep_success_rate",
    "Cool": "val/pick_cool_then_place_in_recep_success_rate",
    "Heat": "val/pick_heat_then_place_in_recep_success_rate",
    "Pick2": "val/pick_two_obj_and_place_success_rate",
    "ALL": "val/success_rate",
    "TestScore": "val/text/test_score",
}

WESHOP_COLUMNS = {
    "Succ.": "val/success_rate",
    "Score.": "val/webshop_task_score",
}


@dataclass(frozen=True)
class MethodSpec:
    method: str
    relative_dir: str


ALFWORLD_METHODS: list[MethodSpec] = [
    MethodSpec("GRPO", "grpo_exp/grpo"),
    MethodSpec("GRPO+ASFD", "grpo_exp/grpo_ASFD"),
    MethodSpec("RewardFlow", "grpo_exp/grpo_rewardflow"),
    MethodSpec("GIGPO", "gigpo_exp/gigpo"),
    MethodSpec("GIGPO+ASFD", "gigpo_exp/gigpo_ASFD"),
    MethodSpec("DAPO", "dapo_exp/dapo"),
    MethodSpec("DAPO+ASFD", "dapo_exp/dapo_ASFD"),
]

WESHOP_METHODS: list[MethodSpec] = [
    MethodSpec("GRPO", "grpo_exp/grpo"),
    MethodSpec("GRPO+ASFD", "grpo_exp/grpo_ASFD"),
    MethodSpec("RewardFlow", "grpo_exp/grpo_rewardflow"),
    MethodSpec("GIGPO", "gigpo_exp/gigpo"),
    MethodSpec("GIGPO+ASFD", "gigpo_exp/gigpo_ASFD"),
    MethodSpec("DAPO", "dapo_exp/dapo"),
    MethodSpec("DAPO+ASFD", "dapo_exp/dapo_ASFD"),
    MethodSpec("GRPO+ASFD (No Evolve)", "grpo_exp/grpo_ASFD_no_evolve"),
    MethodSpec("GIGPO+ASFD (No Evolve)", "gigpo_exp/gigpo_ASFD_no_evolve"),
    MethodSpec("DAPO+ASFD (No Evolve)", "dapo_exp/dapo_ASFD_no_evolve"),
]

ABLATION_WESHOP_METHODS: list[MethodSpec] = [
    MethodSpec("GRPO", "grpo"),
    MethodSpec("GRPO+ASFD", "grpo_ASFD"),
    MethodSpec("Random Miss", "random_miss"),
    MethodSpec("No Evolve", "no_evolve"),
    MethodSpec("No Pos Protect", "no_pos_protect"),
]

ABLATION_ALFWORLD_METHODS: list[MethodSpec] = [
    MethodSpec("GRPO", "grpo"),
    MethodSpec("GRPO+ASFD", "grpo_ASFD"),
    MethodSpec("Random Miss", "random_miss"),
    MethodSpec("No Evolve", "no_evolve"),
    MethodSpec("No Pos Protect", "no_pos_protect"),
]


class SummaryDataError(Exception):
    """Raised when configured experiment data cannot produce a complete summary row."""


def parse_decimal(value: str | None) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def read_wide_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_best_row(
    rows: list[dict[str, str]],
    success_col: str,
    tie_break_col: str,
    max_step: int,
) -> dict[str, str] | None:
    candidates: list[tuple[Decimal, Decimal, int, dict[str, str]]] = []
    for row in rows:
        try:
            step = int(Decimal(row.get("step", "")))
        except (InvalidOperation, TypeError, ValueError):
            continue
        if step < 0 or step > max_step:
            continue

        success = parse_decimal(row.get(success_col))
        if success is None:
            continue

        tie_value = parse_decimal(row.get(tie_break_col))
        # Prefer the earliest step only if both success and tie-breaker are equal.
        candidates.append((success, tie_value if tie_value is not None else Decimal("-Infinity"), -step, row))

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def select_step_row(rows: list[dict[str, str]], required_step: int) -> dict[str, str] | None:
    target = Decimal(required_step)
    for row in rows:
        if parse_decimal(row.get("step")) == target:
            return row
    return None


def first_hit_step(
    rows: list[dict[str, str]],
    success_col: str,
    threshold: Decimal,
    max_step: int,
) -> int | None:
    hits: list[int] = []
    for row in rows:
        try:
            step = int(Decimal(row.get("step", "")))
        except (InvalidOperation, TypeError, ValueError):
            continue
        if step < 0 or step > max_step:
            continue

        success = parse_decimal(row.get(success_col))
        if success is not None and success >= threshold:
            hits.append(step)

    return min(hits) if hits else None


def format_percent(value: str | None) -> str:
    decimal_value = parse_decimal(value)
    if decimal_value is None:
        return ""
    return str((decimal_value * Decimal("100")).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def format_score(value: str | None) -> str:
    decimal_value = parse_decimal(value)
    if decimal_value is None:
        return ""
    return str(decimal_value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)).rstrip("0").rstrip(".")


def format_speedup(baseline_step: int | None, method_step: int | None) -> str:
    if baseline_step is None or method_step is None:
        return ""
    if method_step == 0:
        return "inf" if baseline_step > 0 else "1.00"
    speedup = Decimal(baseline_step) / Decimal(method_step)
    return str(speedup.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def select_summary_row(
    method: str,
    result_dir: Path,
    success_col: str,
    tie_break_col: str,
    max_step: int,
    required_step: int | None,
) -> dict[str, str]:
    csv_path = result_dir / WIDE_CSV_NAME
    rows = read_wide_csv(csv_path)
    if required_step is not None:
        row = select_step_row(rows, required_step)
        if row is None:
            raise SummaryDataError(f"{method}: missing required step {required_step} in {csv_path}")
        return row

    row = select_best_row(
        rows=rows,
        success_col=success_col,
        tie_break_col=tie_break_col,
        max_step=max_step,
    )
    if row is None:
        raise SummaryDataError(f"{method}: no usable best row in {csv_path}")
    return row


def require_metrics(
    method: str,
    result_dir: Path,
    row: dict[str, str],
    columns: dict[str, str],
) -> None:
    missing = [label for label, column in columns.items() if parse_decimal(row.get(column)) is None]
    if missing:
        step = row.get("step", "")
        raise SummaryDataError(
            f"{method}: missing required metrics at step {step} in "
            f"{result_dir / WIDE_CSV_NAME}: {', '.join(missing)}"
        )


def summarize_alfworld_dir(
    method: str,
    result_dir: Path,
    max_step: int,
    required_step: int | None = None,
) -> list[str]:
    row = select_summary_row(
        method=method,
        result_dir=result_dir,
        success_col=ALFWORLD_COLUMNS["ALL"],
        tie_break_col=ALFWORLD_COLUMNS["TestScore"],
        max_step=max_step,
        required_step=required_step,
    )
    require_metrics(method, result_dir, row, ALFWORLD_COLUMNS)

    return [
        method,
        str(required_step) if required_step is not None else row.get("step", ""),
        format_percent(row.get(ALFWORLD_COLUMNS["Pick"])),
        format_percent(row.get(ALFWORLD_COLUMNS["Look"])),
        format_percent(row.get(ALFWORLD_COLUMNS["Clean"])),
        format_percent(row.get(ALFWORLD_COLUMNS["Heat"])),
        format_percent(row.get(ALFWORLD_COLUMNS["Cool"])),
        format_percent(row.get(ALFWORLD_COLUMNS["Pick2"])),
        format_percent(row.get(ALFWORLD_COLUMNS["ALL"])),
        format_score(row.get(ALFWORLD_COLUMNS["TestScore"])),
    ]


def summarize_weshop_dir(
    method: str,
    result_dir: Path,
    max_step: int,
    required_step: int | None = None,
) -> list[str]:
    row = select_summary_row(
        method=method,
        result_dir=result_dir,
        success_col=WESHOP_COLUMNS["Succ."],
        tie_break_col=WESHOP_COLUMNS["Score."],
        max_step=max_step,
        required_step=required_step,
    )
    require_metrics(method, result_dir, row, WESHOP_COLUMNS)

    return [
        method,
        str(required_step) if required_step is not None else row.get("step", ""),
        format_percent(row.get(WESHOP_COLUMNS["Score."])),
        format_percent(row.get(WESHOP_COLUMNS["Succ."])),
    ]


def summarize_group(
    base_dir: Path,
    base_model_dir: Path,
    methods: list[MethodSpec],
    summarizer: Callable[[str, Path, int, int | None], list[str]],
    header: list[str],
    output_path: Path,
    max_step: int,
    speedup_success_col: str,
    speedup_threshold: Decimal,
    speedup_base_by_method: dict[str, str],
) -> None:
    method_dirs = {method.method: base_dir / method.relative_dir for method in methods}
    hit_steps = {
        method: first_hit_step(
            rows=read_wide_csv(result_dir / WIDE_CSV_NAME),
            success_col=speedup_success_col,
            threshold=speedup_threshold,
            max_step=max_step,
        )
        for method, result_dir in method_dirs.items()
    }

    base_model_row = summarizer("Base Model", base_model_dir, max_step, 0)
    base_model_row.extend(["", ""])
    rows: list[list[str]] = [base_model_row]
    for method in methods:
        row = summarizer(method.method, method_dirs[method.method], max_step)
        hit_step = hit_steps[method.method]
        baseline_method = speedup_base_by_method.get(method.method)
        baseline_step = hit_steps.get(baseline_method) if baseline_method else None
        row.extend([
            "" if hit_step is None else str(hit_step),
            format_speedup(baseline_step, hit_step),
        ])
        rows.append(row)

    write_csv(output_path, header, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build summary CSV files from exported tensorboard_scalars_wide.csv files.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=MAX_STEP,
        help=f"Only consider steps in [0, max-step]. Default: {MAX_STEP}",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=SUMMARY_ROOT,
        help=f"Directory for generated summary CSVs. Default: {SUMMARY_ROOT}",
    )
    parser.add_argument(
        "--main-root",
        type=Path,
        default=MAIN_ROOT,
        help=f"Root directory for main results. Default: {MAIN_ROOT}",
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=ABLATION_ROOT,
        help=f"Root directory for ablation results. Default: {ABLATION_ROOT}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_root: Path = args.summary_root
    main_root: Path = args.main_root
    ablation_root: Path = args.ablation_root

    alfworld_header = [
        "Method",
        "Best Step",
        "Pick",
        "Look",
        "Clean",
        "Heat",
        "Cool",
        "Pick2",
        "ALL",
        "TestScore",
        "Hit Step@70",
        "Speedup@70",
    ]
    weshop_header = ["Method", "Best Step", "Score.", "Succ.", "Hit Step@60", "Speedup@60"]

    main_speedup_base_by_method = {
        "GRPO": "GRPO",
        "GRPO+ASFD": "GRPO",
        "RewardFlow": "GRPO",
        "GIGPO": "GIGPO",
        "GIGPO+ASFD": "GIGPO",
        "DAPO": "DAPO",
        "DAPO+ASFD": "DAPO",
        "GRPO+ASFD (No Evolve)": "GRPO",
        "GIGPO+ASFD (No Evolve)": "GIGPO",
        "DAPO+ASFD (No Evolve)": "DAPO",
    }
    ablation_speedup_base_by_method = {
        "GRPO": "GRPO",
        "GRPO+ASFD": "GRPO",
        "Random Miss": "GRPO",
        "No Evolve": "GRPO",
        "No Pos Protect": "GRPO",
    }

    jobs = [
        (
            main_root / "1_5B" / "alfworld",
            main_root / "1_5B" / "alfworld" / "grpo_exp" / "grpo",
            ALFWORLD_METHODS,
            summarize_alfworld_dir,
            alfworld_header,
            summary_root / "main_1_5B_alfworld.csv",
            ALFWORLD_COLUMNS["ALL"],
            ALFWORLD_SPEEDUP_THRESHOLD,
            main_speedup_base_by_method,
        ),
        (
            main_root / "7B" / "alfworld",
            main_root / "7B" / "alfworld" / "grpo_exp" / "grpo",
            ALFWORLD_METHODS,
            summarize_alfworld_dir,
            alfworld_header,
            summary_root / "main_7B_alfworld.csv",
            ALFWORLD_COLUMNS["ALL"],
            ALFWORLD_SPEEDUP_THRESHOLD,
            main_speedup_base_by_method,
        ),
        (
            main_root / "1_5B" / "weshop",
            main_root / "1_5B" / "weshop" / "grpo_exp" / "grpo",
            WESHOP_METHODS,
            summarize_weshop_dir,
            weshop_header,
            summary_root / "main_1_5B_weshop.csv",
            WESHOP_COLUMNS["Succ."],
            WESHOP_SPEEDUP_THRESHOLD,
            main_speedup_base_by_method,
        ),
        (
            main_root / "7B" / "weshop",
            main_root / "7B" / "weshop" / "grpo_exp" / "grpo",
            WESHOP_METHODS,
            summarize_weshop_dir,
            weshop_header,
            summary_root / "main_7B_weshop.csv",
            WESHOP_COLUMNS["Succ."],
            WESHOP_SPEEDUP_THRESHOLD,
            main_speedup_base_by_method,
        ),
        (
            ablation_root / "1_5B" / "weshop",
            main_root / "1_5B" / "weshop" / "grpo_exp" / "grpo",
            ABLATION_WESHOP_METHODS,
            summarize_weshop_dir,
            weshop_header,
            summary_root / "ablation_1_5B_weshop.csv",
            WESHOP_COLUMNS["Succ."],
            WESHOP_SPEEDUP_THRESHOLD,
            ablation_speedup_base_by_method,
        ),
        (
            ablation_root / "1_5B" / "alfworld",
            main_root / "1_5B" / "alfworld" / "grpo_exp" / "grpo",
            ABLATION_ALFWORLD_METHODS,
            summarize_alfworld_dir,
            alfworld_header,
            summary_root / "ablation_1_5B_alfworld.csv",
            ALFWORLD_COLUMNS["ALL"],
            ALFWORLD_SPEEDUP_THRESHOLD,
            ablation_speedup_base_by_method,
        ),
    ]

    try:
        for (
            base_dir,
            base_model_dir,
            methods,
            summarizer,
            header,
            output_path,
            speedup_success_col,
            speedup_threshold,
            speedup_base_by_method,
        ) in jobs:
            summarize_group(
                base_dir=base_dir,
                base_model_dir=base_model_dir,
                methods=methods,
                summarizer=summarizer,
                header=header,
                output_path=output_path,
                max_step=args.max_step,
                speedup_success_col=speedup_success_col,
                speedup_threshold=speedup_threshold,
                speedup_base_by_method=speedup_base_by_method,
            )
            print(f"wrote {output_path}")
    except SummaryDataError as exc:
        print(f"summary data error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
