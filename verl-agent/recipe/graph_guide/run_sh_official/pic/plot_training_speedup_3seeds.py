#!/usr/bin/env python3
"""Plot mean validation curves with sample-std bands over three env seeds."""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt


REPO = Path("/mnt/public/dingbowen/project_sft_rl_agenetic_training")
DEFAULT_RESULTS_ROOT = REPO / "verl-agent/recipe/graph_guide/results"
DEFAULT_OUT_DIR = (
    REPO
    / "verl-agent/recipe/graph_guide/run_sh_official/pic/3_env_seed_figures"
)
DEFAULT_SEEDS = ("seed0", "seed1", "seed2")

MODEL_ENV_ROWS = [
    ("1_5B", "alfworld", "Qwen2.5-1.5B-Instruct / ALFWorld"),
    ("1_5B", "weshop", "Qwen2.5-1.5B-Instruct / WebShop"),
    ("7B", "alfworld", "Qwen2.5-7B-Instruct / ALFWorld"),
    ("7B", "weshop", "Qwen2.5-7B-Instruct / WebShop"),
]

PANELS = [
    (
        "GRPO",
        [
            ("GRPO", "grpo_exp/grpo"),
            ("RewardFlow", "grpo_exp/grpo_rewardflow"),
            ("GRPO+ASFD", "grpo_exp/grpo_ASFD"),
        ],
    ),
    (
        "DAPO",
        [
            ("DAPO", "dapo_exp/dapo"),
            ("DAPO+ASFD", "dapo_exp/dapo_ASFD"),
        ],
    ),
    (
        "GiGPO",
        [
            ("GIGPO", "gigpo_exp/gigpo"),
            ("GIGPO+ASFD", "gigpo_exp/gigpo_ASFD"),
        ],
    ),
]

LABELS = {
    "GRPO": "GRPO",
    "RewardFlow": "RewardFlow",
    "GRPO+ASFD": "GRPO+ASFD",
    "DAPO": "DAPO",
    "DAPO+ASFD": "DAPO+ASFD",
    "GIGPO": "GiGPO",
    "GIGPO+ASFD": "GiGPO+ASFD",
}

COLORS = {
    "GRPO": "#4E79A7",
    "RewardFlow": "#59A14F",
    "GRPO+ASFD": "#E15759",
    "DAPO": "#4E79A7",
    "DAPO+ASFD": "#E15759",
    "GIGPO": "#4E79A7",
    "GIGPO+ASFD": "#E15759",
}

LINESTYLES = {
    "GRPO": "-",
    "RewardFlow": "-.",
    "GRPO+ASFD": "-",
    "DAPO": "-",
    "DAPO+ASFD": "-",
    "GIGPO": "-",
    "GIGPO+ASFD": "-",
}

ASFD_PAIRS = {
    "GRPO": ("GRPO", "GRPO+ASFD"),
    "DAPO": ("DAPO", "DAPO+ASFD"),
    "GiGPO": ("GIGPO", "GIGPO+ASFD"),
}


@dataclass(frozen=True)
class CurveStats:
    steps: list[int]
    mean: list[float]
    std: list[float]


class CurveDataError(RuntimeError):
    """Raised when seed curves cannot be aggregated safely."""


def read_curve(path: Path, max_step: int) -> dict[int, float]:
    if not path.exists():
        raise CurveDataError(f"missing curve: {path}")

    curve: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "value"}
        if not required.issubset(reader.fieldnames or []):
            raise CurveDataError(f"{path} must contain step and value columns")

        for row in reader:
            try:
                step = int(float(row["step"]))
                value = float(row["value"]) * 100.0
            except (TypeError, ValueError) as exc:
                raise CurveDataError(f"invalid numeric row in {path}: {row}") from exc
            if step > max_step:
                continue
            if step in curve:
                raise CurveDataError(f"duplicate step {step} in {path}")
            curve[step] = value

    if not curve:
        raise CurveDataError(f"no points through step {max_step}: {path}")
    return dict(sorted(curve.items()))


def aggregate_seed_curves(paths: list[Path], max_step: int) -> CurveStats:
    if len(paths) != 3:
        raise CurveDataError(f"expected exactly 3 seed curves, got {len(paths)}")

    curves = [read_curve(path, max_step=max_step) for path in paths]
    steps = list(curves[0])
    for path, curve in zip(paths[1:], curves[1:]):
        if list(curve) != steps:
            raise CurveDataError(f"step grid differs in {path}")

    columns = [[curve[step] for curve in curves] for step in steps]
    return CurveStats(
        steps=steps,
        mean=[statistics.fmean(values) for values in columns],
        std=[statistics.stdev(values) for values in columns],
    )


def crossing_step(
    steps: list[int], values: list[float], threshold: float
) -> float | None:
    if not steps or len(steps) != len(values):
        return None
    if values[0] >= threshold:
        return float(steps[0])

    for index in range(1, len(steps)):
        previous_step = float(steps[index - 1])
        previous_value = values[index - 1]
        step = float(steps[index])
        value = values[index]
        if value < threshold:
            continue
        if value == previous_value:
            return step
        ratio = (threshold - previous_value) / (value - previous_value)
        return previous_step + ratio * (step - previous_step)
    return None


def format_speedup(speedup: float) -> str:
    if speedup >= 1.0:
        return f"{speedup:.2f}x faster"
    return f"{speedup:.2f}x speedup (slower)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean +/- sample-std validation curves over three env seeds."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument("--max-step", type=int, default=150)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def threshold_for_env(env: str) -> float:
    return 70.0 if env == "alfworld" else 60.0


def curve_paths(
    results_root: Path,
    seeds: tuple[str, ...],
    model: str,
    env: str,
    relative_dir: str,
) -> list[Path]:
    return [
        results_root
        / seed
        / "main"
        / model
        / env
        / relative_dir
        / "val_success_rate.csv"
        for seed in seeds
    ]


def style_axes(ax: plt.Axes, env: str, max_step: int) -> None:
    threshold = threshold_for_env(env)
    ax.axhline(
        threshold,
        color="#7C8798",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    ax.set_xlim(0, max_step)
    ax.set_ylim(0, 103)
    tick_interval = max(1, max_step // 5)
    xticks = list(range(0, max_step + 1, tick_interval))
    if xticks[-1] != max_step:
        xticks.append(max_step)
    ax.set_xticks(xticks)

    yticks = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    if threshold not in yticks:
        yticks.append(threshold)
        yticks.sort()
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(value)) for value in yticks])
    ax.grid(True, color="#E8ECF2", linewidth=0.65, alpha=0.9)
    ax.tick_params(axis="both", labelsize=9.5, colors="#4B5563", width=0.7)

    for tick, tick_value in zip(ax.yaxis.get_major_ticks(), yticks):
        if abs(tick_value - threshold) < 1e-6:
            tick.label1.set_color("#1F2937")
            tick.label1.set_fontweight("semibold")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#AAB2BF")
    ax.spines["bottom"].set_color("#AAB2BF")
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)


def choose_speedup_label_y(
    curves: list[CurveStats], x_mid: float, threshold: float
) -> float:
    candidates = [threshold + 4.0, threshold + 12.0, threshold + 20.0]
    candidates = [min(94.0, max(8.0, value)) for value in candidates]
    nearby_values: list[float] = []
    for stats in curves:
        for step, mean, std in zip(stats.steps, stats.mean, stats.std):
            if abs(step - x_mid) <= 35.0:
                nearby_values.extend(
                    [max(0.0, mean - std), mean, min(100.0, mean + std)]
                )
    if not nearby_values:
        return candidates[0]

    best_y = candidates[0]
    best_score = float("-inf")
    for y_value in candidates:
        label_bottom = y_value - 1.5
        label_top = y_value + 8.5
        distances: list[float] = []
        overlaps = 0
        for curve_value in nearby_values:
            if label_bottom <= curve_value <= label_top:
                overlaps += 1
                distances.append(0.0)
            else:
                distances.append(
                    min(
                        abs(curve_value - label_bottom),
                        abs(curve_value - label_top),
                    )
                )
        score = min(distances) - overlaps * 10.0 - abs(y_value - threshold) * 0.1
        if score > best_score:
            best_y = y_value
            best_score = score
    return best_y


def annotate_speedup(
    ax: plt.Axes,
    env: str,
    panel_title: str,
    curves_by_method: dict[str, CurveStats],
) -> None:
    base_method, asfd_method = ASFD_PAIRS[panel_title]
    base_curve = curves_by_method.get(base_method)
    asfd_curve = curves_by_method.get(asfd_method)
    if base_curve is None or asfd_curve is None:
        return

    threshold = threshold_for_env(env)
    base_crossing = crossing_step(base_curve.steps, base_curve.mean, threshold)
    asfd_crossing = crossing_step(asfd_curve.steps, asfd_curve.mean, threshold)
    if base_crossing is None or asfd_crossing is None:
        print(
            f"[WARN] no speedup arrow for {panel_title}/{env}: "
            "a mean curve does not reach the target"
        )
        return
    if asfd_crossing <= 0.0:
        print(
            f"[WARN] no speedup arrow for {panel_title}/{env}: "
            "ASFD crossing is at step zero"
        )
        return

    x_left = min(base_crossing, asfd_crossing)
    x_right = max(base_crossing, asfd_crossing)
    if abs(x_right - x_left) < 1e-6:
        x_left = max(0.0, x_left - 3.0)
        x_right += 3.0
    x_mid = (x_left + x_right) / 2.0
    label_y = choose_speedup_label_y(
        list(curves_by_method.values()), x_mid=x_mid, threshold=threshold
    )
    speedup = base_crossing / asfd_crossing
    label = format_speedup(speedup)

    ax.annotate(
        "",
        xy=(x_right, threshold),
        xytext=(x_left, threshold),
        arrowprops={
            "arrowstyle": "<|-|>",
            "color": "#111827",
            "lw": 1.15,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 9.0,
        },
        zorder=6,
    )
    ax.text(
        x_mid,
        label_y,
        label,
        ha="center",
        va="bottom",
        fontsize=9.8 if speedup < 1.0 else 11.5,
        fontweight="semibold",
        color="#111827",
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        zorder=7,
    )


def plot_panel(
    ax: plt.Axes,
    results_root: Path,
    seeds: tuple[str, ...],
    model: str,
    env: str,
    panel_title: str,
    series: list[tuple[str, str]],
    max_step: int,
) -> None:
    curves_by_method: dict[str, CurveStats] = {}
    for method, relative_dir in series:
        paths = curve_paths(results_root, seeds, model, env, relative_dir)
        stats = aggregate_seed_curves(paths, max_step=max_step)
        curves_by_method[method] = stats
        color = COLORS[method]
        lower = [max(0.0, mean - std) for mean, std in zip(stats.mean, stats.std)]
        upper = [min(100.0, mean + std) for mean, std in zip(stats.mean, stats.std)]
        ax.fill_between(
            stats.steps,
            lower,
            upper,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            stats.steps,
            stats.mean,
            label=LABELS[method],
            color=color,
            linestyle=LINESTYLES[method],
            linewidth=2.2 if method.endswith("+ASFD") else 1.8,
            alpha=0.98,
            zorder=3,
        )

    style_axes(ax, env=env, max_step=max_step)
    ax.set_title(panel_title, fontsize=12.5, pad=6, color="#111827")
    annotate_speedup(ax, env, panel_title, curves_by_method)
    ax.legend(
        loc="lower right",
        fontsize=8.8,
        frameon=False,
        handlelength=2.5,
        labelspacing=0.35,
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=dpi)
    print(f"[INFO] wrote {pdf_path}")
    print(f"[INFO] wrote {png_path}")


def make_overview(
    results_root: Path,
    seeds: tuple[str, ...],
    out_dir: Path,
    max_step: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(MODEL_ENV_ROWS),
        ncols=len(PANELS),
        figsize=(11.6, 12.7),
        sharex=True,
        sharey=True,
    )
    for row_index, (model, env, row_title) in enumerate(MODEL_ENV_ROWS):
        for column_index, (panel_title, series) in enumerate(PANELS):
            ax = axes[row_index][column_index]
            plot_panel(
                ax,
                results_root,
                seeds,
                model,
                env,
                panel_title,
                series,
                max_step,
            )
            if column_index == 0:
                ax.set_ylabel(f"{row_title}\nSuccess rate (%)", fontsize=10.8)
            else:
                ax.tick_params(labelleft=False)
            if row_index == len(MODEL_ENV_ROWS) - 1:
                ax.set_xlabel("Training step", fontsize=10.8)

    fig.suptitle(
        "Validation Curves Across Environment Seeds",
        fontsize=17,
        y=0.995,
        color="#111827",
    )
    fig.text(
        0.5,
        0.974,
        r"Mean $\pm$ 1 sample std over env seeds 0, 1, and 2",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#5B6472",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.956), h_pad=2.0, w_pad=0.8)
    save_figure(fig, out_dir, "training_speedup_3seeds_overview", dpi)
    plt.close(fig)


def make_row_figures(
    results_root: Path,
    seeds: tuple[str, ...],
    out_dir: Path,
    max_step: int,
    dpi: int,
) -> None:
    for model, env, row_title in MODEL_ENV_ROWS:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(PANELS),
            figsize=(11.6, 3.35),
            sharex=True,
            sharey=True,
        )
        for ax, (panel_title, series) in zip(axes, PANELS):
            plot_panel(
                ax,
                results_root,
                seeds,
                model,
                env,
                panel_title,
                series,
                max_step,
            )
            ax.set_xlabel("Training step", fontsize=10.8)
        axes[0].set_ylabel("Success rate (%)", fontsize=10.8)
        fig.suptitle(row_title, fontsize=15.5, y=0.98, color="#111827")
        fig.text(
            0.5,
            0.895,
            r"Mean $\pm$ 1 sample std over env seeds 0, 1, and 2",
            ha="center",
            va="top",
            fontsize=9.5,
            color="#5B6472",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.82), w_pad=0.9)
        save_figure(
            fig,
            out_dir,
            f"training_speedup_3seeds_{model}_{env}",
            dpi,
        )
        plt.close(fig)


def write_latex_snippet(out_dir: Path) -> None:
    snippet = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/training_speedup_3seeds_overview.pdf}
\caption{Validation success-rate curves on ALFWorld and WebShop over environment seeds 0, 1, and 2. Lines show the mean and shaded regions show $\pm 1$ sample standard deviation across the three seeds. Dashed horizontal lines mark the fixed success-rate targets used for computing training speedup: 70 points for ALFWorld and 60 points for WebShop. Black double arrows compare the threshold-crossing steps of the base and ASFD mean curves.}
\label{fig:training_speedup_3seeds_overview}
\end{figure*}
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "training_speedup_3seeds_latex_snippet.tex"
    path.write_text(snippet, encoding="utf-8")
    print(f"[INFO] wrote {path}")


def render_all(
    results_root: Path,
    out_dir: Path,
    max_step: int = 150,
    dpi: int = 300,
    seeds: tuple[str, ...] = DEFAULT_SEEDS,
) -> None:
    if len(seeds) != 3:
        raise CurveDataError(f"expected exactly 3 seeds, got {len(seeds)}")
    make_overview(results_root, seeds, out_dir, max_step, dpi)
    make_row_figures(results_root, seeds, out_dir, max_step, dpi)
    write_latex_snippet(out_dir)


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.75,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    render_all(
        results_root=args.results_root,
        out_dir=args.out_dir,
        max_step=args.max_step,
        dpi=args.dpi,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
