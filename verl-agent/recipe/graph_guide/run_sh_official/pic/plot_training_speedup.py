#!/usr/bin/env python3
"""Plot validation success-rate curves and annotate training speedup.

The script reads exported TensorBoard scalars from recipe/graph_guide/results/main
and speedup statistics from recipe/graph_guide/results/summary. It generates
paper-ready PDF/PNG figures under run_sh_official/pic/figures.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


REPO = Path("/mnt/public/dingbowen/project_sft_rl_agenetic_training")
DEFAULT_MAIN_DIR = REPO / "verl-agent/recipe/graph_guide/results/main"
DEFAULT_SUMMARY_DIR = REPO / "verl-agent/recipe/graph_guide/results/summary"
DEFAULT_OUT_DIR = (
    REPO / "verl-agent/recipe/graph_guide/run_sh_official/pic/figures"
)

MODEL_ENV_ROWS = [
    ("1_5B", "alfworld", "Qwen2.5-1.5B-Ins / ALFWorld"),
    ("1_5B", "weshop", "Qwen2.5-1.5B-Ins / WebShop"),
    ("7B", "alfworld", "Qwen2.5-7B-Ins / ALFWorld"),
    ("7B", "weshop", "Qwen2.5-7B-Ins / WebShop"),
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
    "GRPO+ASFD": "GRPO+ASFD",
    "RewardFlow": "RewardFlow",
    "DAPO": "DAPO",
    "DAPO+ASFD": "DAPO+ASFD",
    "GIGPO": "GiGPO",
    "GIGPO+ASFD": "GiGPO+ASFD",
}

ASFD_PAIRS = {
    "GRPO": ("GRPO", "GRPO+ASFD"),
    "DAPO": ("DAPO", "DAPO+ASFD"),
    "GiGPO": ("GIGPO", "GIGPO+ASFD"),
}

COLORS = {
    "GRPO": "#4C78A8",
    "GRPO+ASFD": "#E45756",
    "RewardFlow": "#59A14F",
    "DAPO": "#4C78A8",
    "DAPO+ASFD": "#E45756",
    "GIGPO": "#4C78A8",
    "GIGPO+ASFD": "#E45756",
}

LINESTYLES = {
    "GRPO": "-",
    "GRPO+ASFD": "-",
    "RewardFlow": "-.",
    "DAPO": "-",
    "DAPO+ASFD": "-",
    "GIGPO": "-",
    "GIGPO+ASFD": "-",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", type=Path, default=DEFAULT_MAIN_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-step", type=int, default=150)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_curve(path: Path, max_step: int) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(float(row["step"]))
            if step > max_step:
                continue
            steps.append(step)
            values.append(float(row["value"]) * 100.0)
    return steps, values


def read_summary(summary_dir: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for model, env, _ in MODEL_ENV_ROWS:
        path = summary_dir / f"main_{model}_{env}.csv"
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows[(model, env, row["Method"])] = row
    return rows


def threshold_for_env(env: str) -> tuple[str, float]:
    if env == "alfworld":
        return "70", 70.0
    return "60", 60.0


def style_axes(ax: plt.Axes, env: str, max_step: int) -> None:
    _, threshold = threshold_for_env(env)
    ax.axhline(
        threshold,
        color="#6B7280",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    ax.set_xlim(0, max_step)
    ax.set_ylim(0, 103)
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    yticks = [0, 20, 40, 60, 80, 100]
    if threshold not in yticks:
        yticks.append(threshold)
        yticks.sort()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(t)}" for t in yticks])
    ax.grid(True, color="#E5E7EB", linewidth=0.6, alpha=0.9)
    ax.tick_params(labelsize=10)
    for tick, tick_value in zip(ax.yaxis.get_major_ticks(), yticks):
        tick.label1.set_fontweight("normal")
        tick.label1.set_color("#6B7280")
        if abs(tick_value - threshold) < 1e-6:
            tick.label1.set_fontweight("bold")
            tick.label1.set_color("#111827")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")


def annotate_asfd_speedup(
    ax: plt.Axes,
    summary: dict[tuple[str, str, str], dict[str, str]],
    model: str,
    env: str,
    panel_title: str,
    plotted_curves: list[tuple[list[int], list[float]]],
    curves_by_method: dict[str, tuple[list[int], list[float]]],
) -> None:
    base_method, asfd_method = ASFD_PAIRS[panel_title]
    threshold_name, threshold = threshold_for_env(env)
    base_row = summary.get((model, env, base_method))
    asfd_row = summary.get((model, env, asfd_method))
    if not base_row or not asfd_row:
        return

    base_hit = float(base_row.get(f"Hit Step@{threshold_name}", "nan"))
    asfd_hit = float(asfd_row.get(f"Hit Step@{threshold_name}", "nan"))
    speed = float(asfd_row.get(f"Speedup@{threshold_name}", "nan"))
    if speed != speed or base_hit != base_hit or asfd_hit != asfd_hit:
        return

    y_arrow = threshold
    base_cross = target_crossing_x(
        curves_by_method.get(base_method), threshold, fallback=base_hit
    )
    asfd_cross = target_crossing_x(
        curves_by_method.get(asfd_method), threshold, fallback=asfd_hit
    )
    x_left = min(asfd_cross, base_cross)
    x_right = max(asfd_cross, base_cross)
    if abs(x_right - x_left) < 1e-6:
        x_left = max(0.0, x_left - 4.0)
        x_right = min(150.0, x_right + 4.0)

    x_mid = (x_left + x_right) / 2.0
    text_y = choose_speedup_label_y(
        plotted_curves=plotted_curves,
        x_mid=x_mid,
        threshold=threshold,
    )

    ax.annotate(
        "",
        xy=(x_right, y_arrow),
        xytext=(x_left, y_arrow),
        arrowprops={
            "arrowstyle": "<|-|>",
            "color": "black",
            "lw": 1.25,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 9.5,
        },
        zorder=6,
    )
    ax.text(
        x_mid,
        text_y,
        f"{speed:.2f}x faster",
        ha="center",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color="black",
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        zorder=7,
    )


def target_crossing_x(
    curve: tuple[list[int], list[float]] | None,
    threshold: float,
    fallback: float,
) -> float:
    """Return the interpolated x-coordinate where a curve reaches threshold."""

    if curve is None:
        return fallback
    steps, values = curve
    if not steps:
        return fallback
    if values[0] >= threshold:
        return float(steps[0])
    for idx in range(1, len(steps)):
        prev_step, prev_value = float(steps[idx - 1]), float(values[idx - 1])
        step, value = float(steps[idx]), float(values[idx])
        if value >= threshold:
            if abs(value - prev_value) < 1e-12:
                return step
            ratio = (threshold - prev_value) / (value - prev_value)
            ratio = min(1.0, max(0.0, ratio))
            return prev_step + ratio * (step - prev_step)
    return fallback


def choose_speedup_label_y(
    plotted_curves: list[tuple[list[int], list[float]]],
    x_mid: float,
    threshold: float,
) -> float:
    """Pick a label height that is far from nearby curve segments."""

    candidates = [
        threshold + 4.0,
        threshold + 10.0,
        threshold - 16.0,
    ]
    candidates = [min(94.0, max(8.0, y)) for y in candidates]
    candidates = list(dict.fromkeys(candidates))

    nearby_values: list[float] = []
    x_window = 38.0
    for steps, values in plotted_curves:
        for step, value in zip(steps, values):
            if abs(step - x_mid) <= x_window:
                nearby_values.append(value)

    if not nearby_values:
        return candidates[0]

    preferred_y = min(94.0, threshold + 4.0)
    text_height = 8.0
    best_y = candidates[0]
    best_score = -1.0
    for y in candidates:
        bottom = y - 1.5
        top = y + text_height
        distances = []
        overlaps = 0
        for value in nearby_values:
            if bottom <= value <= top:
                overlaps += 1
                distances.append(0.0)
            else:
                distances.append(min(abs(value - bottom), abs(value - top)))
        # Prefer labels close to the arrow; move farther only when curves are
        # genuinely crowded around the target line.
        score = min(distances) - overlaps * 12.0 - abs(y - preferred_y) * 0.18
        if score > best_score:
            best_score = score
            best_y = y
    return best_y


def plot_panel(
    ax: plt.Axes,
    main_dir: Path,
    summary: dict[tuple[str, str, str], dict[str, str]],
    model: str,
    env: str,
    panel_title: str,
    series: list[tuple[str, str]],
    max_step: int,
) -> None:
    plotted_curves: list[tuple[list[int], list[float]]] = []
    curves_by_method: dict[str, tuple[list[int], list[float]]] = {}
    for method, rel_dir in series:
        path = main_dir / model / env / rel_dir / "val_success_rate.csv"
        if not path.exists():
            print(f"[WARN] missing curve: {path}")
            continue
        steps, values = read_curve(path, max_step=max_step)
        if not steps:
            print(f"[WARN] empty curve within step <= {max_step}: {path}")
            continue
        plotted_curves.append((steps, values))
        curves_by_method[method] = (steps, values)
        ax.plot(
            steps,
            values,
            label=LABELS.get(method, method),
            color=COLORS.get(method, "#111827"),
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=2.0 if method.endswith("+ASFD") else 1.7,
            alpha=0.98,
        )

    style_axes(ax, env, max_step)
    ax.set_title(panel_title, fontsize=13, pad=6)
    annotate_asfd_speedup(
        ax, summary, model, env, panel_title, plotted_curves, curves_by_method
    )
    ax.legend(loc="lower right", fontsize=9.5, frameon=False, handlelength=2.4)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    print(f"[INFO] wrote {pdf}")
    print(f"[INFO] wrote {png}")


def make_overview(
    main_dir: Path,
    summary: dict[tuple[str, str, str], dict[str, str]],
    out_dir: Path,
    max_step: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(MODEL_ENV_ROWS),
        ncols=len(PANELS),
        figsize=(10.8, 12.4),
        sharex=True,
        sharey=False,
    )
    for row_idx, (model, env, row_title) in enumerate(MODEL_ENV_ROWS):
        for col_idx, (panel_title, series) in enumerate(PANELS):
            ax = axes[row_idx][col_idx]
            plot_panel(
                ax=ax,
                main_dir=main_dir,
                summary=summary,
                model=model,
                env=env,
                panel_title=panel_title,
                series=series,
                max_step=max_step,
            )
            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\nSuccess rate (%)", fontsize=12)
            else:
                ax.tick_params(labelleft=False)
            if row_idx == len(MODEL_ENV_ROWS) - 1:
                ax.set_xlabel("Training step", fontsize=12)
    fig.suptitle("Validation Curves and Training Speedup", fontsize=17, y=0.995)
    fig.tight_layout(h_pad=2.2, w_pad=0.8)
    save_figure(fig, out_dir, "training_speedup_overview", dpi)
    plt.close(fig)


def make_row_figures(
    main_dir: Path,
    summary: dict[tuple[str, str, str], dict[str, str]],
    out_dir: Path,
    max_step: int,
    dpi: int,
) -> None:
    for model, env, row_title in MODEL_ENV_ROWS:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(PANELS),
            figsize=(10.8, 3.1),
            sharex=True,
            sharey=True,
        )
        for ax, (panel_title, series) in zip(axes, PANELS):
            plot_panel(
                ax=ax,
                main_dir=main_dir,
                summary=summary,
                model=model,
                env=env,
                panel_title=panel_title,
                series=series,
                max_step=max_step,
            )
            ax.set_xlabel("Training step", fontsize=12)
        axes[0].set_ylabel("Success rate (%)", fontsize=12)
        fig.suptitle(row_title, fontsize=16, y=1.03)
        fig.tight_layout(w_pad=0.9)
        save_figure(fig, out_dir, f"training_speedup_{model}_{env}", dpi)
        plt.close(fig)


def write_latex_snippet(out_dir: Path) -> None:
    snippet = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/training_speedup_overview.pdf}
\caption{Validation success-rate curves on ALFWorld and WebShop within the first 150 RL steps. Dashed horizontal lines mark the fixed success-rate targets used for computing training speedup: 70 points for ALFWorld and 60 points for WebShop. Model names in the figure abbreviate Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct as Qwen2.5-1.5B-Ins and Qwen2.5-7B-Ins. Black double arrows indicate the speedup of each base algorithm after adding ASFD.}
\label{fig:training_speedup_overview}
\end{figure*}
"""
    path = out_dir / "training_speedup_latex_snippet.tex"
    path.write_text(snippet)
    print(f"[INFO] wrote {path}")


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    summary = read_summary(args.summary_dir)
    make_overview(args.main_dir, summary, args.out_dir, args.max_step, args.dpi)
    make_row_figures(args.main_dir, summary, args.out_dir, args.max_step, args.dpi)
    write_latex_snippet(args.out_dir)


if __name__ == "__main__":
    main()
