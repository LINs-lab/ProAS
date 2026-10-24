#!/usr/bin/env python3
"""Compare WebShop ASFD curves with and without online oracle evolution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from plot_training_speedup_3seeds import (
    DEFAULT_SEEDS,
    CurveDataError,
    CurveStats,
    aggregate_seed_curves,
    crossing_step,
    curve_paths,
    save_figure,
    style_axes,
)


REPO = Path("/mnt/public/dingbowen/project_sft_rl_agenetic_training")
DEFAULT_RESULTS_ROOT = REPO / "verl-agent/recipe/graph_guide/results"
DEFAULT_OUT_DIR = (
    REPO
    / "verl-agent/recipe/graph_guide/run_sh_official/pic/"
    "3_env_seed_figures_webshop_no_oracle_evolve"
)

MODEL_ROWS = [
    ("1_5B", "Qwen2.5-1.5B-Instruct / WebShop"),
    ("7B", "Qwen2.5-7B-Instruct / WebShop"),
]


PANELS = [
    (
        "GRPO",
        [
            ("GRPO", "grpo_exp/grpo"),
            ("GRPO+ASFD", "grpo_exp/grpo_ASFD"),
            (
                "GRPO+ASFD (No Evolve)",
                "grpo_exp/grpo_ASFD_no_evolve",
            ),
        ],
    ),
    (
        "DAPO",
        [
            ("DAPO", "dapo_exp/dapo"),
            ("DAPO+ASFD", "dapo_exp/dapo_ASFD"),
            (
                "DAPO+ASFD (No Evolve)",
                "dapo_exp/dapo_ASFD_no_evolve",
            ),
        ],
    ),
    (
        "GiGPO",
        [
            ("GIGPO", "gigpo_exp/gigpo"),
            ("GIGPO+ASFD", "gigpo_exp/gigpo_ASFD"),
            (
                "GIGPO+ASFD (No Evolve)",
                "gigpo_exp/gigpo_ASFD_no_evolve",
            ),
        ],
    ),
]

COLORS = {
    "base": "#4E79A7",
    "evolve": "#E15759",
    "no_evolve": "#2A9D8F",
}

LINESTYLES = {
    "base": "-",
    "evolve": "-",
    "no_evolve": "--",
}


def format_ratio(
    base_crossing: float | None, variant_crossing: float | None
) -> str:
    if base_crossing is None or variant_crossing is None or variant_crossing <= 0.0:
        return "not reached"
    return f"{base_crossing / variant_crossing:.2f}x"


def speedup_rows(
    base_crossing: float | None,
    evolve_crossing: float | None,
    no_evolve_crossing: float | None,
) -> list[tuple[str, str]]:
    return [
        ("Evolve", format_ratio(base_crossing, evolve_crossing)),
        ("No Evolve", format_ratio(base_crossing, no_evolve_crossing)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare WebShop ASFD curves with and without online oracle evolution "
            "over three environment seeds."
        )
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument("--max-step", type=int, default=150)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def curve_role(method: str) -> str:
    if method.endswith("(No Evolve)"):
        return "no_evolve"
    if "+ASFD" in method:
        return "evolve"
    return "base"


def annotate_speedups(
    ax: plt.Axes,
    base_curve: CurveStats,
    evolve_curve: CurveStats,
    no_evolve_curve: CurveStats,
) -> None:
    threshold = 60.0
    base_crossing = crossing_step(base_curve.steps, base_curve.mean, threshold)
    evolve_crossing = crossing_step(evolve_curve.steps, evolve_curve.mean, threshold)
    no_evolve_crossing = crossing_step(
        no_evolve_curve.steps, no_evolve_curve.mean, threshold
    )
    rows = speedup_rows(base_crossing, evolve_crossing, no_evolve_crossing)

    ax.text(
        0.035,
        0.965,
        "Speedup @ 60%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="semibold",
        color="#4B5563",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
            "pad": 2.2,
        },
        zorder=8,
    )
    for row_index, ((label, value), color) in enumerate(
        zip(rows, (COLORS["evolve"], COLORS["no_evolve"]))
    ):
        ax.text(
            0.035,
            0.895 - row_index * 0.065,
            f"{label}: {value}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            fontweight="semibold",
            color=color,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
                "pad": 1.2,
            },
            zorder=8,
        )


def plot_panel(
    ax: plt.Axes,
    results_root: Path,
    seeds: tuple[str, ...],
    model: str,
    panel_title: str,
    series: list[tuple[str, str]],
    max_step: int,
) -> None:
    curves: list[CurveStats] = []
    for method, relative_dir in series:
        stats = aggregate_seed_curves(
            curve_paths(
                results_root,
                seeds,
                model,
                "weshop",
                relative_dir,
            ),
            max_step=max_step,
        )
        curves.append(stats)
        role = curve_role(method)
        color = COLORS[role]
        lower = [max(0.0, mean - std) for mean, std in zip(stats.mean, stats.std)]
        upper = [min(100.0, mean + std) for mean, std in zip(stats.mean, stats.std)]
        ax.fill_between(
            stats.steps,
            lower,
            upper,
            color=color,
            alpha=0.11,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            stats.steps,
            stats.mean,
            label=method,
            color=color,
            linestyle=LINESTYLES[role],
            linewidth=2.15 if role != "base" else 1.85,
            alpha=0.98,
            zorder=3,
        )

    style_axes(ax, env="weshop", max_step=max_step)
    ax.set_title(panel_title, fontsize=12.5, pad=6, color="#111827")
    annotate_speedups(ax, curves[0], curves[1], curves[2])
    ax.legend(
        loc="lower right",
        fontsize=8.2,
        frameon=False,
        handlelength=2.5,
        labelspacing=0.34,
    )


def make_overview(
    results_root: Path,
    seeds: tuple[str, ...],
    out_dir: Path,
    max_step: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(MODEL_ROWS),
        ncols=len(PANELS),
        figsize=(11.6, 6.9),
        sharex=True,
        sharey=True,
    )
    for row_index, (model, row_title) in enumerate(MODEL_ROWS):
        for column_index, (panel_title, series) in enumerate(PANELS):
            ax = axes[row_index][column_index]
            plot_panel(
                ax,
                results_root,
                seeds,
                model,
                panel_title,
                series,
                max_step,
            )
            if column_index == 0:
                ax.set_ylabel(f"{row_title}\nSuccess rate (%)", fontsize=10.8)
            else:
                ax.tick_params(labelleft=False)
            if row_index == len(MODEL_ROWS) - 1:
                ax.set_xlabel("Training step", fontsize=10.8)

    fig.suptitle(
        "WebShop: Effect of Online Oracle Evolution",
        fontsize=17,
        y=0.995,
        color="#111827",
    )
    fig.text(
        0.5,
        0.957,
        r"Mean $\pm$ 1 sample std over env seeds 0, 1, and 2",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#5B6472",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=1.8, w_pad=0.8)
    save_figure(
        fig,
        out_dir,
        "webshop_oracle_evolve_comparison_3seeds_overview",
        dpi,
    )
    plt.close(fig)


def make_row_figures(
    results_root: Path,
    seeds: tuple[str, ...],
    out_dir: Path,
    max_step: int,
    dpi: int,
) -> None:
    for model, row_title in MODEL_ROWS:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(PANELS),
            figsize=(11.6, 3.45),
            sharex=True,
            sharey=True,
        )
        for ax, (panel_title, series) in zip(axes, PANELS):
            plot_panel(
                ax,
                results_root,
                seeds,
                model,
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
            f"webshop_oracle_evolve_comparison_3seeds_{model}",
            dpi,
        )
        plt.close(fig)


def write_latex_snippet(out_dir: Path) -> None:
    snippet = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/webshop_oracle_evolve_comparison_3seeds_overview.pdf}
\caption{WebShop validation success-rate curves over environment seeds 0, 1, and 2. Lines show the mean and shaded regions show $\pm 1$ sample standard deviation. Each panel compares the base algorithm, ASFD with online oracle evolution, and ASFD with online oracle evolution disabled. Colored annotations report speedup at the 60-point target relative to the corresponding base algorithm.}
\label{fig:webshop_oracle_evolve_comparison}
\end{figure*}
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "webshop_oracle_evolve_comparison_3seeds_latex_snippet.tex"
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
