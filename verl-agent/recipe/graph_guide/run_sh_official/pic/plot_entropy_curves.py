#!/usr/bin/env python3
"""Plot policy entropy curves for ALFWorld and WebShop runs.

The script reads actor_entropy_loss.csv files from results/main and generates
paper-ready entropy figures under run_sh_official/pic/figures.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path("/mnt/public/dingbowen/project_sft_rl_agenetic_training")
DEFAULT_MAIN_DIR = REPO / "verl-agent/recipe/graph_guide/results/main"
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
            values.append(float(row["value"]))
    return steps, values


def style_axes(ax: plt.Axes, max_step: int) -> None:
    ax.set_xlim(0, max_step)
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    ax.grid(True, color="#E5E7EB", linewidth=0.6, alpha=0.9)
    ax.tick_params(labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")


def plot_panel(
    ax: plt.Axes,
    main_dir: Path,
    model: str,
    env: str,
    panel_title: str,
    series: list[tuple[str, str]],
    max_step: int,
) -> None:
    values_for_ylim: list[float] = []
    for method, rel_dir in series:
        path = main_dir / model / env / rel_dir / "actor_entropy_loss.csv"
        if not path.exists():
            print(f"[WARN] missing curve: {path}")
            continue
        steps, values = read_curve(path, max_step=max_step)
        if not steps:
            print(f"[WARN] empty curve within step <= {max_step}: {path}")
            continue
        values_for_ylim.extend(values)
        ax.plot(
            steps,
            values,
            label=LABELS.get(method, method),
            color=COLORS.get(method, "#111827"),
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=2.0 if method.endswith("+ASFD") else 1.7,
            alpha=0.98,
        )

    if values_for_ylim:
        low = min(values_for_ylim)
        high = max(values_for_ylim)
        pad = max(0.08, (high - low) * 0.12)
        ax.set_ylim(max(0.0, low - pad), high + pad)
    style_axes(ax, max_step)
    ax.set_title(panel_title, fontsize=13, pad=6)
    ax.legend(loc="best", fontsize=9.5, frameon=False, handlelength=2.4)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    print(f"[INFO] wrote {pdf}")
    print(f"[INFO] wrote {png}")


def make_overview(main_dir: Path, out_dir: Path, max_step: int, dpi: int) -> None:
    fig, axes = plt.subplots(
        nrows=len(MODEL_ENV_ROWS),
        ncols=len(PANELS),
        figsize=(10.8, 11.0),
        sharex=True,
        sharey=False,
    )
    for row_idx, (model, env, row_title) in enumerate(MODEL_ENV_ROWS):
        for col_idx, (panel_title, series) in enumerate(PANELS):
            ax = axes[row_idx][col_idx]
            plot_panel(
                ax=ax,
                main_dir=main_dir,
                model=model,
                env=env,
                panel_title=panel_title,
                series=series,
                max_step=max_step,
            )
            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\nPolicy entropy", fontsize=12)
            if row_idx == len(MODEL_ENV_ROWS) - 1:
                ax.set_xlabel("Training step", fontsize=12)
    fig.suptitle("Policy Entropy Curves", fontsize=17, y=0.995)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    save_figure(fig, out_dir, "training_entropy_overview", dpi)
    plt.close(fig)


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
    make_overview(args.main_dir, args.out_dir, args.max_step, args.dpi)


if __name__ == "__main__":
    main()
