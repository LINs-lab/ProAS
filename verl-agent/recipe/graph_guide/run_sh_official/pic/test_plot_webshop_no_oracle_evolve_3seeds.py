from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import pytest


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


def load_plot_module():
    try:
        return importlib.import_module("plot_webshop_no_oracle_evolve_3seeds")
    except ImportError as exc:
        pytest.fail(f"WebShop comparison module is not implemented: {exc}")


def write_curve(path: Path, points: list[tuple[int, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wall_time", "step", "value"])
        for index, (step, value) in enumerate(points):
            writer.writerow([1000.0 + index, step, value])
    return path


def test_panels_include_base_evolve_and_no_evolve() -> None:
    module = load_plot_module()

    assert module.PANELS[0][1] == [
        ("GRPO", "grpo_exp/grpo"),
        ("GRPO+ASFD", "grpo_exp/grpo_ASFD"),
        (
            "GRPO+ASFD (No Evolve)",
            "grpo_exp/grpo_ASFD_no_evolve",
        ),
    ]
    assert module.PANELS[1][1][-1][1] == "dapo_exp/dapo_ASFD_no_evolve"
    assert module.PANELS[2][1][-1][1] == "gigpo_exp/gigpo_ASFD_no_evolve"


def test_speedup_rows_format_both_variants() -> None:
    module = load_plot_module()

    rows = module.speedup_rows(
        base_crossing=100.0,
        evolve_crossing=50.0,
        no_evolve_crossing=80.0,
    )

    assert rows == [("Evolve", "2.00x"), ("No Evolve", "1.25x")]


def test_speedup_rows_report_unreached_variant() -> None:
    module = load_plot_module()

    rows = module.speedup_rows(
        base_crossing=100.0,
        evolve_crossing=None,
        no_evolve_crossing=80.0,
    )

    assert rows == [("Evolve", "not reached"), ("No Evolve", "1.25x")]


def test_render_all_writes_overview_rows_and_latex(tmp_path: Path) -> None:
    module = load_plot_module()
    results_root = tmp_path / "results"
    out_dir = tmp_path / "figures"

    for seed_index, seed in enumerate(("seed0", "seed1", "seed2")):
        seed_offset = seed_index * 0.01
        for model in ("1_5B", "7B"):
            for _, series in module.PANELS:
                for method, relative_dir in series:
                    if method.endswith("(No Evolve)"):
                        values = [0.10, 0.50, 0.72]
                    elif method.endswith("+ASFD"):
                        values = [0.10, 0.55, 0.75]
                    else:
                        values = [0.10, 0.40, 0.65]
                    curve_path = (
                        results_root
                        / seed
                        / "main"
                        / model
                        / "weshop"
                        / relative_dir
                        / "val_success_rate.csv"
                    )
                    write_curve(
                        curve_path,
                        [
                            (0, values[0] + seed_offset),
                            (5, values[1] + seed_offset),
                            (10, values[2] + seed_offset),
                        ],
                    )

    module.render_all(
        results_root=results_root,
        out_dir=out_dir,
        max_step=10,
        dpi=40,
    )

    expected_stems = [
        "webshop_oracle_evolve_comparison_3seeds_overview",
        "webshop_oracle_evolve_comparison_3seeds_1_5B",
        "webshop_oracle_evolve_comparison_3seeds_7B",
    ]
    for stem in expected_stems:
        assert (out_dir / f"{stem}.pdf").stat().st_size > 0
        assert (out_dir / f"{stem}.png").stat().st_size > 0
    latex_path = out_dir / "webshop_oracle_evolve_comparison_3seeds_latex_snippet.tex"
    assert latex_path.stat().st_size > 0
