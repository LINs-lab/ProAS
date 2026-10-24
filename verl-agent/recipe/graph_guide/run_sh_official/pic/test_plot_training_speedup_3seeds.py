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
        return importlib.import_module("plot_training_speedup_3seeds")
    except ImportError as exc:
        pytest.fail(f"plotting module is not implemented: {exc}")


def write_curve(path: Path, points: list[tuple[int, float]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wall_time", "step", "value"])
        for index, (step, value) in enumerate(points):
            writer.writerow([1000.0 + index, step, value])
    return path


def test_aggregate_seed_curves_uses_percentage_sample_std(tmp_path: Path) -> None:
    module = load_plot_module()
    paths = [
        write_curve(tmp_path / "seed0.csv", [(0, 0.1), (5, 0.2)]),
        write_curve(tmp_path / "seed1.csv", [(0, 0.2), (5, 0.3)]),
        write_curve(tmp_path / "seed2.csv", [(0, 0.3), (5, 0.4)]),
    ]

    stats = module.aggregate_seed_curves(paths, max_step=5)

    assert stats.steps == [0, 5]
    assert stats.mean == pytest.approx([20.0, 30.0])
    assert stats.std == pytest.approx([10.0, 10.0])


def test_read_curve_rejects_duplicate_steps(tmp_path: Path) -> None:
    module = load_plot_module()
    path = write_curve(tmp_path / "duplicate.csv", [(0, 0.1), (0, 0.2)])

    with pytest.raises(module.CurveDataError, match="duplicate step 0"):
        module.read_curve(path, max_step=5)


def test_aggregate_seed_curves_rejects_mismatched_step_grids(
    tmp_path: Path,
) -> None:
    module = load_plot_module()
    paths = [
        write_curve(tmp_path / "seed0.csv", [(0, 0.1), (5, 0.2)]),
        write_curve(tmp_path / "seed1.csv", [(0, 0.2), (10, 0.3)]),
        write_curve(tmp_path / "seed2.csv", [(0, 0.3), (5, 0.4)]),
    ]

    with pytest.raises(module.CurveDataError, match="step grid differs"):
        module.aggregate_seed_curves(paths, max_step=10)


def test_crossing_step_interpolates_between_recorded_points() -> None:
    module = load_plot_module()

    crossing = module.crossing_step([0, 5, 10], [50.0, 60.0, 80.0], 70.0)

    assert crossing == pytest.approx(7.5)


def test_crossing_step_returns_none_when_target_is_not_reached() -> None:
    module = load_plot_module()

    assert module.crossing_step([0, 5, 10], [10.0, 20.0, 30.0], 70.0) is None


def test_format_speedup_does_not_describe_slowdown_as_faster() -> None:
    module = load_plot_module()

    assert module.format_speedup(0.86) == "0.86x speedup (slower)"
    assert module.format_speedup(1.42) == "1.42x faster"


def test_speedup_label_is_kept_above_the_threshold() -> None:
    module = load_plot_module()
    crowded_curve = module.CurveStats(
        steps=[0, 50, 100],
        mean=[60.0, 75.0, 90.0],
        std=[5.0, 5.0, 5.0],
    )

    label_y = module.choose_speedup_label_y(
        [crowded_curve], x_mid=50.0, threshold=70.0
    )

    assert label_y >= 70.0


def test_render_all_writes_overview_rows_and_latex(tmp_path: Path) -> None:
    module = load_plot_module()
    results_root = tmp_path / "results"
    out_dir = tmp_path / "figures"

    for seed_index, seed in enumerate(module.DEFAULT_SEEDS):
        seed_offset = seed_index * 0.01
        for model, env, _ in module.MODEL_ENV_ROWS:
            for _, series in module.PANELS:
                for method, relative_dir in series:
                    if method.endswith("+ASFD"):
                        values = [0.10, 0.72, 0.90]
                    else:
                        values = [0.10, 0.52, 0.82]
                    points = [
                        (0, values[0] + seed_offset),
                        (5, values[1] + seed_offset),
                        (10, values[2] + seed_offset),
                    ]
                    curve_path = (
                        results_root
                        / seed
                        / "main"
                        / model
                        / env
                        / relative_dir
                        / "val_success_rate.csv"
                    )
                    curve_path.parent.mkdir(parents=True, exist_ok=True)
                    write_curve(curve_path, points)

    module.render_all(
        results_root=results_root,
        out_dir=out_dir,
        max_step=10,
        dpi=40,
    )

    expected_stems = [
        "training_speedup_3seeds_overview",
        "training_speedup_3seeds_1_5B_alfworld",
        "training_speedup_3seeds_1_5B_weshop",
        "training_speedup_3seeds_7B_alfworld",
        "training_speedup_3seeds_7B_weshop",
    ]
    for stem in expected_stems:
        assert (out_dir / f"{stem}.pdf").stat().st_size > 0
        assert (out_dir / f"{stem}.png").stat().st_size > 0
    assert (out_dir / "training_speedup_3seeds_latex_snippet.tex").stat().st_size > 0
