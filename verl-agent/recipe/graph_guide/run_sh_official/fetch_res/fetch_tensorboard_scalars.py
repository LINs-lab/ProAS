#!/usr/bin/env python3
"""Fetch selected TensorBoard scalar metrics and save them as CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import urlopen


DEFAULT_TENSORBOARD_URL = "http://localhost:6014"
RESULT_ROOT = (
    "/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent/"
    "recipe/graph_guide/results/main/7B"
)
RESULT_ROOT_1_5B = (
    "/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent/"
    "recipe/graph_guide/results/main/1_5B"
)
ABLATION_RESULT_ROOT_1_5B = (
    "/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent/"
    "recipe/graph_guide/results/ablation/1_5B"
)


ALFWORLD_TAGS: list[str] = [
    "val/look_at_obj_in_light_success_rate",
    "val/pick_and_place_success_rate",
    "val/pick_clean_then_place_in_recep_success_rate",
    "val/pick_cool_then_place_in_recep_success_rate",
    "val/pick_heat_then_place_in_recep_success_rate",
    "val/pick_two_obj_and_place_success_rate",
    "val/success_rate",
    "val/text/test_score",
    "actor/entropy_loss",
]

WESHOP_TAGS: list[str] = [
    "val/success_rate",
    "val/text/test_score",
    "val/webshop_task_score",
    "actor/entropy_loss",
]

TAG_ALIASES: dict[str, list[str]] = {
    # TensorBoard currently records this WebShop metric with the explanatory
    # suffix in the tag name; keep the shorter requested name as the CSV label.
    "val/webshop_task_score": [
        "val/webshop_task_score",
        "val/webshop_task_score (not success_rate)",
    ],
}


@dataclass(frozen=True)
class ExportJob:
    run: str
    output_dir: str
    tags: list[str]
    tb_url: str | None = None
    job_id: str | None = None


RUN_ALIASES: dict[str, list[str]] = {
    # The run script is named gigpo_webshop.sh and the current TensorBoard
    # service exposes this run as weshop/gigpo_webshop.
    "weshop/gigpo_weshop": ["weshop/gigpo_weshop", "weshop/gigpo_webshop"],
    "weshop/dapo_graph_guide_step_future_miss_no_dis_mask_weshop": [
        "weshop/dapo_graph_guide_step_future_miss_no_dis_mask_weshop",
        "dapo_graph_guide_step_future_miss_no_dis_mask_weshop",
    ],
}


EXPORT_JOBS: list[ExportJob] = [
    ExportJob(
        run="alfworld/gigpo_graph_guide_step_future_miss_no_dis_mask_alfworld",
        output_dir=f"{RESULT_ROOT}/alfworld/gigpo_exp/gigpo_ASFD",
        tags=ALFWORLD_TAGS,
    ),
    ExportJob(
        run="alfworld/gigpo_alfworld",
        output_dir=f"{RESULT_ROOT}/alfworld/gigpo_exp/gigpo",
        tags=ALFWORLD_TAGS,
    ),
    ExportJob(
        run="alfworld/grpo_alfworld",
        output_dir=f"{RESULT_ROOT}/alfworld/grpo_exp/grpo",
        tags=ALFWORLD_TAGS,
    ),
    ExportJob(
        run="alfworld/graph_guide_step_future_miss_no_dis_mask_alfworld",
        output_dir=f"{RESULT_ROOT}/alfworld/grpo_exp/grpo_ASFD",
        tags=ALFWORLD_TAGS,
    ),
    ExportJob(
        run="alfworld/rewardflow_alfworld_T50_gamma_0_9",
        output_dir=f"{RESULT_ROOT}/alfworld/grpo_exp/grpo_rewardflow",
        tags=ALFWORLD_TAGS,
    ),
    ExportJob(
        run="alfworld/dapo_alfworld",
        output_dir=f"{RESULT_ROOT}/alfworld/dapo_exp/dapo",
        tags=ALFWORLD_TAGS,
    ),
    ExportJob(
        run="alfworld/dapo_graph_guide_step_future_miss_no_dis_mask_alfworld",
        output_dir=f"{RESULT_ROOT}/alfworld/dapo_exp/dapo_ASFD",
        tags=ALFWORLD_TAGS,
    ),
    ExportJob(
        run="rewardflow_alfworld_T_50",
        output_dir=f"{RESULT_ROOT_1_5B}/alfworld/grpo_exp/grpo_rewardflow",
        tags=ALFWORLD_TAGS,
        tb_url="http://localhost:6013",
    ),
    ExportJob(
        run="grpo_alfworld",
        output_dir=f"{RESULT_ROOT_1_5B}/alfworld/grpo_exp/grpo",
        tags=ALFWORLD_TAGS,
        tb_url="http://localhost:6016",
    ),
    ExportJob(
        run="graph_guide_step_future_miss_no_dis_mask_alfworld",
        output_dir=f"{RESULT_ROOT_1_5B}/alfworld/grpo_exp/grpo_ASFD",
        tags=ALFWORLD_TAGS,
        tb_url="http://localhost:6016",
    ),
    ExportJob(
        run="gigpo_alfworld",
        output_dir=f"{RESULT_ROOT_1_5B}/alfworld/gigpo_exp/gigpo",
        tags=ALFWORLD_TAGS,
        tb_url="http://localhost:6016",
    ),
    ExportJob(
        run="gigpo_graph_guide_step_future_miss_no_dis_mask_alfworld",
        output_dir=f"{RESULT_ROOT_1_5B}/alfworld/gigpo_exp/gigpo_ASFD",
        tags=ALFWORLD_TAGS,
        tb_url="http://localhost:6016",
    ),
    ExportJob(
        run="dapo_alfworld",
        output_dir=f"{RESULT_ROOT_1_5B}/alfworld/dapo_exp/dapo",
        tags=ALFWORLD_TAGS,
        tb_url="http://localhost:6016",
    ),
    ExportJob(
        run="dapo_graph_guide_step_future_miss_no_dis_mask_alfworld",
        output_dir=f"{RESULT_ROOT_1_5B}/alfworld/dapo_exp/dapo_ASFD",
        tags=ALFWORLD_TAGS,
        tb_url="http://localhost:6016",
    ),
    ExportJob(
        run="weshop/dapo_weshop",
        output_dir=f"{RESULT_ROOT_1_5B}/weshop/dapo_exp/dapo",
        tags=WESHOP_TAGS,
        tb_url="http://localhost:6016",
        job_id="1_5B/weshop/dapo_weshop",
    ),
    ExportJob(
        run="weshop/dapo_graph_guide_step_future_miss_no_dis_mask_weshop",
        output_dir=f"{RESULT_ROOT_1_5B}/weshop/dapo_exp/dapo_ASFD",
        tags=WESHOP_TAGS,
        tb_url="http://localhost:6018",
        job_id="1_5B/weshop/dapo_graph_guide_step_future_miss_no_dis_mask_weshop",
    ),
    ExportJob(
        run="weshop/gigpo_webshop",
        output_dir=f"{RESULT_ROOT_1_5B}/weshop/gigpo_exp/gigpo",
        tags=WESHOP_TAGS,
        tb_url="http://localhost:6016",
        job_id="1_5B/weshop/gigpo_webshop",
    ),
    ExportJob(
        run="weshop/gigpo_graph_guide_step_future_miss_no_dis_mask_weshop",
        output_dir=f"{RESULT_ROOT_1_5B}/weshop/gigpo_exp/gigpo_ASFD",
        tags=WESHOP_TAGS,
        tb_url="http://localhost:6016",
        job_id="1_5B/weshop/gigpo_graph_guide_step_future_miss_no_dis_mask_weshop",
    ),
    ExportJob(
        run="weshop/grpo_webshop",
        output_dir=f"{RESULT_ROOT_1_5B}/weshop/grpo_exp/grpo",
        tags=WESHOP_TAGS,
        tb_url="http://localhost:6016",
        job_id="1_5B/weshop/grpo_webshop",
    ),
    ExportJob(
        run="weshop/graph_guide_step_future_miss_no_dis_mask_weshop",
        output_dir=f"{RESULT_ROOT_1_5B}/weshop/grpo_exp/grpo_ASFD",
        tags=WESHOP_TAGS,
        tb_url="http://localhost:6016",
        job_id="1_5B/weshop/graph_guide_step_future_miss_no_dis_mask_weshop",
    ),
    ExportJob(
        run="rewardflow_webshop_2",
        output_dir=f"{RESULT_ROOT_1_5B}/weshop/grpo_exp/grpo_rewardflow",
        tags=WESHOP_TAGS,
        tb_url="http://localhost:6013",
        job_id="1_5B/weshop/rewardflow_webshop_2",
    ),
    ExportJob(
        run="grpo_alfworld",
        output_dir=f"{ABLATION_RESULT_ROOT_1_5B}/alfworld/grpo",
        tags=ALFWORLD_TAGS,
        tb_url="http://localhost:6016",
        job_id="ablation/1_5B/alfworld/grpo",
    ),
    ExportJob(
        run="graph_guide_step_future_miss_no_dis_mask_alfworld",
        output_dir=f"{ABLATION_RESULT_ROOT_1_5B}/alfworld/grpo_ASFD",
        tags=ALFWORLD_TAGS,
        tb_url="http://localhost:6016",
        job_id="ablation/1_5B/alfworld/grpo_ASFD",
    ),
    ExportJob(
        run="graph_guide_random_hit_step_future_miss_no_dis_mask_alfworld",
        output_dir=f"{ABLATION_RESULT_ROOT_1_5B}/alfworld/random_miss",
        tags=ALFWORLD_TAGS,
        tb_url="http://localhost:6016",
        job_id="ablation/1_5B/alfworld/random_miss",
    ),
    ExportJob(
        run="graph_guide_step_future_miss_no_dis_mask_no_oracle_evolve_alfworld",
        output_dir=f"{ABLATION_RESULT_ROOT_1_5B}/alfworld/no_evolve",
        tags=ALFWORLD_TAGS,
        tb_url="http://localhost:6016",
        job_id="ablation/1_5B/alfworld/no_evolve",
    ),
    ExportJob(
        run="graph_guide_step_future_miss_no_pos_mask_alfworld",
        output_dir=f"{ABLATION_RESULT_ROOT_1_5B}/alfworld/no_pos_protect",
        tags=ALFWORLD_TAGS,
        tb_url="http://localhost:6016",
        job_id="ablation/1_5B/alfworld/no_pos_protect",
    ),
    ExportJob(
        run="grpo_webshop",
        output_dir=f"{ABLATION_RESULT_ROOT_1_5B}/weshop/grpo",
        tags=WESHOP_TAGS,
        tb_url="http://localhost:6018",
        job_id="ablation/1_5B/weshop/grpo",
    ),
    ExportJob(
        run="graph_guide_step_future_miss_no_dis_mask_weshop",
        output_dir=f"{ABLATION_RESULT_ROOT_1_5B}/weshop/grpo_ASFD",
        tags=WESHOP_TAGS,
        tb_url="http://localhost:6018",
        job_id="ablation/1_5B/weshop/grpo_ASFD",
    ),
    ExportJob(
        run="graph_guide_random_hit_step_future_miss_no_dis_mask_weshop",
        output_dir=f"{ABLATION_RESULT_ROOT_1_5B}/weshop/random_miss",
        tags=WESHOP_TAGS,
        tb_url="http://localhost:6018",
        job_id="ablation/1_5B/weshop/random_miss",
    ),
    ExportJob(
        run="graph_guide_step_future_miss_no_dis_mask_no_oracle_evolve_weshop",
        output_dir=f"{ABLATION_RESULT_ROOT_1_5B}/weshop/no_evolve",
        tags=WESHOP_TAGS,
        tb_url="http://localhost:6018",
        job_id="ablation/1_5B/weshop/no_evolve",
    ),
    ExportJob(
        run="graph_guide_step_future_miss_no_pos_mask_weshop",
        output_dir=f"{ABLATION_RESULT_ROOT_1_5B}/weshop/no_pos_protect",
        tags=WESHOP_TAGS,
        tb_url="http://localhost:6018",
        job_id="ablation/1_5B/weshop/no_pos_protect",
    ),
    ExportJob(
        run="weshop/dapo_graph_guide_step_future_miss_no_dis_mask_weshop",
        output_dir=f"{RESULT_ROOT}/weshop/dapo_exp/dapo_ASFD",
        tags=WESHOP_TAGS,
    ),
    ExportJob(
        run="weshop/dapo_weshop",
        output_dir=f"{RESULT_ROOT}/weshop/dapo_exp/dapo",
        tags=WESHOP_TAGS,
    ),
    ExportJob(
        run="weshop/gigpo_weshop",
        output_dir=f"{RESULT_ROOT}/weshop/gigpo_exp/gigpo",
        tags=WESHOP_TAGS,
    ),
    ExportJob(
        run="weshop/gigpo_graph_guide_step_future_miss_no_dis_mask_weshop",
        output_dir=f"{RESULT_ROOT}/weshop/gigpo_exp/gigpo_ASFD",
        tags=WESHOP_TAGS,
    ),
    ExportJob(
        run="weshop/grpo_webshop",
        output_dir=f"{RESULT_ROOT}/weshop/grpo_exp/grpo",
        tags=WESHOP_TAGS,
    ),
    ExportJob(
        run="weshop/graph_guide_step_future_miss_no_dis_mask_weshop",
        output_dir=f"{RESULT_ROOT}/weshop/grpo_exp/grpo_ASFD",
        tags=WESHOP_TAGS,
    ),
    ExportJob(
        run="weshop/rewardflow_webshop_T15_gamma_0_9",
        output_dir=f"{RESULT_ROOT}/weshop/grpo_exp/grpo_rewardflow",
        tags=WESHOP_TAGS,
    ),
]


class TensorBoardClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    def get_json(self, path: str, params: dict[str, str] | None = None) -> object:
        url = urljoin(self.base_url, path.lstrip("/"))
        if params:
            url = f"{url}?{urlencode(params)}"
        try:
            with urlopen(url, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} while fetching {url}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach TensorBoard at {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"TensorBoard returned invalid JSON for {url}") from exc

    def runs(self) -> set[str]:
        data = self.get_json("/data/runs")
        if not isinstance(data, list):
            raise RuntimeError("Unexpected /data/runs response; expected a JSON list")
        return {str(run) for run in data}

    def scalar_tags(self) -> dict[str, set[str]]:
        data = self.get_json("/data/plugin/scalars/tags")
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected scalar tags response; expected a JSON object")
        return {
            str(run): set(str(tag) for tag in tags.keys())
            for run, tags in data.items()
            if isinstance(tags, dict)
        }

    def scalars(self, run: str, tag: str) -> list[tuple[float, int, float]]:
        data = self.get_json(
            "/data/plugin/scalars/scalars",
            {"run": run, "tag": tag, "format": "json"},
        )
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected scalar data for run={run!r}, tag={tag!r}")

        rows: list[tuple[float, int, float]] = []
        for item in data:
            if not isinstance(item, list) or len(item) != 3:
                raise RuntimeError(f"Malformed scalar point for run={run!r}, tag={tag!r}: {item!r}")
            wall_time, step, value = item
            rows.append((float(wall_time), int(step), float(value)))
        return rows


def resolve_run(requested_run: str, available_runs: set[str]) -> str | None:
    candidates = RUN_ALIASES.get(requested_run, [requested_run])
    for candidate in candidates:
        if candidate in available_runs:
            return candidate
    return None


def csv_metric_name(tag: str) -> str:
    name = tag.strip().replace("/", "__")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "metric"


def write_csv(path: str, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def resolve_tag(requested_tag: str, available_tags: set[str]) -> str | None:
    candidates = TAG_ALIASES.get(requested_tag, [requested_tag])
    for candidate in candidates:
        if candidate in available_tags:
            return candidate
    return None


def export_job(
    client: TensorBoardClient,
    job: ExportJob,
    actual_run: str,
    available_tags: set[str],
    dry_run: bool,
) -> tuple[int, int, list[str]]:
    os.makedirs(job.output_dir, exist_ok=True)
    missing: list[str] = []
    metric_data: dict[str, tuple[str, list[tuple[float, int, float]]]] = {}

    for requested_tag in job.tags:
        actual_tag = resolve_tag(requested_tag, available_tags)
        if actual_tag is None:
            missing.append(requested_tag)
            continue
        if dry_run:
            metric_data[requested_tag] = (actual_tag, [])
            continue
        metric_data[requested_tag] = (actual_tag, client.scalars(actual_run, actual_tag))

    if dry_run:
        return len(metric_data), 0, missing

    long_rows: list[list[object]] = []
    steps: set[int] = set()
    by_step: dict[str, dict[int, tuple[float, float]]] = {}

    for requested_tag, (actual_tag, rows) in metric_data.items():
        metric_file = os.path.join(job.output_dir, f"{csv_metric_name(requested_tag)}.csv")
        write_csv(metric_file, ["wall_time", "step", "value"], rows)

        by_step[requested_tag] = {}
        for wall_time, step, value in rows:
            long_rows.append([job.run, actual_run, requested_tag, actual_tag, wall_time, step, value])
            steps.add(step)
            by_step[requested_tag][step] = (wall_time, value)

    long_rows.sort(key=lambda row: (str(row[2]), int(row[5])))
    write_csv(
        os.path.join(job.output_dir, "tensorboard_scalars_long.csv"),
        ["requested_run", "actual_run", "requested_tag", "actual_tag", "wall_time", "step", "value"],
        long_rows,
    )

    wide_header = ["step"]
    for requested_tag in job.tags:
        if requested_tag in by_step:
            wide_header.append(requested_tag)

    wide_rows: list[list[object]] = []
    for step in sorted(steps):
        row: list[object] = [step]
        for requested_tag in job.tags:
            if requested_tag not in by_step:
                continue
            point = by_step[requested_tag].get(step)
            row.append("" if point is None else point[1])
        wide_rows.append(row)

    write_csv(os.path.join(job.output_dir, "tensorboard_scalars_wide.csv"), wide_header, wide_rows)

    metadata = {
        "requested_run": job.run,
        "actual_run": actual_run,
        "tensorboard_url": client.base_url.rstrip("/"),
        "exported_at_unix": time.time(),
        "output_dir": job.output_dir,
        "metrics": [
            {
                "requested_tag": requested_tag,
                "actual_tag": metric_data[requested_tag][0],
                "points": len(metric_data[requested_tag][1]),
                "csv": f"{csv_metric_name(requested_tag)}.csv",
            }
            for requested_tag in job.tags
            if requested_tag in metric_data
        ],
        "missing_tags": missing,
    }
    with open(os.path.join(job.output_dir, "tensorboard_export_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    total_points = sum(len(rows) for _, rows in metric_data.values())
    return len(metric_data), total_points, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch selected TensorBoard scalar metrics for configured AlfWorld/WebShop runs.",
    )
    parser.add_argument(
        "--tb-url",
        default=os.environ.get("TENSORBOARD_URL", DEFAULT_TENSORBOARD_URL),
        help=f"TensorBoard base URL. Default: {DEFAULT_TENSORBOARD_URL}",
    )
    parser.add_argument(
        "--only-run",
        action="append",
        default=[],
        help="Export only this TensorBoard run or configured job id. Can be provided multiple times.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for each TensorBoard request.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check run/tag availability without writing scalar CSV data.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any configured run or tag is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    only_runs = set(args.only_run)
    jobs = [
        job
        for job in EXPORT_JOBS
        if not only_runs or job.run in only_runs or (job.job_id is not None and job.job_id in only_runs)
    ]
    if not jobs:
        print("ERROR: no configured export job matches --only-run", file=sys.stderr)
        return 1

    client_cache: dict[str, tuple[TensorBoardClient, set[str], dict[str, set[str]]]] = {}

    def get_client_data(tb_url: str) -> tuple[TensorBoardClient, set[str], dict[str, set[str]]]:
        normalized_url = tb_url.rstrip("/")
        if normalized_url not in client_cache:
            client = TensorBoardClient(normalized_url, args.timeout)
            client_cache[normalized_url] = (client, client.runs(), client.scalar_tags())
        return client_cache[normalized_url]

    had_missing = False
    for job in jobs:
        tb_url = job.tb_url or args.tb_url
        try:
            client, runs, scalar_tags = get_client_data(tb_url)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        actual_run = resolve_run(job.run, runs)
        if actual_run is None:
            had_missing = True
            print(f"[MISSING RUN] {job.run} on {tb_url}", file=sys.stderr)
            continue

        try:
            metric_count, point_count, missing_tags = export_job(
                client=client,
                job=job,
                actual_run=actual_run,
                available_tags=scalar_tags.get(actual_run, set()),
                dry_run=args.dry_run,
            )
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        if missing_tags:
            had_missing = True
            print(f"[MISSING TAGS] {job.run}: {', '.join(missing_tags)}", file=sys.stderr)

        action = "checked" if args.dry_run else "exported"
        run_note = job.run if actual_run == job.run else f"{job.run} (using {actual_run})"
        print(
            f"[{action}] {run_note} from {client.base_url.rstrip('/')} -> "
            f"{job.output_dir} ({metric_count} metrics, {point_count} points)"
        )

    if args.strict and had_missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
