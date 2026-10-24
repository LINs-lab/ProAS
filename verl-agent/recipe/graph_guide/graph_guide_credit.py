from __future__ import annotations

import json
import hashlib
import importlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from verl import DataProto


RECIPE_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECIPE_DIR.parents[1]
DEFAULT_ORACLE_INDEX = (
    RECIPE_DIR
    / "oracle_build"
    / "results"
    / "oracle_graphs_textworld_train"
    / "oracle_graphs"
    / "index.json"
)

GRAPH_GUIDE_DUMP_KEYS = [
    "graph_guide_task_key",
    "graph_guide_reasonable",
    "graph_guide_reason",
    "graph_guide_hit_source",
    "graph_guide_spit",
    "graph_guide_progress_reward",
    "graph_guide_reward_before_progress",
    "graph_guide_spit_num",
    "graph_guide_spit_den",
    "graph_guide_delta_spit",
    "graph_guide_delta_spit_num",
    "graph_guide_progress_delta",
    "graph_guide_hit",
    "graph_guide_avg_future_miss",
    "graph_guide_step_future_miss",
    "graph_guide_turn_future_miss",
    "graph_guide_distance_available",
    "graph_guide_d_anchor",
    "graph_guide_d_next",
    "graph_guide_graph_step_miss",
    "graph_guide_graph_future_miss",
    "graph_guide_graph_step_damage",
    "graph_guide_graph_future_damage",
    "graph_guide_milestone_hit",
    "graph_guide_milestone_segment_id",
    "graph_guide_milestone_next_hit_distance",
    "graph_guide_milestone_step_miss",
    "graph_guide_milestone_future_miss",
    "graph_guide_damage",
    "graph_guide_step_regret",
    "graph_guide_blame_score",
    "graph_guide_scale",
    "graph_guide_adv_mask",
    "graph_guide_new_eligible",
    "graph_guide_row_adv",
    "graph_guide_adv_sign",
    "graph_guide_adv_cluster_key",
    "graph_guide_adv_cluster_size",
    "graph_guide_positive_mask_applied",
    "graph_guide_negative_reweight_applied",
    "graph_guide_cluster_mean_blame",
    "graph_guide_traj_adv",
    "graph_guide_oracle_key",
    "graph_guide_oracle_missing",
    "graph_guide_reference_path",
    "graph_guide_from_signature",
    "graph_guide_to_signature",
    "graph_guide_from_signature_hash",
    "graph_guide_to_signature_hash",
    "graph_guide_matched_subgoals",
    "graph_guide_matched_substeps",
    "graph_guide_covered_options",
    "graph_guide_purchased_asin",
    "graph_guide_purchased_options",
]


def _load_adapter(env_adapter: str):
    adapter = str(env_adapter or "alfworld").lower()
    if adapter in {"alfworld", "textworld", "alfred"}:
        return importlib.import_module("recipe.graph_guide.oracle_build.alfworld_reference_graph")
    if adapter == "webshop":
        return importlib.import_module("recipe.graph_guide.oracle_build.webshop_reference_graph")
    raise ValueError(f"unknown graph_guide.env_adapter={env_adapter!r}")


@dataclass(frozen=True)
class GraphGuideConfig:
    enabled: bool = False
    progress_reward_enabled: bool = False
    progress_reward_beta: float = 0.5
    env_adapter: str = "alfworld"
    annotation_source: str = "oracle"
    oracle_index_path: str = str(DEFAULT_ORACLE_INDEX)
    signature_mode: str = "entity"
    beta: float = 1.0
    normalize_blame: bool = True
    strict_missing_oracle: bool = True
    online_oracle_update: bool = False
    online_max_paths_per_task: int = 32
    disadvantage_mode: str = "trajectory"
    blame_source: str = "step_regret"
    blame_normalization: str = "mean"
    scale_mode: str = "blame"
    constant_scale: float = 1.0
    softmax_temperature: float = 1.0
    adv_cluster_round_digits: int = 6
    mask_positive_unreasonable: bool = False
    mask_positive_unreasonable_mode: str = "advantage"
    mask_negative_reasonable: bool = True
    negative_mask_mode: str = "reasonable"
    negative_mask_scope: str = "episode_failure"
    scale_eligible_mode: str = "unreasonable"
    reweight_disadvantage_unreasonable: bool = True
    eps: float = 1e-8
    hit_source: str = "oracle"
    random_hit_prob: float = 0.5
    random_hit_seed: int = 0
    migpo_reasonable_source: str = "next_hit_segment"
    migpo_threshold: float = 0.95


def graph_guide_config(config: DictConfig) -> GraphGuideConfig:
    section = OmegaConf.select(config, "algorithm.graph_guide")
    if section is None:
        return GraphGuideConfig()
    values = OmegaConf.to_container(section, resolve=True)
    if not isinstance(values, Mapping):
        return GraphGuideConfig()
    allowed = set(GraphGuideConfig.__dataclass_fields__)
    kwargs = {key: value for key, value in values.items() if key in allowed}
    configured_hit_source = kwargs.get("hit_source", "oracle")
    hit_source = str(configured_hit_source or "oracle").lower()
    if hit_source not in {"oracle", "random"}:
        raise ValueError(
            f"unknown graph_guide.hit_source={configured_hit_source!r}; expected 'oracle' or 'random'"
        )
    kwargs["hit_source"] = hit_source
    return GraphGuideConfig(**kwargs)


def graph_guide_enabled(config: DictConfig) -> bool:
    return graph_guide_config(config).enabled


def _use_random_hit(cfg: GraphGuideConfig) -> bool:
    return str(cfg.hit_source or "oracle").lower() == "random"


def _use_migpo_native(cfg: GraphGuideConfig) -> bool:
    adapter = str(cfg.env_adapter or "").lower()
    source = str(cfg.annotation_source or "").lower()
    return adapter in {"migpo_native", "migpo-milestone", "migpo_milestone"} or source in {
        "migpo_milestone",
        "migpo-native",
        "migpo_native",
    }


def _deterministic_random01(*parts: Any) -> float:
    payload = json.dumps([str(part) for part in parts], ensure_ascii=False, sort_keys=True)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _randomize_hit_annotations(
    annotations: Sequence[Mapping[str, Any]],
    *,
    cfg: GraphGuideConfig,
    traj_uid: str,
) -> List[Dict[str, Any]]:
    p = min(1.0, max(0.0, float(cfg.random_hit_prob)))
    randomized = [dict(ann) for ann in annotations]
    for local_idx, ann in enumerate(randomized):
        step_idx = int(ann.get("traj_step_idx", local_idx) or local_idx)
        value = _deterministic_random01(cfg.random_hit_seed, traj_uid, step_idx, local_idx)
        hit = 1 if value < p else 0
        ann["hit"] = hit
        ann["reasonable"] = hit
        ann["reason"] = f"random_hit_{hit}"

    misses = [1 - int(ann.get("hit", 0) or 0) for ann in randomized]
    turn_future_miss = float(sum(misses))
    dis_indices = [idx for idx, ann in enumerate(randomized) if int(ann.get("reasonable", 0) or 0) == 0]
    dis_set = set(dis_indices)
    regrets = [float(ann.get("step_regret", ann.get("turn_success_damage", 0.0)) or 0.0) for ann in randomized]
    mean_regret = sum(regrets[idx] for idx in dis_indices) / len(dis_indices) if dis_indices else 0.0
    for idx, ann in enumerate(randomized):
        future_miss = float(sum(misses[idx:]))
        future_tail = len(misses[idx:])
        ann["step_future_miss"] = future_miss
        ann["turn_future_miss"] = turn_future_miss
        ann["avg_future_miss_num"] = future_miss
        ann["avg_future_miss_den"] = future_tail
        ann["avg_future_miss"] = future_miss / float(future_tail) if future_tail else 0.0
        if idx not in dis_set:
            ann["normalized_step_regret"] = 0.0
        elif mean_regret <= 0.0:
            ann["normalized_step_regret"] = 1.0 / float(len(dis_indices)) if dis_indices else 0.0
        else:
            ann["normalized_step_regret"] = regrets[idx] / mean_regret
    return randomized


class OracleGraphCache:
    def __init__(
        self,
        index_path: str | Path,
        strict_missing: bool = True,
        env_adapter: str = "alfworld",
    ):
        self.index_path = Path(index_path).expanduser()
        if not self.index_path.is_absolute():
            self.index_path = (REPO_ROOT / self.index_path).resolve()
        self.strict_missing = strict_missing
        self.env_adapter = str(env_adapter or "alfworld").lower()
        self.adapter = _load_adapter(self.env_adapter)
        self.output_dir = self.index_path.parent.parent
        self.index = json.loads(self.index_path.read_text())
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._merge_counts: Dict[str, int] = defaultdict(int)

    def task_key_for_row(self, row: Mapping[str, Any]) -> str:
        if self.env_adapter == "webshop":
            task_desc = str(row.get("task_desc") or "")
            if not task_desc:
                return ""
            return self.adapter.task_key(task_desc)
        gamefile = str(row.get("gamefile") or "")
        if not gamefile:
            return ""
        return self.adapter.task_key_from_gamefile(gamefile)

    def get(self, row: Mapping[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        task_key = self.task_key_for_row(row)
        if not task_key:
            if self.strict_missing:
                raise KeyError(f"empty task key for graph-guided credit adapter={self.env_adapter}")
            return None, None
        if task_key in self._cache:
            return task_key, self._cache[task_key]

        item = self.index.get("graphs", {}).get(task_key)
        if item is None or "path" not in item:
            if self.strict_missing:
                raise KeyError(f"oracle graph missing for adapter={self.env_adapter} task_key={task_key}")
            return task_key, None

        graph_path = self.output_dir / str(item["path"])
        oracle = json.loads(graph_path.read_text())
        self._cache[task_key] = oracle
        return task_key, oracle

    def maybe_merge_success_path(
        self,
        task_key: str,
        oracle: Dict[str, Any],
        rollout_rows: Sequence[Dict[str, Any]],
        signature_mode: str,
        traj_uid: str,
        max_paths_per_task: int,
    ) -> bool:
        """Add a successful rollout as an in-memory alternative oracle path."""
        if self._merge_counts[task_key] >= max_paths_per_task:
            return False
        merge_fn = getattr(self.adapter, "merge_success_trajectory_into_oracle", None)
        if merge_fn is None:
            return False
        merged = merge_fn(
            oracle,
            rollout_rows,
            signature_mode=signature_mode,
            traj_uid=traj_uid,
        )
        if merged:
            self._merge_counts[task_key] += 1
            return True
        return False


def _as_scalar(value: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value)
        if arr.shape:
            return float(arr.reshape(-1)[0])
        return float(arr.item())
    except Exception:
        try:
            return float(value)
        except Exception:
            return default


def _as_bool(value: Any) -> bool:
    return bool(_as_scalar(value, 0.0))


def _row_dict(data: DataProto, idx: int) -> Dict[str, Any]:
    return {key: values[idx] for key, values in data.non_tensor_batch.items()}


def _unique_steps(rows: Sequence[Tuple[int, Dict[str, Any]]]) -> List[Tuple[int, Dict[str, Any]]]:
    by_step: Dict[int, Tuple[int, Dict[str, Any]]] = {}
    for row_idx, row in rows:
        step_idx = int(_as_scalar(row.get("traj_step_idx"), 0.0))
        by_step.setdefault(step_idx, (row_idx, row))
    return [by_step[step] for step in sorted(by_step)]


def _valid_adv_mean(advantages: torch.Tensor, mask: torch.Tensor, row_idx: int) -> float:
    valid = mask[row_idx].bool()
    if not torch.any(valid):
        return 0.0
    return float(advantages[row_idx][valid].detach().float().mean().cpu().item())


def _episode_success(rows: Sequence[Dict[str, Any]]) -> bool:
    return any(_as_scalar(row.get("episode_rewards"), 0.0) > 0 for row in rows) or any(
        _as_bool(row.get("won")) for row in rows
    )


def _is_action_valid(row: Mapping[str, Any]) -> bool:
    if "is_action_valid" not in row:
        return True
    value = row.get("is_action_valid")
    if value is None:
        return True
    return _as_bool(value)


def _compute_step_scales(
    annotations: Sequence[Dict[str, Any]],
    cfg: GraphGuideConfig,
    eligible_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[float], List[float], List[int]]:
    unreasonable = [idx for idx, ann in enumerate(annotations) if int(ann.get("reasonable", 0)) == 0]
    eligible = list(eligible_indices) if eligible_indices is not None else list(unreasonable)
    blame = [0.0 for _ in annotations]
    source_name = str(cfg.blame_source).lower()
    source_key = {
        "avg_future_miss": "avg_future_miss",
        "future_miss": "avg_future_miss",
        "step_future_miss": "step_future_miss",
        "sum_future_miss": "step_future_miss",
        "future_miss_sum": "step_future_miss",
        "turn_future_miss": "turn_future_miss",
        "turn_future_miss_sum": "turn_future_miss",
        "graph_future_miss": "graph_future_miss",
        "milestone_future_miss": "milestone_future_miss",
        "migpo_future_miss": "milestone_future_miss",
        "step_regret": "step_regret",
        "normalized_step_regret": "normalized_step_regret",
        "damage": "turn_success_damage",
    }.get(source_name, source_name)
    for idx in eligible:
        if source_name in {"position_suffix", "suffix_position", "position", "early_position"}:
            blame[idx] = cfg.beta * float(len(annotations) - idx)
        else:
            blame[idx] = cfg.beta * float(annotations[idx].get(source_key, 0.0) or 0.0)

    if not eligible:
        return blame, [0.0 for _ in annotations], [0 for _ in annotations]

    scale_mode = str(cfg.scale_mode or "blame").lower()
    if scale_mode in {"constant", "const", "fixed"}:
        scales = [
            float(cfg.constant_scale) if idx in eligible else 0.0
            for idx in range(len(annotations))
        ]
        masks = [1 if idx in eligible else 0 for idx in range(len(annotations))]
        return [0.0 for _ in annotations], scales, masks
    if scale_mode not in {"blame", "score", "default"}:
        raise ValueError(f"unknown graph_guide.scale_mode={cfg.scale_mode!r}")

    normalization = str(cfg.blame_normalization or "mean")
    if normalization in {"softmax", "softmax_mean1"}:
        temp = max(float(cfg.softmax_temperature), cfg.eps)
        logits = np.array([blame[idx] / temp for idx in eligible], dtype=np.float64)
        logits = logits - float(np.max(logits))
        weights = np.exp(logits)
        denom = float(np.sum(weights))
        if denom <= cfg.eps:
            normed = np.ones(len(eligible), dtype=np.float64) / float(len(eligible))
        else:
            normed = weights / denom
        if normalization == "softmax_mean1":
            normed = normed * float(len(eligible))
        scales = [0.0 for _ in annotations]
        for local_idx, ann_idx in enumerate(eligible):
            scales[ann_idx] = float(normed[local_idx])
    elif cfg.normalize_blame and normalization == "mean":
        mean_blame = sum(blame[idx] for idx in eligible) / max(1, len(eligible))
        if mean_blame <= cfg.eps:
            scales = [1.0 if idx in eligible else 0.0 for idx in range(len(annotations))]
        else:
            scales = [blame[idx] / mean_blame if idx in eligible else 0.0 for idx in range(len(annotations))]
    else:
        scales = [blame[idx] if idx in eligible else 0.0 for idx in range(len(annotations))]
    masks = [1 if idx in eligible else 0 for idx in range(len(annotations))]
    return blame, scales, masks


def _blame_source_key(source_name: str) -> str:
    return {
        "avg_future_miss": "avg_future_miss",
        "future_miss": "avg_future_miss",
        "step_future_miss": "step_future_miss",
        "sum_future_miss": "step_future_miss",
        "future_miss_sum": "step_future_miss",
        "turn_future_miss": "turn_future_miss",
        "turn_future_miss_sum": "turn_future_miss",
        "graph_future_miss": "graph_future_miss",
        "milestone_future_miss": "milestone_future_miss",
        "migpo_future_miss": "milestone_future_miss",
        "step_regret": "step_regret",
        "normalized_step_regret": "normalized_step_regret",
        "damage": "turn_success_damage",
    }.get(source_name, source_name)


def _adv_cluster_key(row_adv: float, round_digits: int) -> str:
    digits = max(0, int(round_digits))
    return f"{round(float(row_adv), digits):.{digits}f}"


def _compute_advantage_cluster_scales(
    annotations: Sequence[Dict[str, Any]],
    row_adv: Sequence[float],
    *,
    round_digits: int = 6,
    blame_source: str = "step_future_miss",
    beta: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[List[float], List[float], List[int], List[str], List[int], List[float]]:
    """Compute per-negative-advantage-cluster mean-1 scales.

    The returned mask is the new eligible indicator: only rows with negative
    row-level advantage belong to a cluster and can receive negative reweight.
    """
    if len(row_adv) != len(annotations):
        raise ValueError(
            "row_adv and annotations must have the same length: "
            f"{len(row_adv)} != {len(annotations)}"
        )

    blame = [0.0 for _ in annotations]
    scales = [0.0 for _ in annotations]
    new_eligible = [0 for _ in annotations]
    cluster_keys = ["" for _ in annotations]
    cluster_sizes = [0 for _ in annotations]
    cluster_mean_blame = [0.0 for _ in annotations]
    source_key = _blame_source_key(str(blame_source).lower())

    clusters: Dict[str, List[int]] = defaultdict(list)
    for idx, adv in enumerate(row_adv):
        if float(adv) >= 0.0:
            continue
        key = _adv_cluster_key(float(adv), round_digits)
        clusters[key].append(idx)
        cluster_keys[idx] = key
        new_eligible[idx] = 1
        blame[idx] = float(beta) * float(annotations[idx].get(source_key, 0.0) or 0.0)

    for key, indices in clusters.items():
        mean_blame = sum(blame[idx] for idx in indices) / max(1, len(indices))
        for idx in indices:
            cluster_sizes[idx] = len(indices)
            cluster_mean_blame[idx] = float(mean_blame)
            scales[idx] = 1.0 if mean_blame <= eps else blame[idx] / mean_blame

    return blame, scales, new_eligible, cluster_keys, cluster_sizes, cluster_mean_blame


def _negative_mask_indices(annotations: Sequence[Dict[str, Any]], cfg: GraphGuideConfig) -> set[int]:
    if not cfg.mask_negative_reasonable:
        return set()
    mode = str(cfg.negative_mask_mode or "reasonable").lower()
    if mode in {"none", "false", "off", "no"}:
        return set()
    if mode in {"hit", "hit1", "hit_1", "progress"}:
        return {idx for idx, ann in enumerate(annotations) if int(ann.get("hit", 0) or 0) == 1}
    if mode in {"reasonable", "reasonableness"}:
        return {idx for idx, ann in enumerate(annotations) if int(ann.get("reasonable", 0)) == 1}
    raise ValueError(f"unknown graph_guide.negative_mask_mode={cfg.negative_mask_mode!r}")


def _negative_mask_applies(*, episode_success: bool, should_scale: bool, cfg: GraphGuideConfig) -> bool:
    mode = str(cfg.negative_mask_scope or "episode_failure").lower()
    if mode in {"episode_failure", "failure", "failed", "not_success"}:
        return not episode_success
    if mode in {"disadvantage", "negative_advantage", "negative", "dispreferred"}:
        return bool(should_scale)
    if mode in {"always", "all"}:
        return True
    raise ValueError(f"unknown graph_guide.negative_mask_scope={cfg.negative_mask_scope!r}")


def _use_advantage_cluster_mode(cfg: GraphGuideConfig) -> bool:
    mode = str(cfg.disadvantage_mode or "trajectory").lower()
    return mode in {"advantage_cluster", "adv_cluster", "new_eligible"}


def _adv_sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _should_mask_positive_unreasonable(
    *,
    row_adv: float,
    episode_success: bool,
    reasonable: int,
    cfg: GraphGuideConfig,
) -> bool:
    if not cfg.mask_positive_unreasonable or int(reasonable) != 0:
        return False
    mode = str(cfg.mask_positive_unreasonable_mode or "advantage").lower()
    if mode in {"advantage", "adv", "positive_advantage", "positive-advantage"}:
        return row_adv > 0.0
    if mode in {"episode_success", "success", "legacy"}:
        return episode_success
    if mode in {"always", "all"}:
        return True
    raise ValueError(
        "unknown graph_guide.mask_positive_unreasonable_mode="
        f"{cfg.mask_positive_unreasonable_mode!r}"
    )


def _scale_eligible_indices(
    annotations: Sequence[Dict[str, Any]],
    cfg: GraphGuideConfig,
    negative_masked: set[int],
) -> List[int]:
    mode = str(cfg.scale_eligible_mode or "unreasonable").lower()
    if mode in {"unreasonable", "dis", "bad"}:
        return [idx for idx, ann in enumerate(annotations) if int(ann.get("reasonable", 0)) == 0]
    if mode in {"unmasked", "eligible"}:
        return [idx for idx in range(len(annotations)) if idx not in negative_masked]
    if mode == "all":
        return list(range(len(annotations)))
    raise ValueError(f"unknown graph_guide.scale_eligible_mode={cfg.scale_eligible_mode!r}")


def _default_arrays(size: int) -> Dict[str, List[Any]]:
    return {
        "graph_guide_task_key": ["" for _ in range(size)],
        "graph_guide_reasonable": [1 for _ in range(size)],
        "graph_guide_reason": ["not_annotated" for _ in range(size)],
        "graph_guide_hit_source": ["oracle" for _ in range(size)],
        "graph_guide_spit": [0.0 for _ in range(size)],
        "graph_guide_spit_num": [0.0 for _ in range(size)],
        "graph_guide_spit_den": [0.0 for _ in range(size)],
        "graph_guide_delta_spit": [0.0 for _ in range(size)],
        "graph_guide_delta_spit_num": [0.0 for _ in range(size)],
        "graph_guide_progress_delta": [0.0 for _ in range(size)],
        "graph_guide_hit": [0 for _ in range(size)],
        "graph_guide_avg_future_miss": [0.0 for _ in range(size)],
        "graph_guide_step_future_miss": [0.0 for _ in range(size)],
        "graph_guide_turn_future_miss": [0.0 for _ in range(size)],
        "graph_guide_distance_available": [0 for _ in range(size)],
        "graph_guide_d_anchor": [0.0 for _ in range(size)],
        "graph_guide_d_next": [0.0 for _ in range(size)],
        "graph_guide_graph_step_miss": [0.0 for _ in range(size)],
        "graph_guide_graph_future_miss": [0.0 for _ in range(size)],
        "graph_guide_graph_step_damage": [0.0 for _ in range(size)],
        "graph_guide_graph_future_damage": [0.0 for _ in range(size)],
        "graph_guide_milestone_hit": [0 for _ in range(size)],
        "graph_guide_milestone_segment_id": [-1 for _ in range(size)],
        "graph_guide_milestone_next_hit_distance": [0.0 for _ in range(size)],
        "graph_guide_milestone_step_miss": [0.0 for _ in range(size)],
        "graph_guide_milestone_future_miss": [0.0 for _ in range(size)],
        "graph_guide_damage": [0.0 for _ in range(size)],
        "graph_guide_step_regret": [0.0 for _ in range(size)],
        "graph_guide_blame_score": [0.0 for _ in range(size)],
        "graph_guide_scale": [1.0 for _ in range(size)],
        "graph_guide_adv_mask": [0 for _ in range(size)],
        "graph_guide_new_eligible": [0 for _ in range(size)],
        "graph_guide_row_adv": [0.0 for _ in range(size)],
        "graph_guide_adv_sign": [0 for _ in range(size)],
        "graph_guide_adv_cluster_key": ["" for _ in range(size)],
        "graph_guide_adv_cluster_size": [0 for _ in range(size)],
        "graph_guide_positive_mask_applied": [0 for _ in range(size)],
        "graph_guide_negative_reweight_applied": [0 for _ in range(size)],
        "graph_guide_cluster_mean_blame": [0.0 for _ in range(size)],
        "graph_guide_traj_adv": [0.0 for _ in range(size)],
        "graph_guide_oracle_key": ["" for _ in range(size)],
        "graph_guide_oracle_missing": [0 for _ in range(size)],
        "graph_guide_reference_path": ["" for _ in range(size)],
        "graph_guide_from_signature": [[] for _ in range(size)],
        "graph_guide_to_signature": [[] for _ in range(size)],
        "graph_guide_from_signature_hash": ["" for _ in range(size)],
        "graph_guide_to_signature_hash": ["" for _ in range(size)],
        "graph_guide_matched_subgoals": [[] for _ in range(size)],
        "graph_guide_matched_substeps": [[] for _ in range(size)],
        "graph_guide_covered_options": [[] for _ in range(size)],
        "graph_guide_purchased_asin": ["" for _ in range(size)],
        "graph_guide_purchased_options": [{} for _ in range(size)],
    }


def _write_graph_guide_arrays(data: DataProto, out: Mapping[str, Sequence[Any]]) -> None:
    for key, values in out.items():
        if key in {
            "graph_guide_spit",
            "graph_guide_spit_num",
            "graph_guide_spit_den",
            "graph_guide_delta_spit",
            "graph_guide_delta_spit_num",
            "graph_guide_progress_delta",
            "graph_guide_avg_future_miss",
            "graph_guide_step_future_miss",
            "graph_guide_turn_future_miss",
            "graph_guide_d_anchor",
            "graph_guide_d_next",
            "graph_guide_graph_step_miss",
            "graph_guide_graph_future_miss",
            "graph_guide_graph_step_damage",
            "graph_guide_graph_future_damage",
            "graph_guide_milestone_next_hit_distance",
            "graph_guide_milestone_step_miss",
            "graph_guide_milestone_future_miss",
            "graph_guide_damage",
            "graph_guide_step_regret",
            "graph_guide_blame_score",
            "graph_guide_scale",
            "graph_guide_row_adv",
            "graph_guide_cluster_mean_blame",
            "graph_guide_traj_adv",
        }:
            data.non_tensor_batch[key] = np.array(values, dtype=np.float32)
        elif key in {
            "graph_guide_reasonable",
            "graph_guide_hit",
            "graph_guide_distance_available",
            "graph_guide_milestone_hit",
            "graph_guide_milestone_segment_id",
            "graph_guide_adv_mask",
            "graph_guide_new_eligible",
            "graph_guide_adv_sign",
            "graph_guide_adv_cluster_size",
            "graph_guide_positive_mask_applied",
            "graph_guide_negative_reweight_applied",
            "graph_guide_oracle_missing",
        }:
            data.non_tensor_batch[key] = np.array(values, dtype=np.int32)
        else:
            data.non_tensor_batch[key] = np.array(values, dtype=object)


def _write_annotation_to_out(
    out: Dict[str, List[Any]],
    row_idx: int,
    ann: Mapping[str, Any],
    *,
    task_key: str = "",
    hit_source: str = "oracle",
    selected_ref_id: str = "",
    traj_adv: Optional[float] = None,
    blame_score: Optional[float] = None,
    scale: Optional[float] = None,
    adv_mask: Optional[int] = None,
) -> None:
    out["graph_guide_task_key"][row_idx] = task_key or ""
    out["graph_guide_oracle_key"][row_idx] = task_key or ""
    out["graph_guide_hit_source"][row_idx] = hit_source
    if traj_adv is not None:
        out["graph_guide_traj_adv"][row_idx] = float(traj_adv)
    out["graph_guide_reasonable"][row_idx] = int(ann.get("reasonable", 0) or 0)
    out["graph_guide_reason"][row_idx] = str(ann.get("reason", "unknown"))
    out["graph_guide_spit"][row_idx] = float(ann.get("spit", 0.0) or 0.0)
    out["graph_guide_spit_num"][row_idx] = float(ann.get("spit_num", 0.0) or 0.0)
    out["graph_guide_spit_den"][row_idx] = float(ann.get("spit_den", 0.0) or 0.0)
    out["graph_guide_delta_spit"][row_idx] = float(ann.get("delta_spit", 0.0) or 0.0)
    out["graph_guide_delta_spit_num"][row_idx] = float(ann.get("delta_spit_num", 0.0) or 0.0)
    out["graph_guide_progress_delta"][row_idx] = float(ann.get("progress_delta", 0.0) or 0.0)
    out["graph_guide_hit"][row_idx] = int(ann.get("hit", 0) or 0)
    out["graph_guide_avg_future_miss"][row_idx] = float(ann.get("avg_future_miss", 0.0) or 0.0)
    out["graph_guide_step_future_miss"][row_idx] = float(ann.get("step_future_miss", 0.0) or 0.0)
    out["graph_guide_turn_future_miss"][row_idx] = float(ann.get("turn_future_miss", 0.0) or 0.0)
    out["graph_guide_distance_available"][row_idx] = int(ann.get("graph_distance_available", 0) or 0)
    out["graph_guide_d_anchor"][row_idx] = float(ann.get("graph_d_anchor", 0.0) or 0.0)
    out["graph_guide_d_next"][row_idx] = float(ann.get("graph_d_next", 0.0) or 0.0)
    out["graph_guide_graph_step_miss"][row_idx] = float(ann.get("graph_step_miss", 0.0) or 0.0)
    out["graph_guide_graph_future_miss"][row_idx] = float(ann.get("graph_future_miss", 0.0) or 0.0)
    out["graph_guide_graph_step_damage"][row_idx] = float(ann.get("graph_step_damage", 0.0) or 0.0)
    out["graph_guide_graph_future_damage"][row_idx] = float(ann.get("graph_future_damage", 0.0) or 0.0)
    out["graph_guide_milestone_hit"][row_idx] = int(ann.get("milestone_hit", 0) or 0)
    out["graph_guide_milestone_segment_id"][row_idx] = int(ann.get("milestone_segment_id", -1) or 0)
    out["graph_guide_milestone_next_hit_distance"][row_idx] = float(
        ann.get("milestone_next_hit_distance", 0.0) or 0.0
    )
    out["graph_guide_milestone_step_miss"][row_idx] = float(ann.get("milestone_step_miss", 0.0) or 0.0)
    out["graph_guide_milestone_future_miss"][row_idx] = float(
        ann.get("milestone_future_miss", 0.0) or 0.0
    )
    out["graph_guide_damage"][row_idx] = float(ann.get("turn_success_damage", 0.0) or 0.0)
    out["graph_guide_step_regret"][row_idx] = float(ann.get("step_regret", 0.0) or 0.0)
    if blame_score is not None:
        out["graph_guide_blame_score"][row_idx] = float(blame_score)
    if scale is not None:
        out["graph_guide_scale"][row_idx] = float(scale)
    if adv_mask is not None:
        out["graph_guide_adv_mask"][row_idx] = int(adv_mask)
    out["graph_guide_reference_path"][row_idx] = selected_ref_id
    out["graph_guide_from_signature"][row_idx] = list(ann.get("from_signature") or [])
    out["graph_guide_to_signature"][row_idx] = list(ann.get("to_signature") or [])
    out["graph_guide_from_signature_hash"][row_idx] = str(ann.get("from_signature_hash") or "")
    out["graph_guide_to_signature_hash"][row_idx] = str(ann.get("to_signature_hash") or "")
    out["graph_guide_matched_subgoals"][row_idx] = list(ann.get("matched_subgoals") or [])
    out["graph_guide_matched_substeps"][row_idx] = list(ann.get("matched_substeps") or [])
    out["graph_guide_covered_options"][row_idx] = list(ann.get("covered_options_after") or [])
    out["graph_guide_purchased_asin"][row_idx] = str(ann.get("purchased_asin") or "")
    out["graph_guide_purchased_options"][row_idx] = dict(ann.get("purchased_options") or {})


def _row_has_key(rows: Sequence[Mapping[str, Any]], key: str) -> bool:
    return any(key in row for row in rows)


def _as_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return _as_bool(value)


def _action_text_from_row(row: Mapping[str, Any]) -> str:
    for key in ("actions", "executed_action", "raw_action_text", "text_actions"):
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0] if value.size else ""
        text = str(value)
        if text:
            return text
    return ""


def _migpo_dynamic_hits(unique_rows: Sequence[Dict[str, Any]]) -> Optional[List[int]]:
    if not (_row_has_key(unique_rows, "milestone_achieved") or _row_has_key(unique_rows, "subgoal_completed")):
        return None
    hits: List[int] = []
    for row in unique_rows:
        hit = _as_flag(row.get("milestone_achieved")) or _as_flag(row.get("subgoal_completed"))
        hits.append(1 if hit else 0)
    return hits


def _migpo_static_hits_and_segments(
    unique_rows: Sequence[Dict[str, Any]],
    cfg: GraphGuideConfig,
) -> Tuple[List[int], List[int]]:
    hit_flags = [0 for _ in unique_rows]
    segment_ids = [-1 for _ in unique_rows]
    if not unique_rows:
        return hit_flags, segment_ids

    trial_id = str(unique_rows[0].get("trial_id") or "")
    actions = [_action_text_from_row(row) for row in unique_rows]
    if not trial_id or not any(actions):
        return hit_flags, segment_ids

    try:
        from recipe.migpo.core_migpo import compute_segment_rewards, match_milestones
        from recipe.migpo.milestone_loader import get_milestone_loader

        milestones = get_milestone_loader().get_milestones(trial_id)
        if milestones is None:
            return hit_flags, segment_ids
        match_indices = match_milestones(actions, milestones, float(cfg.migpo_threshold))
        _, computed_segment_ids = compute_segment_rewards(len(actions), match_indices, gamma=0.99)
    except Exception:
        return hit_flags, segment_ids

    matched = {int(idx) for idx in match_indices if int(idx) >= 0}
    hit_flags = [1 if idx in matched else 0 for idx in range(len(unique_rows))]
    segment_ids = [int(segment_id) for segment_id in computed_segment_ids.tolist()]
    return hit_flags, segment_ids


def _annotate_migpo_native_trajectory(
    unique_rows: Sequence[Dict[str, Any]],
    cfg: GraphGuideConfig,
) -> List[Dict[str, Any]]:
    dynamic_hits = _migpo_dynamic_hits(unique_rows)
    if dynamic_hits is None:
        hits, segment_ids = _migpo_static_hits_and_segments(unique_rows, cfg)
        source_reason = "migpo_static_milestone"
    else:
        hits = dynamic_hits
        segment_ids = []
        current_segment = 0
        for hit in hits:
            segment_ids.append(current_segment)
            if hit:
                current_segment += 1
        source_reason = "migpo_dynamic_milestone"

    hit_indices = [idx for idx, hit in enumerate(hits) if hit]
    no_hit_fallback = len(hit_indices) == 0
    next_hit_distance = [0.0 for _ in unique_rows]
    next_hit_exists = [False for _ in unique_rows]
    next_hit = None
    for idx in reversed(range(len(unique_rows))):
        if hits[idx]:
            next_hit = idx
        if next_hit is not None:
            next_hit_exists[idx] = True
            next_hit_distance[idx] = float(next_hit - idx)

    mode = str(cfg.migpo_reasonable_source or "next_hit_segment").lower()
    if mode not in {"next_hit_segment", "future_hit", "hit_only"}:
        raise ValueError(f"unknown graph_guide.migpo_reasonable_source={cfg.migpo_reasonable_source!r}")

    annotations: List[Dict[str, Any]] = []
    step_misses: List[float] = []
    for local_idx, row in enumerate(unique_rows):
        step_idx = int(_as_scalar(row.get("traj_step_idx"), float(local_idx)))
        is_valid = _is_action_valid(row)
        hit = int(hits[local_idx])
        if not is_valid:
            reasonable = 0
            reason = "migpo_invalid_action"
        elif mode == "hit_only":
            reasonable = hit
            reason = source_reason if hit else "migpo_non_hit"
        elif no_hit_fallback:
            reasonable = 1
            reason = "migpo_no_hit_fallback"
        elif next_hit_exists[local_idx]:
            reasonable = 1
            reason = source_reason if hit else "migpo_prefatory_non_hit"
        else:
            reasonable = 0
            reason = "migpo_failed_tail_after_last_hit"

        step_miss = 1.0 if hit == 0 else 0.0
        annotations.append(
            {
                "traj_step_idx": step_idx,
                "reasonable": reasonable,
                "reason": reason,
                "hit": hit,
                "milestone_hit": hit,
                "milestone_segment_id": int(segment_ids[local_idx]) if local_idx < len(segment_ids) else -1,
                "milestone_next_hit_distance": next_hit_distance[local_idx],
                "milestone_step_miss": step_miss,
                "milestone_future_miss": 0.0,
                "step_future_miss": 0.0,
                "avg_future_miss": 0.0,
                "turn_future_miss": 0.0,
            }
        )
        step_misses.append(step_miss)

    future_miss = 0.0
    total_miss = float(sum(step_misses))
    for idx in reversed(range(len(annotations))):
        future_miss += step_misses[idx]
        tail_len = len(annotations) - idx
        annotations[idx]["milestone_future_miss"] = future_miss
        annotations[idx]["step_future_miss"] = future_miss
        annotations[idx]["avg_future_miss"] = future_miss / float(tail_len) if tail_len else 0.0
        annotations[idx]["turn_future_miss"] = total_miss

    return annotations


def _apply_migpo_native_credit(data: DataProto, cfg: GraphGuideConfig) -> DataProto:
    if "advantages" not in data.batch:
        raise KeyError("graph-guided credit requires batch['advantages'] to be computed first")

    advantages = data.batch["advantages"]
    response_mask = data.batch.get("response_mask")
    if response_mask is None:
        response_mask = torch.ones_like(advantages, dtype=torch.bool)

    rows_by_traj: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx in range(len(data)):
        row = _row_dict(data, idx)
        rows_by_traj[str(row.get("traj_uid"))].append((idx, row))

    out = _default_arrays(len(data))
    for traj_uid, indexed_rows in rows_by_traj.items():
        unique = _unique_steps(indexed_rows)
        if not unique:
            continue
        unique_indices = [idx for idx, _ in unique]
        unique_rows = [row for _, row in unique]
        task_key = str(unique_rows[0].get("task_desc") or unique_rows[0].get("uid") or unique_rows[0].get("trial_id") or "")
        episode_success = _episode_success(unique_rows)
        annotations = _annotate_migpo_native_trajectory(unique_rows, cfg)
        row_adv = [_valid_adv_mean(advantages, response_mask, idx) for idx in unique_indices]
        traj_adv = float(np.mean(row_adv)) if row_adv else 0.0
        is_dis_traj = traj_adv < 0.0
        use_adv_cluster = _use_advantage_cluster_mode(cfg)
        negative_masked = _negative_mask_indices(annotations, cfg)
        if use_adv_cluster:
            (
                blame,
                scales,
                adv_masks,
                cluster_keys,
                cluster_sizes,
                cluster_mean_blame,
            ) = _compute_advantage_cluster_scales(
                annotations,
                row_adv,
                round_digits=int(cfg.adv_cluster_round_digits),
                blame_source=str(cfg.blame_source),
                beta=float(cfg.beta),
                eps=float(cfg.eps),
            )
        else:
            scale_eligible = _scale_eligible_indices(annotations, cfg, negative_masked)
            blame, scales, adv_masks = _compute_step_scales(annotations, cfg, eligible_indices=scale_eligible)
            cluster_keys = ["" for _ in annotations]
            cluster_sizes = [0 for _ in annotations]
            cluster_mean_blame = [0.0 for _ in annotations]

        ann_by_step = {
            int(ann["traj_step_idx"]): (ann_idx, ann)
            for ann_idx, ann in enumerate(annotations)
        }
        for row_idx, row in indexed_rows:
            step_idx = int(_as_scalar(row.get("traj_step_idx"), 0.0))
            ann_info = ann_by_step.get(step_idx)
            out["graph_guide_task_key"][row_idx] = task_key
            out["graph_guide_oracle_key"][row_idx] = ""
            out["graph_guide_hit_source"][row_idx] = "migpo_milestone"
            out["graph_guide_traj_adv"][row_idx] = traj_adv
            if ann_info is None:
                out["graph_guide_reason"][row_idx] = "migpo_native_annotation_missing"
                continue

            ann_idx, ann = ann_info
            reasonable = int(ann.get("reasonable", 0))
            scale = float(scales[ann_idx])
            _write_annotation_to_out(
                out,
                row_idx,
                ann,
                task_key=task_key,
                hit_source="migpo_milestone",
                selected_ref_id="",
                traj_adv=traj_adv,
                blame_score=float(blame[ann_idx]),
                scale=scale,
                adv_mask=int(adv_masks[ann_idx]),
            )
            out["graph_guide_oracle_key"][row_idx] = ""
            out["graph_guide_new_eligible"][row_idx] = int(adv_masks[ann_idx]) if use_adv_cluster else 0
            out["graph_guide_adv_cluster_key"][row_idx] = cluster_keys[ann_idx]
            out["graph_guide_adv_cluster_size"][row_idx] = int(cluster_sizes[ann_idx])
            out["graph_guide_cluster_mean_blame"][row_idx] = float(cluster_mean_blame[ann_idx])

            row_adv_mean = _valid_adv_mean(advantages, response_mask, row_idx)
            row_is_dis = row_adv_mean < 0.0
            out["graph_guide_row_adv"][row_idx] = row_adv_mean
            out["graph_guide_adv_sign"][row_idx] = _adv_sign(row_adv_mean)
            should_scale = row_is_dis if use_adv_cluster else (
                is_dis_traj if cfg.disadvantage_mode == "trajectory" else row_is_dis
            )
            if _should_mask_positive_unreasonable(
                row_adv=row_adv_mean,
                episode_success=episode_success,
                reasonable=reasonable,
                cfg=cfg,
            ):
                out["graph_guide_positive_mask_applied"][row_idx] = 1
                advantages[row_idx] = torch.zeros_like(advantages[row_idx])
                continue
            if (
                ann_idx in negative_masked
                and _negative_mask_applies(
                    episode_success=episode_success,
                    should_scale=should_scale,
                    cfg=cfg,
                )
            ):
                advantages[row_idx] = torch.zeros_like(advantages[row_idx])
                continue
            if (
                cfg.reweight_disadvantage_unreasonable
                and should_scale
                and adv_masks[ann_idx]
            ):
                out["graph_guide_negative_reweight_applied"][row_idx] = 1
                advantages[row_idx] = advantages[row_idx] * scale

    _write_graph_guide_arrays(data, out)
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages
    return data


def annotate_graph_guide_metadata(
    data: DataProto, config: DictConfig, *, update_oracle: bool = False
) -> DataProto:
    """Annotate steps; only training PR opts into ARG evolution here.

    Evaluation keeps the default read-only annotation behavior.
    """
    cfg = graph_guide_config(config)
    if not cfg.enabled:
        return data
    cache = OracleGraphCache(
        cfg.oracle_index_path,
        strict_missing=cfg.strict_missing_oracle,
        env_adapter=cfg.env_adapter,
    )
    rows_by_traj: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx in range(len(data)):
        row = _row_dict(data, idx)
        rows_by_traj[str(row.get("traj_uid"))].append((idx, row))

    out = _default_arrays(len(data))
    for traj_uid, indexed_rows in rows_by_traj.items():
        unique = _unique_steps(indexed_rows)
        if not unique:
            continue
        unique_rows = [row for _, row in unique]
        task_key, oracle = cache.get(unique_rows[0])
        if oracle is None:
            for row_idx, _ in indexed_rows:
                out["graph_guide_task_key"][row_idx] = task_key or ""
                out["graph_guide_oracle_key"][row_idx] = task_key or ""
                out["graph_guide_oracle_missing"][row_idx] = 1
            continue
        if update_oracle and cfg.online_oracle_update and _episode_success(unique_rows) and task_key:
            cache.maybe_merge_success_path(
                task_key,
                oracle,
                unique_rows,
                signature_mode=cfg.signature_mode,
                traj_uid=traj_uid,
                max_paths_per_task=cfg.online_max_paths_per_task,
            )
        annotated = cache.adapter.annotate_trajectory(unique_rows, oracle, signature_mode=cfg.signature_mode)
        annotations = annotated["annotations"]
        hit_source = "oracle"
        if _use_random_hit(cfg):
            annotations = _randomize_hit_annotations(annotations, cfg=cfg, traj_uid=traj_uid)
            hit_source = "random"
        selected_ref = annotated.get("selected_reference_path") or {}
        selected_ref_id = str(selected_ref.get("path_id") or "")
        ann_by_step = {
            int(ann["traj_step_idx"]): ann
            for ann in annotations
        }
        for row_idx, row in indexed_rows:
            step_idx = int(_as_scalar(row.get("traj_step_idx"), 0.0))
            ann = ann_by_step.get(step_idx)
            out["graph_guide_task_key"][row_idx] = task_key or ""
            out["graph_guide_oracle_key"][row_idx] = task_key or ""
            out["graph_guide_hit_source"][row_idx] = hit_source
            if ann is None:
                out["graph_guide_reason"][row_idx] = "annotation_missing"
                continue
            out["graph_guide_reasonable"][row_idx] = int(ann.get("reasonable", 0) or 0)
            out["graph_guide_reason"][row_idx] = str(ann.get("reason", "unknown"))
            out["graph_guide_spit"][row_idx] = float(ann.get("spit", 0.0) or 0.0)
            out["graph_guide_spit_num"][row_idx] = float(ann.get("spit_num", 0.0) or 0.0)
            out["graph_guide_spit_den"][row_idx] = float(ann.get("spit_den", 0.0) or 0.0)
            out["graph_guide_delta_spit"][row_idx] = float(ann.get("delta_spit", 0.0) or 0.0)
            out["graph_guide_delta_spit_num"][row_idx] = float(ann.get("delta_spit_num", 0.0) or 0.0)
            out["graph_guide_progress_delta"][row_idx] = float(ann.get("progress_delta", 0.0) or 0.0)
            out["graph_guide_hit"][row_idx] = int(ann.get("hit", 0) or 0)
            out["graph_guide_avg_future_miss"][row_idx] = float(ann.get("avg_future_miss", 0.0) or 0.0)
            out["graph_guide_step_future_miss"][row_idx] = float(ann.get("step_future_miss", 0.0) or 0.0)
            out["graph_guide_turn_future_miss"][row_idx] = float(ann.get("turn_future_miss", 0.0) or 0.0)
            out["graph_guide_damage"][row_idx] = float(ann.get("turn_success_damage", 0.0) or 0.0)
            out["graph_guide_step_regret"][row_idx] = float(ann.get("step_regret", 0.0) or 0.0)
            out["graph_guide_reference_path"][row_idx] = selected_ref_id
            out["graph_guide_from_signature"][row_idx] = list(ann.get("from_signature") or [])
            out["graph_guide_to_signature"][row_idx] = list(ann.get("to_signature") or [])
            out["graph_guide_from_signature_hash"][row_idx] = str(ann.get("from_signature_hash") or "")
            out["graph_guide_to_signature_hash"][row_idx] = str(ann.get("to_signature_hash") or "")
            out["graph_guide_matched_subgoals"][row_idx] = list(ann.get("matched_subgoals") or [])
            out["graph_guide_matched_substeps"][row_idx] = list(ann.get("matched_substeps") or [])
            out["graph_guide_covered_options"][row_idx] = list(ann.get("covered_options_after") or [])
            out["graph_guide_purchased_asin"][row_idx] = str(ann.get("purchased_asin") or "")
            out["graph_guide_purchased_options"][row_idx] = dict(ann.get("purchased_options") or {})
    _write_graph_guide_arrays(data, out)
    return data


def apply_graph_guide_credit(data: DataProto, config: DictConfig) -> DataProto:
    cfg = graph_guide_config(config)
    if not cfg.enabled:
        return data
    if _use_migpo_native(cfg):
        return _apply_migpo_native_credit(data, cfg)
    if "advantages" not in data.batch:
        raise KeyError("graph-guided credit requires batch['advantages'] to be computed first")

    cache = OracleGraphCache(
        cfg.oracle_index_path,
        strict_missing=cfg.strict_missing_oracle,
        env_adapter=cfg.env_adapter,
    )
    advantages = data.batch["advantages"]
    response_mask = data.batch.get("response_mask")
    if response_mask is None:
        response_mask = torch.ones_like(advantages, dtype=torch.bool)

    rows_by_traj: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx in range(len(data)):
        row = _row_dict(data, idx)
        rows_by_traj[str(row.get("traj_uid"))].append((idx, row))

    out = _default_arrays(len(data))
    for traj_uid, indexed_rows in rows_by_traj.items():
        unique = _unique_steps(indexed_rows)
        if not unique:
            continue
        unique_indices = [idx for idx, _ in unique]
        unique_rows = [row for _, row in unique]
        task_key, oracle = cache.get(unique_rows[0])
        episode_success = _episode_success(unique_rows)
        if oracle is None:
            for row_idx, _ in indexed_rows:
                out["graph_guide_task_key"][row_idx] = task_key or ""
                out["graph_guide_oracle_key"][row_idx] = task_key or ""
                out["graph_guide_oracle_missing"][row_idx] = 1
            continue

        if cfg.online_oracle_update and episode_success and task_key:
            cache.maybe_merge_success_path(
                task_key,
                oracle,
                unique_rows,
                signature_mode=cfg.signature_mode,
                traj_uid=traj_uid,
                max_paths_per_task=cfg.online_max_paths_per_task,
            )
        annotated = cache.adapter.annotate_trajectory(unique_rows, oracle, signature_mode=cfg.signature_mode)
        annotations = annotated["annotations"]
        hit_source = "oracle"
        if _use_random_hit(cfg):
            annotations = _randomize_hit_annotations(annotations, cfg=cfg, traj_uid=traj_uid)
            hit_source = "random"
        selected_ref = annotated.get("selected_reference_path") or {}
        selected_ref_id = str(selected_ref.get("path_id") or "")
        row_adv = [_valid_adv_mean(advantages, response_mask, idx) for idx in unique_indices]
        traj_adv = float(np.mean(row_adv)) if row_adv else 0.0
        is_dis_traj = traj_adv < 0.0
        use_adv_cluster = _use_advantage_cluster_mode(cfg)
        negative_masked = _negative_mask_indices(annotations, cfg)
        if use_adv_cluster:
            (
                blame,
                scales,
                adv_masks,
                cluster_keys,
                cluster_sizes,
                cluster_mean_blame,
            ) = _compute_advantage_cluster_scales(
                annotations,
                row_adv,
                round_digits=int(cfg.adv_cluster_round_digits),
                blame_source=str(cfg.blame_source),
                beta=float(cfg.beta),
                eps=float(cfg.eps),
            )
        else:
            scale_eligible = _scale_eligible_indices(annotations, cfg, negative_masked)
            blame, scales, adv_masks = _compute_step_scales(annotations, cfg, eligible_indices=scale_eligible)
            cluster_keys = ["" for _ in annotations]
            cluster_sizes = [0 for _ in annotations]
            cluster_mean_blame = [0.0 for _ in annotations]

        ann_by_step = {
            int(ann["traj_step_idx"]): (ann_idx, ann)
            for ann_idx, ann in enumerate(annotations)
        }
        for row_idx, row in indexed_rows:
            step_idx = int(_as_scalar(row.get("traj_step_idx"), 0.0))
            ann_info = ann_by_step.get(step_idx)
            out["graph_guide_task_key"][row_idx] = task_key or ""
            out["graph_guide_oracle_key"][row_idx] = task_key or ""
            out["graph_guide_hit_source"][row_idx] = hit_source
            out["graph_guide_traj_adv"][row_idx] = traj_adv
            if ann_info is None:
                out["graph_guide_reason"][row_idx] = "annotation_missing"
                continue
            ann_idx, ann = ann_info
            reasonable = int(ann.get("reasonable", 0))
            scale = float(scales[ann_idx])
            out["graph_guide_reasonable"][row_idx] = reasonable
            out["graph_guide_reason"][row_idx] = str(ann.get("reason", "unknown"))
            out["graph_guide_spit"][row_idx] = float(ann.get("spit", 0.0) or 0.0)
            out["graph_guide_spit_num"][row_idx] = float(ann.get("spit_num", 0.0) or 0.0)
            out["graph_guide_spit_den"][row_idx] = float(ann.get("spit_den", 0.0) or 0.0)
            out["graph_guide_delta_spit"][row_idx] = float(ann.get("delta_spit", 0.0) or 0.0)
            out["graph_guide_delta_spit_num"][row_idx] = float(ann.get("delta_spit_num", 0.0) or 0.0)
            out["graph_guide_progress_delta"][row_idx] = float(ann.get("progress_delta", 0.0) or 0.0)
            out["graph_guide_hit"][row_idx] = int(ann.get("hit", 0) or 0)
            out["graph_guide_avg_future_miss"][row_idx] = float(ann.get("avg_future_miss", 0.0) or 0.0)
            out["graph_guide_step_future_miss"][row_idx] = float(ann.get("step_future_miss", 0.0) or 0.0)
            out["graph_guide_turn_future_miss"][row_idx] = float(ann.get("turn_future_miss", 0.0) or 0.0)
            out["graph_guide_damage"][row_idx] = float(ann.get("turn_success_damage", 0.0) or 0.0)
            out["graph_guide_step_regret"][row_idx] = float(ann.get("step_regret", 0.0) or 0.0)
            out["graph_guide_blame_score"][row_idx] = float(blame[ann_idx])
            out["graph_guide_scale"][row_idx] = scale
            out["graph_guide_adv_mask"][row_idx] = int(adv_masks[ann_idx])
            out["graph_guide_new_eligible"][row_idx] = int(adv_masks[ann_idx]) if use_adv_cluster else 0
            out["graph_guide_adv_cluster_key"][row_idx] = cluster_keys[ann_idx]
            out["graph_guide_adv_cluster_size"][row_idx] = int(cluster_sizes[ann_idx])
            out["graph_guide_cluster_mean_blame"][row_idx] = float(cluster_mean_blame[ann_idx])
            out["graph_guide_reference_path"][row_idx] = selected_ref_id
            out["graph_guide_from_signature"][row_idx] = list(ann.get("from_signature") or [])
            out["graph_guide_to_signature"][row_idx] = list(ann.get("to_signature") or [])
            out["graph_guide_from_signature_hash"][row_idx] = str(ann.get("from_signature_hash") or "")
            out["graph_guide_to_signature_hash"][row_idx] = str(ann.get("to_signature_hash") or "")
            out["graph_guide_matched_subgoals"][row_idx] = list(ann.get("matched_subgoals") or [])
            out["graph_guide_matched_substeps"][row_idx] = list(ann.get("matched_substeps") or [])
            out["graph_guide_covered_options"][row_idx] = list(ann.get("covered_options_after") or [])
            out["graph_guide_purchased_asin"][row_idx] = str(ann.get("purchased_asin") or "")
            out["graph_guide_purchased_options"][row_idx] = dict(ann.get("purchased_options") or {})

            row_adv_mean = _valid_adv_mean(advantages, response_mask, row_idx)
            row_is_dis = row_adv_mean < 0.0
            out["graph_guide_row_adv"][row_idx] = row_adv_mean
            out["graph_guide_adv_sign"][row_idx] = _adv_sign(row_adv_mean)
            should_scale = row_is_dis if use_adv_cluster else (
                is_dis_traj if cfg.disadvantage_mode == "trajectory" else row_is_dis
            )
            if _should_mask_positive_unreasonable(
                row_adv=row_adv_mean,
                episode_success=episode_success,
                reasonable=reasonable,
                cfg=cfg,
            ):
                out["graph_guide_positive_mask_applied"][row_idx] = 1
                advantages[row_idx] = torch.zeros_like(advantages[row_idx])
                continue
            if (
                ann_idx in negative_masked
                and _negative_mask_applies(
                    episode_success=episode_success,
                    should_scale=should_scale,
                    cfg=cfg,
                )
            ):
                advantages[row_idx] = torch.zeros_like(advantages[row_idx])
                continue
            if (
                cfg.reweight_disadvantage_unreasonable
                and should_scale
                and adv_masks[ann_idx]
            ):
                out["graph_guide_negative_reweight_applied"][row_idx] = 1
                advantages[row_idx] = advantages[row_idx] * scale

    _write_graph_guide_arrays(data, out)

    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages
    return data
