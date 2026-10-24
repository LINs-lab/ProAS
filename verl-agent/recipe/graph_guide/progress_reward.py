"""ARG progress reward added before standard GRPO advantage estimation.

Each response row represents an environment step. Its existing outcome reward
and invalid-action penalty are preserved, with beta * SPIT_t added to its last
valid response token. This is a per-step bonus, not a sum of SPIT over time.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from recipe.graph_guide.graph_guide_credit import (
    annotate_graph_guide_metadata,
    graph_guide_config,
)


def apply_graph_progress_reward(data, config):
    cfg = graph_guide_config(config)
    if not cfg.enabled or not cfg.progress_reward_enabled:
        return data
    beta = float(cfg.progress_reward_beta)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("progress_reward_beta must be finite and nonnegative")
    if str(cfg.annotation_source).lower() != "oracle" or str(cfg.hit_source).lower() != "oracle":
        raise ValueError("ARG progress reward requires oracle annotations")
    if data.meta_info.get("graph_guide_progress_reward_applied", False):
        raise ValueError("ARG progress reward has already been applied to this batch")

    scores = data.batch["token_level_scores"]
    rewards = data.batch["token_level_rewards"]
    mask = data.batch.get("response_mask")
    if mask is None:
        mask = data.batch["attention_mask"][:, -rewards.shape[-1]:]
    mask = mask.bool()
    if mask.shape != rewards.shape or scores.shape != rewards.shape:
        raise ValueError("Progress reward requires matching score, reward, and response-mask shapes")
    if not mask.any(dim=-1).all():
        raise ValueError("Cannot assign progress reward to an empty response")

    # Use the same successful-path merge and SPIT annotation as ARG shaping.
    # Raw episode_rewards remain unchanged, so progress cannot turn a failure
    # into a success for oracle evolution or evaluation metrics.
    annotate_graph_guide_metadata(data, config, update_oracle=True)
    reasons = data.non_tensor_batch["graph_guide_reason"]
    missing = data.non_tensor_batch["graph_guide_oracle_missing"]
    if any(reason == "annotation_missing" for reason in reasons) or np.any(missing):
        raise ValueError("ARG progress reward requires an oracle annotation for every step")
    progress = np.asarray(data.non_tensor_batch["graph_guide_spit"], dtype=np.float64)
    if not np.isfinite(progress).all() or np.any(progress < 0) or np.any(progress > 1):
        raise ValueError("SPIT must be finite and lie in [0, 1]")
    bonus = torch.as_tensor(beta * progress, device=rewards.device, dtype=rewards.dtype)
    positions = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)
    last_token = positions.masked_fill(~mask, -1).max(dim=-1).values
    rows = torch.arange(len(data), device=rewards.device)

    # Clone separately: scores and rewards can alias when reward KL is disabled.
    shaped_scores = scores.clone()
    shaped_rewards = rewards.clone()
    shaped_scores[rows, last_token] += bonus.to(shaped_scores.dtype)
    shaped_rewards[rows, last_token] += bonus
    data.non_tensor_batch["graph_guide_reward_before_progress"] = (
        rewards.sum(dim=-1).detach().cpu().numpy()
    )
    data.non_tensor_batch["graph_guide_progress_reward"] = beta * progress
    data.batch["token_level_scores"] = shaped_scores
    data.batch["token_level_rewards"] = shaped_rewards
    data.meta_info["graph_guide_progress_reward_applied"] = True
    return data
