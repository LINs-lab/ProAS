# Copyright 2025 MiGPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Core functions to implement the MiGPO (Milestone-Guided Policy Optimization) algorithm.

MiGPO extends GiGPO by introducing milestone-based reward shaping:
1. Sequential matching of trajectory actions with milestone sequences
2. Reverse-decay reward computation within each segment
3. Advantage computation grouped by milestone index
"""

import numpy as np
import torch
from collections import defaultdict
from typing import List, Tuple, Optional
import uuid

from difflib import SequenceMatcher

from verl import DataProto


# ---------------------------------------------------------- #
# --------------- Step Rewards Computation ----------------- #
# ---------------------------------------------------------- #

def compute_migpo_step_rewards(batch: DataProto,
                                gamma: float = 0.99,
                                threshold: float = 0.85) -> torch.Tensor:
    """
    Compute MiGPO step rewards (milestone-based) with scale [0, 10].
    This is the MiGPO equivalent of GiGPO's compute_step_discounted_returns.

    Automatically detects environment type:
    - WebShop: Uses dynamic milestone detection (milestone_achieved field)
    - ALFWorld: Uses static milestone matching (trial_id field)

    Args:
        batch: Input batch containing actions, traj_uid, trial_id (ALFWorld) or milestone_achieved (WebShop)
        gamma: Decay factor for reverse-decay rewards (default: 0.99)
        threshold: Similarity threshold for milestone matching (default: 0.85, ALFWorld only)

    Returns:
        step_rewards: Tensor of shape (bs,), scale [0, 10] (matches GiGPO)
    """
    # Auto-detect: WebShop uses milestone_achieved, SciWorld uses subgoal_completed
    if 'milestone_achieved' in batch.non_tensor_batch or 'subgoal_completed' in batch.non_tensor_batch:
        return compute_migpo_step_rewards_dynamic(batch, gamma)

    # ALFWorld: Use static milestone matching
    from .milestone_loader import get_milestone_loader
    loader = get_milestone_loader()

    actions = batch.non_tensor_batch.get('actions', None)
    traj_uids = batch.non_tensor_batch['traj_uid']
    trial_ids = batch.non_tensor_batch.get('trial_id', None)

    bsz = len(batch)
    all_step_rewards = np.zeros(bsz, dtype=np.float32)

    if actions is None or trial_ids is None:
        return torch.zeros(bsz, device=batch.batch['input_ids'].device)

    unique_traj_uids = np.unique(traj_uids)

    for uid in unique_traj_uids:
        traj_indices = np.where(traj_uids == uid)[0]
        traj_trial_id = trial_ids[traj_indices[0]]
        milestones = loader.get_milestones(traj_trial_id)

        if milestones is None:
            continue

        traj_actions = [actions[i] for i in traj_indices]
        match_indices = match_milestones(traj_actions, milestones, threshold)
        rewards, _ = compute_segment_rewards(len(traj_actions), match_indices, gamma)

        # Scale by 10x to match GiGPO's [0, 10] scale
        rewards = rewards * 10.0

        for local_idx, global_idx in enumerate(traj_indices):
            all_step_rewards[global_idx] = rewards[local_idx]

    return torch.tensor(all_step_rewards, dtype=torch.float32,
                        device=batch.batch['input_ids'].device)


# ---------------------------------------------------------- #
# --------------- Utility Functions ------------------------ #
# ---------------------------------------------------------- #

def are_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """
    Check whether two text observations are similar enough.

    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.

    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity matching.")
    return SequenceMatcher(None, a, b).ratio() >= threshold


# ---------------------------------------------------------- #
# --------------- Episode-level Advantage ------------------ #
# ---------------------------------------------------------- #

def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.ndarray,
                        traj_index: np.ndarray,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ) -> torch.Tensor:
    """
    Compute episode-level advantage using mean-std normalization.
    (Identical to GiGPO's episode_norm_reward)

    Args:
        token_level_rewards: Tensor of shape (bs, response_length).
        response_mask: Tensor of shape (bs, response_length).
        index: Array of episode group indices.
        traj_index: Array of trajectory indices.
        epsilon: Small value for numerical stability.
        remove_std: If True, only subtract mean (no std normalization).
        compute_mean_std_cross_steps: If True, compute stats across steps within group.

    Returns:
        episode_advantages: Tensor of shape (bs, response_length).
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                id2std[idx] = torch.std(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


# ---------------------------------------------------------- #
# --------------- Core MiGPO Functions --------------------- #
# ---------------------------------------------------------- #

def match_milestones(actions: List[str],
                     milestones: List[str],
                     threshold: float = 0.85) -> List[int]:
    """
    Sequentially match trajectory actions with milestone sequence.

    For each milestone m_k, find the first action a_t that satisfies:
    Sim(a_t, m_k) >= threshold

    Args:
        actions: List of action strings in the trajectory.
        milestones: List of milestone action strings (ordered).
        threshold: Similarity threshold for matching (default: 0.85).

    Returns:
        match_indices: List of action indices for each milestone.
                      -1 indicates the milestone was not matched.
    """
    k = 0  # Current milestone pointer
    match_indices = []

    for t, action in enumerate(actions):
        if k >= len(milestones):
            break
        if are_similar(action, milestones[k], threshold):
            match_indices.append(t)
            k += 1

    # Mark unmatched milestones with -1
    while len(match_indices) < len(milestones):
        match_indices.append(-1)

    return match_indices


def compute_segment_rewards(traj_length: int,
                            match_indices: List[int],
                            gamma: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rewards for each action using reverse-decay within segments.

    For each matched segment [prev_end + 1, match_idx]:
    - Key action (match_idx) gets reward = 1.0
    - Previous actions get reward = gamma^(distance_to_key)

    For unmatched segments (failed attempts):
    - All actions get reward = 0.0

    Args:
        traj_length: Total number of actions in the trajectory.
        match_indices: List of matched action indices for each milestone.
        gamma: Decay factor for reverse-decay (default: 0.99).

    Returns:
        rewards: np.array of shape (traj_length,) with reward values.
        segment_ids: np.array of shape (traj_length,) with segment IDs.
    """
    rewards = np.zeros(traj_length, dtype=np.float32)
    segment_ids = np.full(traj_length, -1, dtype=np.int32)
    prev_end = -1

    for seg_idx, match_idx in enumerate(match_indices):
        if match_idx == -1:
            # Unmatched milestone: remaining actions are "failed attempts"
            for t in range(prev_end + 1, traj_length):
                segment_ids[t] = seg_idx  # Assign to this milestone's segment
                rewards[t] = 0.0  # Failed attempt gets 0 reward
            break

        # Successfully matched segment: [prev_end + 1, match_idx]
        for t in range(prev_end + 1, match_idx + 1):
            segment_ids[t] = seg_idx
            distance_to_key = match_idx - t
            rewards[t] = gamma ** distance_to_key  # Key action gets 1.0

        prev_end = match_idx

    return rewards, segment_ids


def build_segment_groups(segment_ids: np.ndarray,
                         index: np.ndarray,
                         summarize: bool = False) -> np.ndarray:
    """
    Group steps by (episode_index, segment_id) for advantage computation.

    Steps with the same episode index and segment ID are grouped together,
    allowing for within-group advantage normalization.

    Args:
        segment_ids: Array of segment IDs for each step.
        index: Array of episode group indices.
        summarize: Whether to print group size statistics.

    Returns:
        group_uids: Array of unique group UIDs for each step.
    """
    group_uids = np.empty(len(segment_ids), dtype=object)

    # Create mapping from (index, segment_id) to UUID
    key_to_uid = {}
    group_sizes = []

    for i in range(len(segment_ids)):
        key = (index[i], segment_ids[i])
        if key not in key_to_uid:
            key_to_uid[key] = str(uuid.uuid4())
        group_uids[i] = key_to_uid[key]

    # Calculate group sizes for statistics
    if summarize:
        uid_counts = defaultdict(int)
        for uid in group_uids:
            uid_counts[uid] += 1
        group_sizes = list(uid_counts.values())
        print(f"Number of groups: {len(group_sizes)}")
        print(f"Avg group size: {np.mean(group_sizes):.2f}")
        print(f"Min group size: {min(group_sizes)}, Max group size: {max(group_sizes)}")

    return group_uids


def segment_norm_reward(step_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        group_uids: np.ndarray,
                        epsilon: float = 1e-6,
                        remove_std: bool = True) -> torch.Tensor:
    """
    Compute segment-level advantage using mean normalization within groups.

    For each group, compute:
    - If remove_std=True: advantage = reward - mean(group_rewards)
    - If remove_std=False: advantage = (reward - mean) / (std + epsilon)

    Args:
        step_rewards: Tensor of shape (bs,) with step-level rewards.
        response_mask: Tensor of shape (bs, response_length) for masking.
        group_uids: Array of group UIDs for each step.
        epsilon: Small value for numerical stability.
        remove_std: If True, only subtract mean (default: True).

    Returns:
        advantages: Tensor of shape (bs, response_length).
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # Group scores by UID
        for i in range(bsz):
            id2score[group_uids[i]].append(scores[i])

        # Compute mean and std for each group
        for uid in id2score:
            group_scores = id2score[uid]
            if len(group_scores) == 1:
                id2mean[uid] = torch.mean(torch.stack(group_scores))
                id2std[uid] = torch.tensor(1.0)
            else:
                id2mean[uid] = torch.mean(torch.stack(group_scores))
                id2std[uid] = torch.std(torch.stack(group_scores))

        # Normalize scores
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[group_uids[i]]
            else:
                scores[i] = (scores[i] - id2mean[group_uids[i]]) / (id2std[group_uids[i]] + epsilon)

        # Broadcast to response length
        advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return advantages


def compute_migpo_advantage(token_level_rewards: torch.Tensor,
                            step_rewards: torch.Tensor,
                            response_mask: torch.Tensor,
                            actions: np.ndarray,
                            index: np.ndarray,
                            traj_uid: np.ndarray,
                            trial_id: Optional[np.ndarray] = None,
                            gamma: float = 0.99,
                            threshold: float = 0.85,
                            epsilon: float = 1e-6,
                            step_advantage_w: float = 1.0,
                            mode: str = "mean_norm",
                            milestone_achieved: Optional[np.ndarray] = None,
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages for MiGPO (Milestone-Guided Policy Optimization).

    Main entry point for the MiGPO algorithm. Computes joint advantages:
    scores = episode_advantages + step_advantage_w * step_advantages

    Automatically detects environment type:
    - WebShop: Uses milestone_achieved for dynamic grouping
    - ALFWorld: Uses trial_id for static milestone matching

    Args:
        token_level_rewards: Tensor of shape (bs, response_length).
        response_mask: Tensor of shape (bs, response_length).
        actions: Array of action strings for each step.
        index: Array of episode group indices.
        traj_uid: Array of trajectory UIDs.
        trial_id: Array of trial IDs for milestone lookup (ALFWorld).
        gamma: Decay factor for reverse-decay rewards (default: 0.99).
        threshold: Similarity threshold for milestone matching (default: 0.85, ALFWorld only).
        epsilon: Small value for numerical stability (default: 1e-6).
        step_advantage_w: Weight for step-level advantages (default: 1.0).
        mode: Normalization mode, "mean_norm" or "mean_std_norm" (default: "mean_norm").
        milestone_achieved: Array of milestone achievement flags (WebShop).

    Returns:
        advantages: Tensor of shape (bs, response_length).
        returns: Tensor of shape (bs, response_length) (same as advantages).
    """
    # Determine normalization mode
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    device = token_level_rewards.device
    bsz = token_level_rewards.shape[0]
    response_length = token_level_rewards.shape[1]

    # 1. Compute episode-level advantages (identical to GiGPO)
    episode_advantages = episode_norm_reward(
        token_level_rewards, response_mask, index, traj_uid, epsilon, remove_std
    )

    # 2. Compute step-level advantages using pre-computed step_rewards
    # Auto-detect environment type and build segment groups accordingly

    if milestone_achieved is not None:
        # WebShop: Use dynamic milestone detection
        group_uids = build_segment_groups_dynamic(milestone_achieved, index, traj_uid)
    elif trial_id is not None and trial_id[0] is not None:
        # ALFWorld: Use static milestone matching
        from .milestone_loader import get_milestone_loader
        loader = get_milestone_loader()

        # Initialize segment_ids for grouping
        all_segment_ids = np.zeros(bsz, dtype=np.int32)

        # Process each trajectory to compute segment_ids for grouping
        unique_traj_uids = np.unique(traj_uid)

        for uid in unique_traj_uids:
            traj_indices = np.where(traj_uid == uid)[0]
            traj_trial_id = trial_id[traj_indices[0]]
            milestones = loader.get_milestones(traj_trial_id) if traj_trial_id else None

            if milestones is None:
                for global_idx in traj_indices:
                    all_segment_ids[global_idx] = 0
                continue

            traj_actions = [actions[i] for i in traj_indices]
            match_indices = match_milestones(traj_actions, milestones, threshold)
            _, segment_ids = compute_segment_rewards(len(traj_actions), match_indices, gamma)

            for local_idx, global_idx in enumerate(traj_indices):
                all_segment_ids[global_idx] = segment_ids[local_idx]

        # Build segment groups for advantage computation
        group_uids = build_segment_groups(all_segment_ids, index)
    else:
        # No milestone data available, use only episode-level advantages
        print("[MiGPO Warning] No milestone data (trial_id or milestone_achieved) provided, using only episode-level advantages.")
        return episode_advantages, episode_advantages

    # Compute segment-level advantages using pre-computed step_rewards
    step_advantages = segment_norm_reward(
        step_rewards, response_mask, group_uids, epsilon, remove_std
    )

    # 3. Compute joint advantages (identical to GiGPO)
    scores = episode_advantages + step_advantage_w * step_advantages

    return scores, scores


# ---------------------------------------------------------- #
# --------------- WebShop Dynamic Milestone Support -------- #
# ---------------------------------------------------------- #

def compute_migpo_step_rewards_dynamic(
    batch: DataProto,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    WebShop 动态里程碑奖励计算

    算法逻辑与 ALFWorld 完全相同：
    - 使用反向衰减奖励：reward = gamma^distance × 10.0
    - 里程碑步骤获得最高奖励（10.0）
    - 前面的步骤按距离衰减

    唯一区别：
    - ALFWorld: 通过 match_milestones() 匹配得到里程碑步骤
    - WebShop: 直接从 milestone_achieved 获取里程碑步骤

    Args:
        batch: 包含 milestone_achieved 和 traj_uid 的批次数据
        gamma: 衰减因子

    Returns:
        step_rewards: [total_tokens] 的步骤奖励张量，范围 [0, 10]
    """
    device = batch.batch['responses'].device
    total_tokens = batch.batch['responses'].shape[0]
    step_rewards = torch.zeros(total_tokens, dtype=torch.float32, device=device)

    # Support both milestone_achieved (WebShop) and subgoal_completed (SciWorld)
    if 'milestone_achieved' in batch.non_tensor_batch:
        milestone_achieved = batch.non_tensor_batch['milestone_achieved']
    elif 'subgoal_completed' in batch.non_tensor_batch:
        milestone_achieved = batch.non_tensor_batch['subgoal_completed']
    else:
        milestone_achieved = []
    traj_uids = batch.non_tensor_batch['traj_uid']

    for traj_uid in np.unique(traj_uids):
        traj_mask = (traj_uids == traj_uid)
        traj_indices = np.where(traj_mask)[0]

        if len(traj_indices) == 0:
            continue

        # 找到达成里程碑的步骤（等价于 ALFWorld 的匹配结果）
        traj_achieved = [milestone_achieved[i] for i in traj_indices]
        milestone_steps = [i for i, achieved in enumerate(traj_achieved) if achieved]

        if not milestone_steps:
            continue

        # 与 ALFWorld 完全相同的反向衰减逻辑
        segment_start = 0
        for milestone_idx in milestone_steps:
            for step_idx in range(segment_start, milestone_idx + 1):
                distance = milestone_idx - step_idx
                reward_value = (gamma ** distance) * 10.0  # 与 ALFWorld 相同
                step_rewards[traj_indices[step_idx]] = reward_value
            segment_start = milestone_idx + 1

    return step_rewards


def build_segment_groups_dynamic(
    milestone_achieved: np.ndarray,
    index: np.ndarray,
    traj_uid: np.ndarray,
) -> np.ndarray:
    """
    为 WebShop 构建段分组（与 ALFWorld 的 build_segment_groups 逻辑相同）

    返回 group_uids，用于 segment_norm_reward 归一化

    Args:
        milestone_achieved: 布尔数组，标记哪些步骤达成了里程碑
        index: episode group indices
        traj_uid: trajectory UIDs

    Returns:
        group_uids: 每个步骤的分组 UID
    """
    group_uids = np.empty(len(milestone_achieved), dtype=object)

    # 为每个步骤计算其所属的里程碑段
    key_to_uid = {}

    for i in range(len(milestone_achieved)):
        # 计算当前步骤属于哪个里程碑段
        segment_id = 0
        for j in range(i):
            if traj_uid[j] == traj_uid[i] and milestone_achieved[j]:
                segment_id += 1

        # group_uid = (episode_index, segment_id)
        key = (index[i], segment_id)
        if key not in key_to_uid:
            key_to_uid[key] = str(uuid.uuid4())
        group_uids[i] = key_to_uid[key]

    return group_uids
