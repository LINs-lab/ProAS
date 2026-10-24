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
MiGPO: Milestone-Guided Policy Optimization

A reinforcement learning algorithm that uses milestone-based reward shaping
to guide policy optimization in multi-step agent tasks.
"""

from .core_migpo import (
    compute_migpo_advantage,
    compute_migpo_step_rewards,
    match_milestones,
    compute_segment_rewards,
    build_segment_groups,
    segment_norm_reward,
)

__all__ = [
    "compute_migpo_advantage",
    "compute_migpo_step_rewards",
    "match_milestones",
    "compute_segment_rewards",
    "build_segment_groups",
    "segment_norm_reward",
]
