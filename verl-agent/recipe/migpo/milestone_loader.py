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
Milestone data loader for MiGPO algorithm.

Loads expert trajectory milestones from JSON file and provides
efficient lookup by trial_id.
"""

import json
import os
from typing import List, Optional, Dict


class MilestoneLoader:
    """
    Singleton class to load and cache milestone data.

    Loads milestone actions from a JSON file once and provides
    efficient lookup by trial_id.

    JSON format:
    {
        "trajectories": [
            {
                "id": "task_type-Object-..._trial_T...",
                "task_type": "pick_heat_then_place_in_recep",
                "actions": ["action1", "action2", ...]
            },
            ...
        ]
    }
    """

    _instance: Optional['MilestoneLoader'] = None
    _milestones: Dict[str, List[str]] = {}
    _loaded: bool = False

    def __new__(cls, json_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, json_path: str = None):
        # Only load once
        if self._loaded:
            return

        if json_path is None:
            # Default path
            json_path = os.path.join(
                os.path.dirname(__file__),
                'alfworld.json'
            )

        if not os.path.exists(json_path):
            print(f"[MilestoneLoader] Warning: JSON file not found at {json_path}")
            self._loaded = True
            return

        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Build trial_id -> actions mapping
        self._milestones = {
            item['id']: item['actions']
            for item in data.get('trajectories', [])
        }

        self._loaded = True
        print(f"[MilestoneLoader] Loaded {len(self._milestones)} milestone trajectories")

    def get_milestones(self, trial_id: str) -> Optional[List[str]]:
        """
        Get milestone actions for a given trial_id.

        Args:
            trial_id: The trial identifier matching JSON 'id' field.
                     Format: "task_type-Object-..._trial_T..."

        Returns:
            List of milestone action strings, or None if not found.
        """
        return self._milestones.get(trial_id, None)

    def has_milestones(self, trial_id: str) -> bool:
        """Check if milestones exist for a given trial_id."""
        return trial_id in self._milestones

    @property
    def num_trajectories(self) -> int:
        """Return the number of loaded trajectories."""
        return len(self._milestones)

    @classmethod
    def reset(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._milestones = {}
        cls._loaded = False


# Convenience function
def get_milestone_loader(json_path: str = None) -> MilestoneLoader:
    """Get or create the MilestoneLoader singleton instance."""
    return MilestoneLoader(json_path)
