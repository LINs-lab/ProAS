from __future__ import annotations

from typing import List, Sequence, TypeVar


T = TypeVar("T")


def build_worker_game_file_shards(
    game_files: Sequence[T],
    env_num: int,
    group_n: int,
    preserve_order: bool = False,
) -> List[List[T]]:
    """Assign disjoint game shards to task slots and duplicate them within each rollout group."""
    if env_num <= 0:
        raise ValueError("env_num must be positive")
    if group_n <= 0:
        raise ValueError("group_n must be positive")
    if len(game_files) % env_num != 0:
        raise ValueError(
            f"The number of games ({len(game_files)}) must be divisible by env_num ({env_num}) "
            "for deterministic validation sharding."
        )

    ordered_games = list(game_files) if preserve_order else sorted(game_files)
    task_shards = [ordered_games[task_slot::env_num] for task_slot in range(env_num)]
    return [list(task_shards[task_slot]) for task_slot in range(env_num) for _ in range(group_n)]
