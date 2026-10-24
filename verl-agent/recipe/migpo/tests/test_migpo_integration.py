import numpy as np
import torch

from verl import DataProto
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_advantage


def _make_dynamic_migpo_batch():
    return DataProto.from_single_dict(
        {
            "responses": torch.ones((4, 2), dtype=torch.long),
            "attention_mask": torch.ones((4, 4), dtype=torch.long),
            "token_level_rewards": torch.tensor(
                [
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 3.0],
                    [0.0, 3.0],
                ],
                dtype=torch.float32,
            ),
            "step_rewards": torch.tensor([9.0, 10.0, 9.0, 10.0], dtype=torch.float32),
            "uid": np.array(["prompt-a"] * 4, dtype=object),
            "traj_uid": np.array(["traj-a", "traj-a", "traj-b", "traj-b"], dtype=object),
            "actions": np.array(["look", "take apple", "look", "take apple"], dtype=object),
            "milestone_achieved": np.array([False, True, False, True], dtype=bool),
        }
    )


def test_migpo_advantage_estimator_is_registered():
    assert AdvantageEstimator.MiGPO.value == "migpo"


def test_migpo_compute_advantage_uses_step_rewards():
    batch = _make_dynamic_migpo_batch()

    output = compute_advantage(
        batch,
        adv_estimator=AdvantageEstimator.MiGPO,
        migpo_step_advantage_w=1.0,
        migpo_mode="mean_norm",
        migpo_gamma=0.9,
        migpo_threshold=0.95,
    )

    assert output.batch["advantages"].shape == (4, 2)
    assert output.batch["returns"].shape == (4, 2)
    torch.testing.assert_close(output.batch["advantages"], output.batch["returns"])
    torch.testing.assert_close(
        output.batch["advantages"][:, 0],
        torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32),
    )


def test_alfworld_gamefile_derives_migpo_trial_id():
    info = {
        "extra.gamefile": (
            "valid_seen/"
            "pick_clean_then_place_in_recep-ButterKnife-None-DiningTable-16/"
            "trial_T20190909_114508_617301/game.tw-pddl"
        )
    }

    assert (
        TrajectoryCollector._migpo_trial_id_from_info(info)
        == "pick_clean_then_place_in_recep-ButterKnife-None-DiningTable-16_trial_T20190909_114508_617301"
    )


def test_migpo_action_prefers_executed_action():
    assert (
        TrajectoryCollector._migpo_action_from_info(
            {"executed_action": "go to diningtable 1"},
            "<action>go to diningtable 1</action>",
        )
        == "go to diningtable 1"
    )
    assert (
        TrajectoryCollector._migpo_action_from_info(
            {},
            "<action>go to diningtable 1</action>",
        )
        == "<action>go to diningtable 1</action>"
    )
