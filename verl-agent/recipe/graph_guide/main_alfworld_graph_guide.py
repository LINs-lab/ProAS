from __future__ import annotations

import os
from functools import wraps
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


def _patch_graph_guide_trainer(config):
    import verl.trainer.ppo.ray_trainer as ray_trainer

    from recipe.graph_guide.graph_guide_credit import (
        GRAPH_GUIDE_DUMP_KEYS,
        annotate_graph_guide_metadata,
        apply_graph_guide_credit,
        graph_guide_config,
        graph_guide_enabled,
    )

    if not graph_guide_enabled(config):
        return
    graph_cfg = graph_guide_config(config)
    if graph_cfg.progress_reward_enabled:
        if str(config.algorithm.adv_estimator).lower() != "grpo":
            raise ValueError("The ARG progress-reward ablation requires GRPO")
        from recipe.graph_guide.progress_reward import apply_graph_progress_reward

    if not hasattr(ray_trainer, "_graph_guide_original_compute_advantage"):
        ray_trainer._graph_guide_original_compute_advantage = ray_trainer.compute_advantage

        @wraps(ray_trainer._graph_guide_original_compute_advantage)
        def compute_advantage_with_graph_guide(*args, **kwargs):
            if graph_cfg.progress_reward_enabled:
                # The trainer has already applied invalid-action/KL penalties.
                data = args[0] if args else kwargs["data"]
                apply_graph_progress_reward(data, config)
                # PR changes rewards before GRPO; do not apply MUI/APR afterward.
                return ray_trainer._graph_guide_original_compute_advantage(*args, **kwargs)
            data = ray_trainer._graph_guide_original_compute_advantage(*args, **kwargs)
            return apply_graph_guide_credit(data, config)

        ray_trainer.compute_advantage = compute_advantage_with_graph_guide

    if not hasattr(ray_trainer.RayPPOTrainer, "_graph_guide_original_dump_keys"):
        ray_trainer.RayPPOTrainer._graph_guide_original_dump_keys = (
            ray_trainer.RayPPOTrainer._generation_dump_extra_info_keys
        )

        def _generation_dump_extra_info_keys(self):
            keys = list(self._graph_guide_original_dump_keys())
            for key in GRAPH_GUIDE_DUMP_KEYS:
                if key not in keys:
                    keys.append(key)
            return keys

        ray_trainer.RayPPOTrainer._generation_dump_extra_info_keys = _generation_dump_extra_info_keys

    if str(graph_cfg.env_adapter).lower() == "webshop":
        from agent_system.multi_turn_rollout import TrajectoryCollector

        if not hasattr(TrajectoryCollector, "_graph_guide_original_multi_turn_loop"):
            TrajectoryCollector._graph_guide_original_multi_turn_loop = TrajectoryCollector.multi_turn_loop

            def multi_turn_loop_with_graph_guide(self, *args, **kwargs):
                data = self._graph_guide_original_multi_turn_loop(*args, **kwargs)
                return annotate_graph_guide_metadata(data, config)

            TrajectoryCollector.multi_turn_loop = multi_turn_loop_with_graph_guide


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = GraphGuideTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class GraphGuideTaskRunner:
    def run(self, config):
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config, resolve=True))
        _patch_graph_guide_trainer(config)

        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0 and not is_version_ge(
                pkg="vllm",
                minver="0.7.3",
            ):
                raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        import verl.trainer.ppo.ray_trainer as ray_trainer

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            ray_trainer.Role.ActorRollout: global_pool_id,
            ray_trainer.Role.Critic: global_pool_id,
        }
        role_worker_mapping = {
            ray_trainer.Role.ActorRollout: ray.remote(actor_rollout_cls),
            ray_trainer.Role.Critic: ray.remote(CriticWorker),
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[ray_trainer.Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[ray_trainer.Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[ray_trainer.Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[ray_trainer.Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ray_trainer.ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl-agent env training, use env.rollout.n for GRPO groups; "
            "actor_rollout_ref.rollout.n must stay 1."
        )

        from agent_system.multi_turn_rollout import TrajectoryCollector
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = ray_trainer.RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
