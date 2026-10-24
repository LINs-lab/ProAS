"""Evaluate HF SFT checkpoints with run_sh_fair's online RL validator.

Prompts, sampling, environment stepping, action projection, and metrics are
delegated to TrajectoryCollector and RayPPOTrainer._validate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXED_VAL_ROOT = REPO_ROOT / "recipe/graph_guide/data/fixed_val_set"


def parser_for(environment: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"SFT {environment} evaluation using online RL validation (Ray/vLLM).")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Local HF checkpoint directory saved by fsdp_sft_trainer")
    parser.add_argument("--model_size", choices=("1.5B", "1_5B", "7B"), default="1.5B")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0, help="Environment seed, matching the training run")
    parser.add_argument("--val_file", type=Path, default=FIXED_VAL_ROOT / f"{environment}.jsonl",
                        help="Fixed validation JSONL; its directory must contain the shared manifest/assets")
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--n_gpus", type=int, default=None, help="Defaults to 2 for 1.5B, 4 for 7B")
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Defaults to n_gpus")
    parser.add_argument("--dry_run", action="store_true", help="Print resolved config without loading a model or starting Ray")
    if environment == "alfworld":
        parser.add_argument("--alfworld_data", type=Path,
                            default=Path(os.environ.get("ALFWORLD_DATA", Path.home() / ".cache/alfworld")))
    return parser


def build_config(environment: str, args):
    if environment not in ("alfworld", "webshop"):
        raise ValueError(f"Unsupported environment: {environment}")
    n_gpus = args.n_gpus if args.n_gpus is not None else (4 if args.model_size == "7B" else 2)
    tp = args.tensor_parallel_size if args.tensor_parallel_size is not None else n_gpus
    if n_gpus <= 0 or 16 % n_gpus or tp <= 0 or n_gpus % tp:
        raise ValueError("n_gpus must divide 16, and tensor_parallel_size must divide n_gpus")
    val_file = args.val_file.expanduser().resolve()
    if val_file.name != f"{environment}.jsonl":
        raise ValueError(f"--val_file must be named {environment}.jsonl (the runtime loads assets by name)")
    output = (args.output_json or Path(f"results_sft_{environment}_seed{args.seed}.json")).expanduser().resolve()
    artifacts = output.parent / f"{output.stem}_artifacts"
    config = OmegaConf.load(REPO_ROOT / "verl/trainer/config/ppo_trainer.yaml")
    # Validation settings match run_sh_fair. Training-only roles are disabled;
    # this entrypoint never updates actor weights or resumes optimizer state.
    overrides = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "data.train_files": str(artifacts / "placeholders/train.parquet"),
        "data.val_files": str(artifacts / "placeholders/test.parquet"),
        "data.train_batch_size": 16,
        "data.val_batch_size": 128,
        "data.max_prompt_length": 2048 if environment == "alfworld" else 4096,
        "data.max_response_length": 512,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.return_raw_chat": True,
        "actor_rollout_ref.model.path": str(args.checkpoint.expanduser().resolve()),
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 256,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.use_invalid_action_penalty": True,
        "actor_rollout_ref.actor.invalid_action_penalty_coef": 0.1,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "sync",
        "actor_rollout_ref.rollout.dtype": "bfloat16",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": tp,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.6,
        "actor_rollout_ref.rollout.enable_chunked_prefill": False,
        "actor_rollout_ref.rollout.enforce_eager": False,
        "actor_rollout_ref.rollout.free_cache_engine": False,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.4,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "env.env_name": "alfworld/AlfredTWEnv" if environment == "alfworld" else "Webshop",
        "env.seed": args.seed,
        "env.history_length": 2,
        "env.max_steps": 50 if environment == "alfworld" else 15,
        "env.rollout.n": 8,
        "env.fixed_train_set.path": None,
        "env.fixed_val_set.enabled": True,
        "env.fixed_val_set.root": str(val_file.parent),
        "env.webshop.use_small": True,
        "env.webshop.human_goals": False,
        "trainer.n_gpus_per_node": n_gpus,
        "trainer.nnodes": 1,
        "trainer.val_only": True,
        "trainer.val_before_train": True,
        "trainer.resume_mode": "disable",
        "trainer.total_epochs": 1,
        "trainer.logger": ["console"],
        "trainer.default_local_dir": str(artifacts),
        "trainer.validation_data_dir": str(artifacts / "validation_data"),
    }
    for key, value in overrides.items():
        OmegaConf.update(config, key, value)
    if environment == "alfworld":
        OmegaConf.update(config, "ray_init.runtime_env.env_vars.ALFWORLD_DATA",
                         str(args.alfworld_data.expanduser().resolve()), force_add=True)
    OmegaConf.resolve(config)
    return config, output


def preflight(config, environment: str) -> None:
    checkpoint = Path(config.actor_rollout_ref.model.path)
    if not (checkpoint / "config.json").is_file() or not any(
        list(checkpoint.glob(pattern)) for pattern in ("*.safetensors", "pytorch_model*.bin")
    ):
        raise ValueError(f"Expected HF model config and weights in {checkpoint}; export sharded training state first")
    root = Path(config.env.fixed_val_set.root)
    for filename in ("manifest.json", f"{environment}.jsonl"):
        if not (root / filename).is_file():
            raise FileNotFoundError(root / filename)
    with (root / f"{environment}.jsonl").open() as stream:
        cases = [json.loads(line) for line in stream if line.strip()]
    if len(cases) != 128:
        raise ValueError(f"Aligned validation requires all 128 tasks, found {len(cases)}")
    if environment == "alfworld":
        game_root = Path(config.ray_init.runtime_env.env_vars.ALFWORLD_DATA) / "json_2.1.1"
        paths = (root / "alfworld_gamefiles.txt").read_text().splitlines()
        if paths != [case["gamefile_relative"] for case in cases]:
            raise ValueError("ALFWorld JSONL and gamefile list must have the same 128 tasks in the same order")
        missing = [path for path in paths if not (game_root / path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} ALFWorld games under {game_root}: {missing[:2]}")


def validate_trainer(trainer):
    """Call the online validator once, with no checkpoint resume or training."""
    trainer.init_workers()
    trainer.global_steps = 0
    return trainer._validate()


class ValidationRunner:
    """Ray driver; worker setup follows verl.trainer.main_ppo.TaskRunner."""

    def run(self, config):
        import ray
        from agent_system.environments import make_envs
        from agent_system.multi_turn_rollout import TrajectoryCollector
        from agent_system.reward_manager import EpisodeRewardManager
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.use_shm)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=config.data.trust_remote_code, use_fast=True)
        envs, val_envs = make_envs(config)
        try:
            reward = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
            trainer = RayPPOTrainer(
                config=config, tokenizer=tokenizer, processor=processor,
                role_worker_mapping={Role.ActorRollout: ray.remote(ActorRolloutRefWorker)},
                resource_pool_manager=ResourcePoolManager(
                    resource_pool_spec={"global_pool": [config.trainer.n_gpus_per_node]},
                    mapping={Role.ActorRollout: "global_pool"},
                ),
                ray_worker_group_cls=RayWorkerGroup,
                reward_fn=reward, val_reward_fn=reward,
                device_name=config.trainer.device,
                traj_collector=TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor),
                envs=envs, val_envs=val_envs,
            )
            return {key: float(value) for key, value in validate_trainer(trainer).items()}
        finally:
            for manager in (envs, val_envs):
                manager.envs.close()


def evaluate(config, environment: str, output: Path):
    import ray
    from recipe.graph_guide.prepare_ppo_data import prepare_split
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    preflight(config, environment)
    prepare_split(Path(config.data.train_files), "train", 16)
    prepare_split(Path(config.data.val_files), "test", 128)
    artifacts = Path(config.trainer.default_local_dir)
    OmegaConf.save(config, artifacts / "resolved_config.yaml")
    if ray.is_initialized():
        raise RuntimeError("Run this evaluator in a fresh process so the RL runtime environment is applied")
    kwargs = OmegaConf.to_container(config.ray_init, resolve=True)
    kwargs["runtime_env"] = OmegaConf.to_container(OmegaConf.merge(
        get_ppo_ray_runtime_env(), kwargs.get("runtime_env", {})
    ), resolve=True)
    # sys.path changes in the CLI process alone do not configure Ray workers.
    # Match the launchers and use this checkout even from another working dir.
    runtime_vars = kwargs["runtime_env"].setdefault("env_vars", {})
    runtime_vars["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(REPO_ROOT / "verl"), os.environ.get("PYTHONPATH", "")]
    )
    if environment == "alfworld":
        os.environ["ALFWORLD_DATA"] = config.ray_init.runtime_env.env_vars.ALFWORLD_DATA
    ray.init(**kwargs)
    try:
        runner = ray.remote(num_cpus=1)(ValidationRunner).remote()
        metrics = ray.get(runner.run.remote(config))
    finally:
        ray.shutdown()
    summary = {
        "checkpoint": config.actor_rollout_ref.model.path,
        "environment": environment,
        "seed": config.env.seed,
        "evaluation_protocol": "run_sh_fair_online_validation",
        "tasks": 128,
        "success_rate": metrics["val/success_rate"],
        "metrics": metrics,
        "config": OmegaConf.to_container(config, resolve=True),
        "rollout_engine_seed": config.actor_rollout_ref.rollout.get("seed", 0),
        "validation_data_dir": config.trainer.validation_data_dir,
    }
    if environment == "webshop":
        summary["webshop_task_score"] = metrics["val/webshop_task_score (not success_rate)"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output}")
    return summary


def main(environment: str, argv=None):
    parser = parser_for(environment)
    args = parser.parse_args(argv)
    try:
        config, output = build_config(environment, args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    evaluate(config, environment, output)
