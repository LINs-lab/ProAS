# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

'''
main(config)：Hydra 把合成后的 config 注入，再调用 run_ppo。
    config_path="config"：相对当前包/工作目录下的 config 目录找配置。
    config_name="ppo_trainer"：默认用 ppo_trainer.yaml（及 composition）。
    version_base=None：不按 Hydra 1.x 的 version_base 行为变更，保持当前项目习惯。
'''
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)



def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized(): 
        # 确认本地的ray 尚未初始化
        # 若本进程里 Ray 还没 init，再初始化，避免多次 ray.init 冲突。
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        
        
        # 用 config.ray_init 里的 runtime_env 做深度合并
        default_runtime_env = get_ppo_ray_runtime_env() # get_ppo_ray_runtime_env() 提供默认 runtime_env；
        
        # 获取 config.ray_init 里的 runtime_env 配置
        ray_init_kwargs = config.get("ray_init", {}) # 先获得 ray_init，命名为 ray_init_kwargs
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {}) # 再从 ray_init_kwargs 里获得 runtime_env，命名为 runtime_env_kwargs

        # 使用 OmegaConf.merge 把 default_runtime_env 和 runtime_env_kwargs 做深度合并，得到最终的 runtime_env
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        
        # 把合并结果写回 ray_init 的参数字典。
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        
        # 打印实际传给 ray.init 的参数
        print(f"ray init kwargs: {ray_init_kwargs}")
        # 然后 **OmegaConf.to_container(ray_init_kwargs) 展开成普通 dict 传给 ray.init
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # 创建远程 TaskRunner 实例
    runner = TaskRunner.remote()
    # ray.get 同步等待其 run(config) 跑完。
    # 也就是说：主进程只负责起 Ray 和等结果，重活都在 TaskRunner 的 actor 里。
    ray.get(runner.run.remote(config))
'''
小科普：

runtime_env 是什么？

    在 Ray 里，runtime_env 是传给 ray.init(...) 或 ray.remote(..., runtime_env=...) 的一段运行环境描述。
    Ray 会按这份描述，在真正执行 task / actor 的 worker 进程里准备好环境，而不是只在你敲命令的那个 shell 里生效。
    可以把它理解成：**「这些远程进程启动时，要带什么环境」**的配置。

    有什么用？
    常见用途包括：
    1. 环境变量（env_vars）
    让每个 Ray worker 里都有统一的 TOKENIZERS_PARALLELISM、NCCL_*、VLLM_* 等，避免和训练机本地 
    shell 不一致导致 tokenizer/NCCL/vLLM 行为怪异。

    2. 工作目录 / 代码（working_dir、py_modules 等）
    把某目录打成包同步到 worker，或指定依赖，保证集群上各节点跑的是同一份代码、
    同一套依赖（你这份 PPO 里还通过 RAY_JOB_CONFIG_JSON_ENV_VAR 判断是否已有 working_dir，避免重复设置）。

    3. 依赖（pip、conda 等，可选）
    在 worker 里自动装包（你们当前 constants_ppo 里主要用的是 env_vars，没有在这里写 pip）。

你们项目里具体在干什么？
    get_ppo_ray_runtime_env() 返回的主要是 env_vars，例如 tokenizer 并行、NCCL 调试级别、
    vLLM 日志与 LoRA、以及为避免 hang/crash 设的 NCCL_CUMEM_ENABLE、CUDA_DEVICE_MAX_CONNECTIONS 
    等（见 constants_ppo.py 里的 PPO_RAY_RUNTIME_ENV）。

    另外它会：若某个变量在 OS 环境里已经设过，就不再写进 runtime_env，避免覆盖你手动 export 的值；
    若 Ray Job 配置里已有 working_dir，也不会再强行加 working_dir: None。

    在 main_ppo.py 里，这份默认 runtime_env 会和 Hydra 里 config.ray_init.runtime_env 合并，
    再交给 ray.init(**...)，这样你可以在 YAML 里追加或覆盖环境，而不必改 Python 默认值。

一句话：
    runtime_env 是 Ray 的「远程 worker 进程环境说明书」；你们主要用它来统一分布式训练相关的环境
    变量（tokenizer、NCCL、vLLM），并与用户配置合并。
'''




'''
    @ray.remote(num_cpus=1) 占 1 个 CPU，避免把重任务绑在 head 节点调度策略上（注释写了要保证 main 相关逻辑不都在 head 上挤）。
'''
@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        
        # 在 worker 里再 import，减轻 driver 与序列化压力
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        # pprint 打印解析后配置
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config) # OmegaConf.resolve(config) 把配置里的插值/符号都解析成最终值。

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # 创建训练用环境 envs 和验证用 val_envs（和 agent+环境 rollout 相关）。
        from agent_system.environments import make_envs
        envs, val_envs = make_envs(config)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            # 若 rollout 后端是 vllm 且开了 LoRA（lora_rank > 0），检查 vLLM 版本 ≥ 0.7.3，否则直接报错，避免不支持的组合。
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        # 按 actor 并行策略选 worker 类
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            # 若 actor 并行策略是 fsdp 或 fsdp2，则 critic 也必须是 fsdp 或 fsdp2
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            # 从 fsdp_workers 取 ActorRolloutRefWorker 等
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            # 若 rollout.mode == "async" 用异步版
            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup # RayWorkerGroup 做进程组。

        # 若 actor 用megatron 后端并行
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            #：critic 策略必须一致；用 NVMegatronRayWorkerGroup 和 megatron 版 worker。
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        
        # ResourcePoolManager 管理「角色 → GPU 池」
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # role_worker_mapping：每个 Role 对应已 ray.remote 装饰的 worker 类。
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        '''
        小科普：
        
        为什么 role_worker_mapping 里还要 ray.remote(...)？
           核心就一点：ActorRolloutRefWorker / CriticWorker 在源码里是普通 Python 类，
           不是 Ray 自带的 Actor 类型；只有经过 ray.remote(类) 之后，Ray 才知道怎么在集群里起独立进程、
           调度资源、用 .remote() 创建实例。

            例如在 fsdp_workers.py 里定义是：
            class ActorRolloutRefWorker(Worker):
                """
                This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
                or a hybrid engine based on the config.rollout
                """

                def __init__(self, config: DictConfig, role: str):
            这里没有 @ray.remote。
            
            RayPPOTrainer 会把 mapping 里的类交给 RayClassWithInitArgs，再在 RayWorkerGroup 
            里按 GPU 池 spawn 多个 Actor。传给 Ray 的必须是「已经 actor 化」的类，例如：
             actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            所以：mapping 里存的不是「业务 Worker 类」本身，而是「可被 Ray 当作 Actor 创建的那份包装」。

        为什么不直接在 fsdp_workers.py 里写 @ray.remote？
            1. 运行时二选一：同一套逻辑里要在 ActorRolloutRefWorker 和 AsyncActorRolloutRefWorker 之间切换，
                在定义处写死 @ray.remote 不如在 main_ppo 里对最终选中的类做一次 ray.remote(...) 干净。
            2.资源和选项：ray.remote 还可以带 num_gpus、runtime_env 等；在入口统一包一层，便于和配置对齐
                （若写在类上，每个变体都要重复装饰或再 .options()）。
            3.职责分离：Worker 文件主要写「分布式训练逻辑」；是否、如何被 Ray 挂成 Actor 由 trainer 入口决定，
                测试或别的后端也可能只想 import 普通类。

        小结：
            普通类 ActorRolloutRefWorker: 
                只是 Python 类，Ray 不会自动当 Actor 用, 适合写在 workers 模块里
            ray.remote(ActorRolloutRefWorker): 
                适合作为 role_worker_mapping 里交给 RayWorkerGroup 的类型
        '''



        global_pool_id = "global_pool"
        # 全局池 global_pool_id，
        # 每节点 GPU 数 n_gpus_per_node，共 nnodes 个元素，即多机 GPU 布局。
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # mapping：ActorRollout、Critic 都进同一 global_pool。
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        
        # 若 config.reward_model.enable，按 fsdp/megatron 选 RewardModelWorker，
        # 并加入 Role.RewardModel 的 mapping。
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        # 若算法在 reward 里用 KL，或 actor 用 KL loss，则增加 RefPolicy 角色，
        # 用同一 ActorRolloutRefWorker 类当参考策略。
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        
        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        # reward_manager 默认名 episode 时用 EpisodeRewardManager；
        if reward_manager_name == 'episode':
            from agent_system.reward_manager import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        # reward_fn 用于训练时算奖励（num_examine=0
        reward_fn = reward_manager_cls(tokenizer=tokenizer, 
                            num_examine=0, 
                            normalize_by_length=False)

        # Note that we always use function-based RM for validation
        # val_reward_fn 用于验证（num_examine=1，便于多打一点日志/检查）。
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, 
                            num_examine=1,
                            normalize_by_length=False)

        # ResourcePoolManager 实例化，后面 trainer 按池分配资源。
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # 断言 rollout.n == 1：在 verl+环境里 GRPO 不靠 rollout.n>1，而是用 env.rollout.n，避免和纯 verl 的 GRPO 配置混淆。
        assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        
        # TrajectoryCollector 负责多轮轨迹收集（和 agent 系统对接）。
        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(
                config=config, 
                tokenizer=tokenizer, 
                processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn

        # 若 data_config.custom_cls.path 有值，用 load_extern_type 动态加载自定义 Dataset 子类。
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        # RandomSampler + 固定 seed（data_config.seed，默认 1），便于可复现和与 checkpoint 恢复时的顺序一致。
        # 用 sampler 是为了更好的断点续训行为。
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
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


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    '''
    只有直接 python main_ppo.py 时才会走 Hydra 入口；被别的模块 import 时不会误启动训练。
    Hydra 读配置 → run_ppo 初始化 Ray → 远程 TaskRunner 下模型、建环境/数据集、选 FSDP 或 Megatron worker、组资源池与奖励 → RayPPOTrainer.init_workers() + fit()。
    '''
    main()
