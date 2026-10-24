# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
from agent_system.environments.env_package.alfworld.deterministic_sharding import (
    build_worker_game_file_shards,
)

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env, game_files=None, fixed_train_cases=None):
        self.seed = seed
        self.base_env = base_env
        self.env = None
        self._current_game_files = None
        self.fixed_train_cases = (
            list(fixed_train_cases) if fixed_train_cases is not None else None
        )
        self.fixed_train_reset_idx = 0
        if self.fixed_train_cases is not None:
            self._replace_game_files([self.fixed_train_cases[0]["gamefile"]])
        else:
            self._replace_game_files(game_files)

    def _replace_game_files(self, game_files=None):
        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()
        if game_files is not None:
            self.base_env.game_files = list(game_files)
            self.base_env.num_games = len(self.base_env.game_files)
        self._current_game_files = (
            tuple(self.base_env.game_files) if game_files is not None else None
        )
        self.env = self.base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(self.seed)

    def _next_fixed_train_case(self):
        if self.fixed_train_cases is None:
            return None
        if self.fixed_train_reset_idx >= len(self.fixed_train_cases):
            raise RuntimeError(
                "Fixed ALFWorld training schedule is exhausted at reset "
                f"{self.fixed_train_reset_idx + 1}."
            )
        fixed_case = self.fixed_train_cases[self.fixed_train_reset_idx]
        self.fixed_train_reset_idx += 1
        gamefile = fixed_case["gamefile"]
        if self._current_game_files != (gamefile,):
            self._replace_game_files([gamefile])
        return fixed_case
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        fixed_train_case = self._next_fixed_train_case()
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        if fixed_train_case is not None:
            infos['fixed_train_case_id'] = [fixed_train_case['case_id']]
            infos['fixed_train_step'] = [fixed_train_case['step']]
            infos['fixed_train_slot'] = [fixed_train_case['slot']]
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        worker_game_file_shards = None
        fixed_train_case_shards = env_kwargs.get('fixed_train_case_shards')
        fixed_game_files = env_kwargs.get('fixed_game_files')
        deterministic_eval_sharding = bool(env_kwargs.get('deterministic_eval_sharding', False))
        if fixed_train_case_shards is not None:
            if not is_train:
                raise ValueError("fixed_train_case_shards can only be used by ALFWorld training environments")
            if len(fixed_train_case_shards) != self.num_processes:
                raise ValueError(
                    "Fixed ALFWorld training shards must match the number of workers: "
                    f"{len(fixed_train_case_shards)} != {self.num_processes}"
                )
            reset_count = len(fixed_train_case_shards[0]) if fixed_train_case_shards else 0
            print(
                "Fixed ALFWorld training enabled: "
                f"{env_num} task slots x {reset_count} reset batch(es) x {group_n} rollout(s)."
            )
        elif fixed_game_files is not None:
            if is_train:
                raise ValueError("fixed_game_files can only be used by ALFWorld validation environments")
            worker_game_file_shards = build_worker_game_file_shards(
                fixed_game_files,
                env_num=env_num,
                group_n=group_n,
                preserve_order=True,
            )
            games_per_task_slot = len(worker_game_file_shards[0])
            print(
                "Fixed ALFWorld validation enabled: "
                f"{env_num} task slots x {games_per_task_slot} games x {group_n} rollout(s)."
            )
        elif deterministic_eval_sharding and not is_train:
            worker_game_file_shards = build_worker_game_file_shards(
                base_env.game_files,
                env_num=env_num,
                group_n=group_n,
            )
            games_per_task_slot = len(worker_game_file_shards[0])
            print(
                "Deterministic ALFWorld validation sharding enabled: "
                f"{env_num} task slots x {games_per_task_slot} games x {group_n} rollout(s)."
            )

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            game_files = None if worker_game_file_shards is None else worker_game_file_shards[i]
            fixed_train_cases = (
                None if fixed_train_case_shards is None else fixed_train_case_shards[i]
            )
            worker = env_worker.remote(
                config,
                seed + (i // self.group_n),
                base_env,
                game_files,
                fixed_train_cases,
            )
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
