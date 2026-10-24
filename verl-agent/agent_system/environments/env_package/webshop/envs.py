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

import ray
import gym
import numpy as np

from agent_system.environments.fixed_validation import (
    build_fixed_case_assignments,
    install_fixed_webshop_goal,
    select_webshop_worker_seed,
)
from agent_system.environments.fixed_training import (
    build_fixed_train_case_assignments,
    install_fixed_webshop_train_goal,
)

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
        self.milestone_detector = None
        self.prev_obs = None
        self.fixed_case_metadata = {}
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info.update(self.fixed_case_metadata)
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        if self.milestone_detector is not None:
            verbose_info = None
            if done:
                session_id = self.env.session
                verbose_info = self.env.server.user_sessions.get(session_id, {}).get('verbose_info')

            result = self.milestone_detector.process(
                action=action,
                state=self.prev_obs,
                next_state=obs,
                info={'verbose': verbose_info} if verbose_info else None,
            )
            info['milestone_phase'] = result.phase.value
            info['milestone_achieved'] = result.achieved

        self.prev_obs = obs

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx=None, fixed_case=None):
        """Reset the environment with given session index"""
        if fixed_case is None:
            obs, info = self.env.reset(session=idx)
        elif fixed_case.get("fixed_train") is True:
            session_id = install_fixed_webshop_train_goal(self.env, fixed_case)
            obs, info = self.env.reset(session=session_id)
        else:
            session_id = install_fixed_webshop_goal(self.env, fixed_case)
            obs, info = self.env.reset(session=session_id)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        self.fixed_case_metadata = {}
        if fixed_case is not None and fixed_case.get("fixed_train") is True:
            self.fixed_case_metadata = {
                'fixed_train_case_id': fixed_case['case_id'],
                'fixed_train_step': fixed_case['step'],
                'fixed_train_slot': fixed_case['slot'],
            }
            info.update(self.fixed_case_metadata)
        elif fixed_case is not None:
            info['fixed_val_case_id'] = fixed_case['case_id']
            info['fixed_val_fingerprint'] = fixed_case['exact_fingerprint']

        from recipe.migpo.webshop_milestone_detector import MilestoneDetector
        session_id = self.env.session
        goal_data = self.env.server.user_sessions[session_id]['goal']
        self.milestone_detector = MilestoneDetector(
            goal_data,
            self.env.server.product_item_dict,
            self.env.server.product_prices,
        )
        self.prev_obs = obs
        info['milestone_phase'] = self.milestone_detector.phase.value
        info['milestone_achieved'] = False

        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        fixed_validation_cases: list[dict] = None,
        fixed_validation_seed: int = None,
        fixed_train_batches: list[dict] = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self.fixed_train_batches = None
        self.fixed_train_reset_idx = 0
        if fixed_train_batches is not None:
            self.fixed_train_batches = list(fixed_train_batches)

        self.fixed_validation_assignments = None
        if fixed_validation_cases is not None:
            if self.fixed_train_batches is not None:
                raise ValueError(
                    "fixed_train_batches and fixed_validation_cases are mutually exclusive"
                )
            if is_train:
                raise ValueError("fixed_validation_cases can only be used by WebShop validation environments")
            if fixed_validation_seed is None:
                raise ValueError("fixed_validation_seed is required with fixed_validation_cases")
            self.fixed_validation_assignments = build_fixed_case_assignments(
                fixed_validation_cases,
                env_num=env_num,
                group_n=group_n,
            )

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker_seed = select_webshop_worker_seed(
                seed,
                i,
                self.group_n,
                fixed_seed=fixed_validation_seed,
            )
            worker = env_worker.remote(worker_seed, self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        # Send reset commands to all workers
        futures = []
        fixed_train_batch = None
        if self.fixed_train_batches is not None:
            if self.fixed_train_reset_idx >= len(self.fixed_train_batches):
                raise RuntimeError(
                    "Fixed WebShop training schedule is exhausted at reset "
                    f"{self.fixed_train_reset_idx + 1}."
                )
            fixed_train_batch = self.fixed_train_batches[self.fixed_train_reset_idx]
            self.fixed_train_reset_idx += 1
            fixed_train_cases = build_fixed_train_case_assignments(
                fixed_train_batch,
                env_num=self.env_num,
                group_n=self.group_n,
            )
            for worker, fixed_case in zip(self._workers, fixed_train_cases):
                fixed_case["fixed_train"] = True
                futures.append(worker.reset.remote(None, fixed_case))
        elif self.fixed_validation_assignments is None:
            idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
            idx = np.repeat(idx, self.group_n).tolist()
            for worker, i in zip(self._workers, idx):
                futures.append(worker.reset.remote(i))
        else:
            for worker, fixed_case in zip(self._workers, self.fixed_validation_assignments):
                futures.append(worker.reset.remote(None, fixed_case))

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            if fixed_train_batch is not None:
                info = dict(info or {})
                info.setdefault("fixed_train_step", fixed_train_batch["step"])
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    fixed_validation_cases: list[dict] = None,
    fixed_validation_seed: int = None,
    fixed_train_batches: list[dict] = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        fixed_validation_cases=fixed_validation_cases,
        fixed_validation_seed=fixed_validation_seed,
        fixed_train_batches=fixed_train_batches,
    )
