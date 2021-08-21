from seagul.rl.ars.ars_np_queue import ARSAgent, postprocess_default
#from common import *
from seagul.mesh import identity
import seagul.envs

import copy
import gym
import time
import xarray as xr
import numpy as np
import os

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

import torch as torch

#env_names = ["Hopper-v2", "Walker2d-v2", "HalfCheetah-v2" ]
env_names = ['tree_simple-v0']
#env_names = ["HalfCheetah-v2"]
#env_names = ["Hopper-v2"]
#post_fns = [madodiv, variodiv]

#env_names = ["Humanoid-v2"]
#env_names = ["MinitaurBulletEnv-v0"]
post_fns = [identity]# variodiv, madodiv]

torch.set_default_dtype(torch.float32)
num_experiments = len(post_fns)
num_seeds = 4
num_steps = int(2e5)
n_workers = 1

save_dir = "./data_ppo_tree0"
#env_config = {}


if __name__ == "__main__":

    assert not os.path.isdir(save_dir)

    torch.set_default_dtype(torch.float32)

    start = time.time()
    for env_name in env_names:
        for i in range(num_seeds):
            seed = int(np.random.randint(0,2**32-1))
            agent_dir = f"{save_dir}/{env_name}/agent_{seed}/"
            
            os.makedirs(agent_dir, exist_ok=True)
            
            def make_env(env_id, e):
                def _init():
                    set_random_seed(seed + e)
                    return Monitor(gym.make(env_name), filename=agent_dir)
                return _init
            
            env_list = [make_env(env_name, e) for e in range(n_workers)]
            venv = DummyVecEnv(env_list)
            env = VecNormalize(venv)
            
            
            agent = PPO('MlpPolicy',
                        env,
                        verbose=2,
                        seed = int(seed),
                        )
            
        
            
            agent.learn(total_timesteps=num_steps)
            
            print(f"{env_name}, {i}, {time.time() - start}")
            agent.save(agent_dir + "model.zip")
            env.save(agent_dir + "env.zip")


