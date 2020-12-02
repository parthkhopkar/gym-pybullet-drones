import time
import argparse
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
import ray
from ray.tune import register_env
from ray.rllib.agents import ppo

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType

env = TakeoffAviary(gui=True, record=False, act=ActionType.PID, initial_xyzs=np.array([[0,0,1]]))
print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)
obs = env.reset()
start = time.time()
action = np.array([0 , 0, 0.1])
for i in range(10*env.SIM_FREQ):
    obs, reward, done, info = env.step(action)
    # print(f'x:{obs[0]:.3f}  y:{obs[1]:.3f}  z:{obs[2]:.3f}')
    if i%120 == 0:
        action[:1] += 0.1# np.random.choice([0.1,0.2])
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    sync(i, start, env.TIMESTEP)
    # if done:
    #     obs = env.reset()
    #     print('Resetting....')
env.close()