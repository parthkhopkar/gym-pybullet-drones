import time
import argparse
import random
import gym
import numpy as np
import pybullet as p
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

Z = 1
ENV_NORM = 20
DT = 0.3
CTRL_FREQ = int(DT*240)
with open('../files/chaser_data/motion_data_chaser/c0/agent2.txt', 'r') as agent_data:
    trajectory = agent_data.readlines()
    print(len(trajectory))
    print(trajectory)
    init_x, init_y = [float(pos)/ENV_NORM for pos in trajectory[0].split(' ')[0:2]]
    env = TakeoffAviary(gui=True, record=True, act=ActionType.SIX_D_PID, initial_xyzs=np.array([[init_x, init_y, Z]]))

    PYB_CLIENT = env.getPyBulletClient()

    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    obs = env.reset()
    for i in range(len(trajectory)):
        duck_x, duck_y = [float(pos)/ENV_NORM for pos in trajectory[i].split(' ')[:2]]
        p.loadURDF("duck_vhacd.urdf", [duck_x,duck_y,0.05], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=PYB_CLIENT)
        
    start = time.time()
    next_x, next_y = [float(pos.rstrip('\n'))/ENV_NORM for pos in trajectory[0].split(' ')[0:2]]
    vel_x, vel_y = [float(vel.rstrip('\n'))/ENV_NORM for vel in trajectory[0].split(' ')[2:4]]

    for i in range(((len(trajectory)-1)*CTRL_FREQ)):
        obs, reward, done, info = env.step([next_x, next_y, Z, vel_x, vel_y, 0])
        if i%CTRL_FREQ == 0 and i != 0:
            next_x, next_y = [float(pos.rstrip('\n'))/ENV_NORM for pos in trajectory[int(i/CTRL_FREQ) + 1].split(' ')[0:2]]
            vel_x, vel_y = [float(vel.rstrip('\n'))/ENV_NORM for vel in trajectory[int(i/CTRL_FREQ) + 1].split(' ')[2:4]]

        if i%env.SIM_FREQ == 0:
            # print(f'Iteration: {int(i/env.SIM_FREQ) + 1}, {last_x}, {last_y}')
            print(i, info["drone_pos"])
            env.render()
        sync(i, start, env.TIMESTEP)
    env.close() 