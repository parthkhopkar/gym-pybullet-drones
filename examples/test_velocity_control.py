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
VEL_NORM = 20
with open('../files/chaser_data/motion_data_chaser/c0/agent2.txt', 'r') as agent_data:
    trajectory = agent_data.readlines()
    print(len(trajectory))
    print(trajectory)
    init_x, init_y = [float(pos)/20 for pos in trajectory[0].split(' ')[0:2]]
    env = TakeoffAviary(gui=True, record=False, act=ActionType.PID, initial_xyzs=np.array([[init_x, init_y, Z]]))

    PYB_CLIENT = env.getPyBulletClient()

    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    obs = env.reset()
    for i in range(len(trajectory)):
        duck_x, duck_y = [float(pos)/20 for pos in trajectory[i].split(' ')[:2]]
        p.loadURDF("duck_vhacd.urdf", [duck_x,duck_y,0.05], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=PYB_CLIENT)
        
    start = time.time()
    vel_x, vel_y = [float(vel.rstrip('\n'))/VEL_NORM for vel in trajectory[0].split(' ')[2:4]]

    for i in range(((len(trajectory)-1)*72)):
        obs, reward, done, info = env.step([vel_x, vel_y, 0])
        if i%72 == 0 and i != 0:
            vel_x, vel_y = [float(vel.rstrip('\n'))/VEL_NORM for vel in trajectory[int(i/72) + 1].split(' ')[2:4]]

        if i%env.SIM_FREQ == 0:
            # print(f'Iteration: {int(i/env.SIM_FREQ) + 1}, {last_x}, {last_y}')
            print(i, info["drone_pos"])
            env.render()
        sync(i, start, env.TIMESTEP)
    env.close() 