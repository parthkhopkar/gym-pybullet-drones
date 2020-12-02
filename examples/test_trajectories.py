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
with open('../files/chaser_data/motion_data_chaser/c0/agent2.txt', 'r') as agent_data:
    waypoints = agent_data.readlines()
    print(len(waypoints))
    print(waypoints)
    init_x, init_y = [float(pos)/10 for pos in waypoints[0].split(' ')[:2]]
    env = TakeoffAviary(gui=True, record=False, act=ActionType.PID, initial_xyzs=np.array([[init_x, init_y, Z]]))

    PYB_CLIENT = env.getPyBulletClient()

    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    p.loadURDF("duck_vhacd.urdf", [0,0,0.05], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=PYB_CLIENT)

    obs = env.reset()
    for i in range(len(waypoints)):
        duck_x, duck_y = [float(pos)/10 for pos in waypoints[i].split(' ')[:2]]
        p.loadURDF("duck_vhacd.urdf", [duck_x,duck_y,0.05], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=PYB_CLIENT)
        
    start = time.time()
    last_x, last_y = init_x, init_y
    next_x, next_y = [float(pos)/10 for pos in waypoints[1].split(' ')[:2]]
    dx, dy = (next_x - last_x)/env.SIM_FREQ, (next_y - last_y)/env.SIM_FREQ
    target_x, target_y = last_x, last_y

    for i in range(((len(waypoints)-1)*env.SIM_FREQ)):
        target_x, target_y = target_x + dx, target_y + dy
        obs, reward, done, info = env.step([target_x, target_y, Z])
        if i%env.SIM_FREQ == 0 and i != 0:
            last_x, last_y = next_x, next_y
            next_x, next_y = [float(pos)/10 for pos in waypoints[int(i/env.SIM_FREQ) + 1].split(' ')[:2]]
            dx, dy = (next_x - last_x)/env.SIM_FREQ, (next_y - last_y)/env.SIM_FREQ
            target_x, target_y = last_x, last_y
            # print(f'Iteration: {int(i/env.SIM_FREQ) + 1}, {last_x}, {last_y}')
            print(i, info["drone_pos"])
            env.render()
        sync(i, start, env.TIMESTEP)
    env.close() 