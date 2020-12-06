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

from swarms import Environment2D, Boid, Goal, Sphere

Z = 0.5
ENV_NORM = 20
DT = 0.3
CTRL_FREQ = int(DT*240)
init_x, init_y = -1, 0
# Create Swarms env
# TODO: Get actual env bounds
env2d = Environment2D([20, 20, 20, 20])
env2d.add_agent(Boid([init_x, init_y], ndim=2, max_speed = 8))
env2d.add_goal(Goal([1,1], ndim=2))
env2d.add_obstacle(Sphere(size=0.8, position=[0.5,0.5], ndim=2))


env = TakeoffAviary(gui=True, record=True, act=ActionType.PID, initial_xyzs=np.array([[init_x, init_y, 0.8]]))

PYB_CLIENT = env.getPyBulletClient()

p.loadURDF("sphere2.urdf", [0.5,0.5,0.5], 	useFixedBase=1, physicsClientId=PYB_CLIENT)
p.loadURDF("duck_vhacd.urdf", [1,1,0.05],  physicsClientId=PYB_CLIENT)

print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)

start = time.time()

for i in range(20*env.SIM_FREQ):
    env2d.update(DT)
    vel_x, vel_y = env2d.population[0].velocity
    obs, reward, done, info = env.step([vel_x, vel_y, 0])
    env2d.population[0].position = info["drone_pos"][0][:2]
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    sync(i, start, env.TIMESTEP)
env.close()