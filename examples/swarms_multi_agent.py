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
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType

from swarms import Environment2D, Boid, Goal, Sphere

Z = 0.5
ENV_NORM = 20
DT = 0.3
CTRL_FREQ = int(DT*240)
initial_positions = [[0,0,Z], [1,0,Z], [0,1,Z]]
# Create Swarms env
# TODO: Get actual env bounds
env2d = Environment2D([20, 20, 20, 20])
for position in initial_positions:
    env2d.add_agent(Boid(position[:2], ndim=2, size=0.1,max_speed = 3))
env2d.add_goal(Goal([1,1], ndim=2))
# env2d.add_obstacle(Sphere(size=0.8, position=[0.5,0.5], ndim=2))


env = FlockAviary(gui=True, record=False, num_drones=len(initial_positions), act=ActionType.PID, initial_xyzs=np.array(initial_positions))

PYB_CLIENT = env.getPyBulletClient()

# p.loadURDF("sphere2.urdf", [0.5,0.5,0.5], 	useFixedBase=1, physicsClientId=PYB_CLIENT)
p.loadURDF("duck_vhacd.urdf", [1,1,0.05],  physicsClientId=PYB_CLIENT)

print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)

start = time.time()
action = {i:[0,0,0] for i in range(len(env2d.population))}
for i in range(20*env.SIM_FREQ):
    env2d.update(DT)
    # Get velocity from Swarms
    for agent in range(len(env2d.population)):
        action[agent][:2] = env2d.population[agent].velocity
    obs, reward, done, info = env.step(action)
    # Update position of agents in Swarms
    for agent in range(len(env2d.population)):
        env2d.population[agent].position = info[agent][:2]
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    sync(i, start, env.TIMESTEP)
env.close()