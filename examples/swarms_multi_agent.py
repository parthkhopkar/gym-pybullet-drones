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

# The Z position at which to intitialize the drone
# When using velocity based PID control, same value of Z
# needs to be set in Base{Single,Multi}agentAviary.py
Z = 0.8
# Number of entries in the list determines number of drones
initial_positions = [[0,0,Z], [-0.5,0,Z], [0,-0.5,Z]]
goal_x, goal_y = 1, 1
obstacle_x, obstacle_y = 0.5, 0.5
obstacle_present = True
# Create Swarms env
# TODO: Get actual env bounds
env2d = Environment2D([20, 20, 20, 20])
# TODO: Determine env2d config for Boids that is guaranteed to be safe for drone env
for position in initial_positions:
    env2d.add_agent(Boid(position[:2], ndim=2, size=0.06, max_speed = 10, max_acceleration=5))
env2d.add_goal(Goal([goal_x, goal_y], ndim=2))
if obstacle_present:
    env2d.add_obstacle(Sphere(size=0.3, position=[obstacle_x, obstacle_y], ndim=2))


env = FlockAviary(gui=True, record=False, num_drones=len(initial_positions), act=ActionType.PID, initial_xyzs=np.array(initial_positions))
DT = 1/env.SIM_FREQ
PYB_CLIENT = env.getPyBulletClient()

# Initialize obstacle and goal in the drone env
if obstacle_present:
    p.loadURDF("sphere2.urdf", [obstacle_x, obstacle_y,0.5], globalScaling = 0.5, useFixedBase=1, physicsClientId=PYB_CLIENT)
p.loadURDF("duck_vhacd.urdf", [goal_x, goal_y,0.05],  physicsClientId=PYB_CLIENT)

print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)

start = time.time()
# Initialize action dict, (x,y,z) velocity PID control
action = {i:[0,0,0] for i in range(len(env2d.population))}
for i in range(30*env.SIM_FREQ):
    env2d.update(DT)
    # Get velocity from Swarms
    for agent in range(len(env2d.population)):
        action[agent][:2] = env2d.population[agent].velocity
    obs, reward, done, info = env.step(action)
    # Update position and velocity of agents in Swarms
    for agent in range(len(env2d.population)):
        env2d.population[agent].position = info[agent]['position'][:2]
        env2d.population[agent].velocity = info[agent]['velocity'][:2]
    if i%env.SIM_FREQ == 0:
        env.render()
        print(done)
    sync(i, start, env.TIMESTEP)
env.close()