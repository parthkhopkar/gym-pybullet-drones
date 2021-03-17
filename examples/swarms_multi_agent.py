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

config = {
        "cohesion": 2e5,
        "separation": 2e5,
        "alignment": 2e5,
        "obstacle_avoidance": 2e10,
        "goal_steering": 2e10
    }
Boid.set_model(config)
# The Z position at which to intitialize the drone
# When using velocity based PID control, same value of Z
# needs to be set in Base{Single,Multi}agentAviary.py
Z = 0.5
N = 5
# Number of entries in the list determines number of drones
# initial_positions = [[0,0,Z], [2,0,Z], [0,2,Z],[1.5,1.5,Z],[-0.5,0.5,Z]]
initial_positions = [[0,1,Z], [0,1.5,Z], [0,0,Z],[0,-0.5,Z],[0,-1,Z]]
# initial_positions = [np.concatenate((np.random.uniform(-2,2,2), [Z])) for i in range(N)]
goal_x, goal_y = 1., 1.
# goal_x, goal_y = np.random.uniform(-2,2,2)
obstacle_x, obstacle_y = 0.5, 0.5
obstacle_present = True
logging = False

# swarms_info = []  # Swarms states
# pybullet_info = []  # Pubullet states
# Create Swarms env
# TODO: Get actual env bounds
env2d = Environment2D([20, 20, 20, 20])
goal = Goal([goal_x, goal_y], ndim=2)
env2d.add_goal(goal)
# TODO: Determine env2d config for Boids that is guaranteed to be safe for drone env
for position in initial_positions:
    agent = Boid(position[:2], ndim=2, size=0.12, max_speed = 10, max_acceleration=5)
    agent.set_goal(goal)
    env2d.add_agent(agent)


if obstacle_present:
    env2d.add_obstacle(Sphere(size=1.5, position=[obstacle_x, obstacle_y], ndim=2))


env = FlockAviary(gui=True, record=False, num_drones=len(initial_positions), act=ActionType.PID, initial_xyzs=np.array(initial_positions), aggregate_phy_steps=int(5))
logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                num_drones=len(initial_positions))
DT = (1/env.SIM_FREQ)
PYB_CLIENT = env.getPyBulletClient()

# Initialize obstacle and goal in the drone env
if obstacle_present:
    p.loadURDF("sphere2.urdf", [obstacle_x, obstacle_y,0.5], globalScaling = 0.5, useFixedBase=1, physicsClientId=PYB_CLIENT)
p.loadURDF("duck_vhacd.urdf", [goal_x, goal_y,0.05],  physicsClientId=PYB_CLIENT)

print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)

start = time.time()
# Initialize action dict, (x,y,z) velocity PID control
action = {i:[0.,0.,0.] for i in range(len(env2d.population))}
for i in range(12*int(env.SIM_FREQ/env.AGGR_PHY_STEPS)):
    env2d.update(DT)
    swarms_state = []
    # print(env2d.population[0].velocity.tolist())
    swarms_state.extend(env2d.population[0].position.tolist())
    swarms_state.extend(env2d.population[0].velocity.tolist())
    # Get velocity from Swarms
    for agent in range(len(env2d.population)):
        action[agent][:2] = env2d.population[agent].velocity  # Swarms posn and vel
    obs, reward, done, info = env.step(action)
    if logging:
        for j in range(len(initial_positions)):
            logger.log(drone=j,
                    timestamp=i/env.SIM_FREQ,
                    state= np.hstack([obs[j][0:3], np.zeros(4), obs[j][3:15], np.resize(action[j], (4))]),
                    goal=np.array((goal_x, goal_y))
                    )
    # pybullet_state = []
    # pybullet_state.extend(info[0]['position'][:2].tolist())
    # pybullet_state.extend(info[0]['velocity'][:2].tolist())
    # # print(swarms_state, pybullet_state)
    # swarms_info.append(swarms_state)
    # pybullet_info.append(pybullet_state)
    # Update position and velocity of agents in Swarms
    for agent in range(len(env2d.population)):
        env2d.population[agent].position = info[agent]['position'][:2]  # Executed posn and vel
        env2d.population[agent].velocity = info[agent]['velocity'][:2]
    if i%env.SIM_FREQ == 0:
        env.render()
        # print(done)
    sync(i, start, env.TIMESTEP)
env.close()

# with open('./delta_info.npy', 'wb') as out_file:
#     np.save(out_file, np.array(swarms_info))
#     np.save(out_file, np.array(pybullet_info))
# # logger.save()
# # logger.plot()