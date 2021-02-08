import time
import gym
import argparse
import multiprocessing
import numpy as np
import pybullet as p
from swarms.utils import system_edges

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType

from swarms import Environment2D, Boid, Goal, Sphere

config = {
        "cohesion": 2e5,
        "separation": 2e5,
        "alignment": 2e5,
        "obstacle_avoidance": 2e6,
        "goal_steering": 2e10
    }
Boid.set_model(config)
# The Z position at which to intitialize the drone
# When using velocity based PID control, same value of Z
# needs to be set in Base{Single,Multi}agentAviary.py
Z = 0.5
N_DRONES = 5
timeseries_data_all = []
edge_data_all = []
time_data_all = []
pybullet_data_all = []

def generate_traj(sample_num):
    start_time = time.time()
    # Number of entries in the list determines number of drones
    initial_positions = [np.concatenate((np.random.RandomState().uniform(-1.5,1.5,2), [Z])) for i in range(N_DRONES)]
    # goal_x, goal_y = 1.5, 1.5
    goal_x, goal_y = np.random.RandomState().uniform(-1,1,2)
    obstacle_x, obstacle_y = 0.5, 0.5
    obstacle_present = False
    # Create Swarms env
    # TODO: Get actual env bounds
    env2d = Environment2D([20, 20, 20, 20])
    goal = Goal([goal_x, goal_y], ndim=2)
    env2d.add_goal(goal)
    timeseries = []  # Append lists of (N_DRONES + G + O, dim) to this list
    edge_data = system_edges(obstacles=0, boids=N_DRONES, vicseks=0)
    times = []  # Append lists of T size to this list
    pybullet_data = []
    # TODO: Determine env2d config for Boids that is guaranteed to be safe for drone env
    for position in initial_positions:
        agent = Boid(position[:2], ndim=2, size=0.12, max_speed = 10, max_acceleration=5)
        agent.set_goal(goal)
        env2d.add_agent(agent)


    if obstacle_present:
        env2d.add_obstacle(Sphere(size=1.5, position=[obstacle_x, obstacle_y], ndim=2))


    env = FlockAviary(gui=False, record=False, num_drones=len(initial_positions), act=ActionType.PID, initial_xyzs=np.array(initial_positions), aggregate_phy_steps=int(5))
    logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=len(initial_positions))
    DT = 1/env.SIM_FREQ
    PYB_CLIENT = env.getPyBulletClient()

    # Initialize obstacle and goal in the drone env
    if obstacle_present:
        p.loadURDF("sphere2.urdf", [obstacle_x, obstacle_y,0.5], globalScaling = 0.5, useFixedBase=1, physicsClientId=PYB_CLIENT)
    # p.loadURDF("duck_vhacd.urdf", [goal_x, goal_y,0.05],  physicsClientId=PYB_CLIENT)

    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    start = time.time()
    # Initialize action dict, (x,y,z) velocity PID control
    action = {i:[0,0,0] for i in range(len(env2d.population))}
    for i in range(12*int(env.SIM_FREQ/env.AGGR_PHY_STEPS)):
        env2d.update(DT)
        # Log updated positions
        system_state = [env2d.goals[0].position.tolist() + env2d.goals[0].velocity.tolist()]
        system_state += [env2d.population[agent].position.tolist() + env2d.population[agent].velocity.tolist() \
                        for agent in range(len(env2d.population))]
        system_state = np.array(system_state)
        timeseries.append(system_state)
        times.append(i/env.SIM_FREQ)

        # Get velocity from Swarms
        for agent in range(len(env2d.population)):
            action[agent][:2] = env2d.population[agent].velocity
        obs, reward, done, info = env.step(action)

        # Update position and velocity of agents in Swarms
        for agent in range(len(env2d.population)):
            env2d.population[agent].position = info[agent]['position'][:2]
            env2d.population[agent].velocity = info[agent]['velocity'][:2]
        pybullet_state = [info[agent]['position'][:2].tolist() + info[agent]['velocity'][:2].tolist() \
                        for agent in range(len(env2d.population))]
        pybullet_data.append(pybullet_state)
        # if i%env.SIM_FREQ == 0:
        #     env.render()
        #     # print(done)
        sync(i, start, env.TIMESTEP)
    
    timeseries = np.array(timeseries)
    times = np.array(times)
    pybullet_data = np.array(pybullet_data)



    env.close()
    time_taken = time.time() - start_time
    print(f'{sample_num} took {time_taken} seconds')
    return timeseries, edge_data, times, pybullet_data
    # logger.plot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1,
                        help='number of sample trajectories to generate')
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix for generated data file {train/valid/test}')
    parser.add_argument('--suffix', type=str, default='',
                        help='suffix for generated data file')

    ARGS = parser.parse_args()

    starttime = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(generate_traj, [i for i in range(ARGS.samples)])
    
    for result in results:
        timeseries_data_all.append(result[0])
        edge_data_all.append(result[1])
        time_data_all.append(result[2])
        pybullet_data_all.append(result[3])

    timeseries_data_all = np.array(timeseries_data_all)
    edge_data_all = np.array(edge_data_all)
    time_data_all = np.array(time_data_all)
    pybullet_data_all = np.array(pybullet_data_all)

    print('Time taken = {} seconds'.format(time.time() - starttime))
    print(f'timeseries shape: {timeseries_data_all.shape} | edges shape: {edge_data_all.shape} | time shape: {time_data_all.shape}')

    # SNR analysis
    print(f'{"Trajectory":10} | {"std_x":10} | {"std_y":10} | {"baseline":10} | {"std_x > baseline?":17} | {"std_y > baseline?":17}')
    for i in range(timeseries_data_all.shape[0]):
        # calculate std
        diff_x = pybullet_data_all[i,:,:,1] - timeseries_data_all[i,:,1:,1]  # T x N
        diff_y = pybullet_data_all[i,:,:,2] - timeseries_data_all[i,:,1:,2]  # T x N
        # print(f'diff_x shape: {diff_x.shape} diff_y shape: {diff_y.shape}')
        std_x = np.std(np.sum(diff_x, axis=1)/N_DRONES)
        std_y = np.std(np.sum(diff_y, axis=1)/N_DRONES)
        # print(f'std_x: {std_x} std_y: {std_y}')

        # calculate baseline
        diff = timeseries_data_all[i,1:,1:,:2] - timeseries_data_all[i,:-1,1:,:2]  # (T-1) x N x d
        # print(f'diff shape: {diff.shape}')
        l1_norm = np.linalg.norm(diff, ord=1, axis=-1)  # (T-1) X N x 1
        # print(f'l1 norm shape: {l1_norm.shape}')
        baseline = np.sum(l1_norm)/(2*N_DRONES*(timeseries_data_all.shape[1]-1))
        # print(baseline)

        # Check
        print(f'{i+1:10d} | {std_x:10.5f} | {std_y:10.5f} | {baseline:10.5f} | {str(std_x > baseline):17} | {str(std_y > baseline):17}')
    # np.save(f'./data/{ARGS.prefix}_timeseries_{ARGS.suffix}.npy', timeseries_data_all)
    # np.save(f'./data/{ARGS.prefix}_edge_{ARGS.suffix}.npy', edge_data_all)
    # np.save(f'./data/{ARGS.prefix}_time_{ARGS.suffix}.npy', time_data_all)
    
    