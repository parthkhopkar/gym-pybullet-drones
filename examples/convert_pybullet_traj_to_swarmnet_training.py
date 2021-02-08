import os
import argparse
import numpy as np
from swarms.utils import system_edges

FILES_PATH = '../files/logs/'

def convert_format(file_name):
    print(f'Processing file {file_name}')

    # Read the .npy log file
    with open(FILES_PATH + file_name, 'rb') as file:
        timestamps = np.load(file)
        states = np.load(file)
        goal = np.load(file)
        # controls = np.load(file)

    # Extract timeseries data
    time_data = np.expand_dims(timestamps[0], 0)  # For batches in SwarmNet
    print('time_data shape:', time_data.shape)

    # Extract position and velocity coordinates in 2D
    # Convert dimension of timeseries data from (N, xyz, t) to (t, N+G, xyz')
    timeseries_data = states.transpose(2,0,1)
    timeseries_data = np.delete(timeseries_data, 2, 2)
    timeseries_data = np.delete(timeseries_data, slice(4 ,None, None), 2)
    
    goal_array = np.array([goal[0,0,0], goal[0,1,0], 0., 0.])
    goal_array = np.broadcast_to(goal_array, (timeseries_data.shape[0], 1, 4))  
    # goal_data = goal.transpose(2,0,1)
    # goal_data = goal_data[:,0,:]
    # goal_data = np.expand_dims(goal_data, 1)
    # print(goal_data.shape)
    timeseries_data = np.concatenate((goal_array, timeseries_data), axis=1)
    timeseries_data = np.expand_dims(timeseries_data, 0)                                                                
    print('timeseries data shape: ', timeseries_data.shape)

    # Get edge data from function in Swarms
    edge_data = system_edges(obstacles=0, boids=timeseries_data.shape[2]-1, vicseks=0)
    edge_data = np.expand_dims(edge_data, 0)
    print('edge data shape', edge_data.shape)

    return timeseries_data, edge_data, time_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix for generated data file {train/valid/test}')
    parser.add_argument('--suffix', type=str, default='',
                        help='suffix for generated data file')

    ARGS = parser.parse_args()

    file_list = os.listdir(FILES_PATH)

    timeseries_data_all = []
    edge_data_all = []
    time_data_all = []


    for file_name in file_list:
        if file_name.startswith('save'):
            time_series, edges, times = convert_format(file_name)
            timeseries_data_all.append(time_series)
            edge_data_all.append(edges)
            time_data_all.append(times)

    timeseries_data_all = np.concatenate(timeseries_data_all)
    edge_data_all = np.concatenate(edge_data_all) 
    time_data_all = np.concatenate(time_data_all)

    print(f'timeseries data shape in main: {timeseries_data_all.shape}')

    # Save edge, timeseriies and time .npy files
    np.save(f'./data/{ARGS.prefix}_timeseries_{ARGS.suffix}.npy', timeseries_data_all)
    np.save(f'./data/{ARGS.prefix}_edge_{ARGS.suffix}.npy', edge_data_all)
    np.save(f'./data/{ARGS.prefix}_time_{ARGS.suffix}.npy', time_data_all)