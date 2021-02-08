# Script to delta info betn Swarms and Pybullet env
import numpy as np

with open('./delta_info.npy', 'rb') as file:
    swarms_info = np.load(file)
    pybullet_info = np.load(file)
    print('Swarms info shape', swarms_info.shape)
    print('Pybullet info shape', pybullet_info.shape)
    diff = swarms_info - pybullet_info
    mean = np.mean(diff, axis=0)
    std = np.std(diff, axis=0)
    print(f'Mean: {mean}, Std: {std}')

    swarms_step_diff = np.abs(swarms_info[1:,:] - swarms_info[:-1,:])
    print(np.sum(swarms_step_diff > np.broadcast_to(std, swarms_step_diff.shape), axis = 0))
