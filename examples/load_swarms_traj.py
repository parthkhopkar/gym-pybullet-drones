# Script to load Swarms traj in interactive mode
import numpy as np

with open('./data/train_edge_2agentrand.npy', 'rb') as file:
    edge_data_all = np.load(file)