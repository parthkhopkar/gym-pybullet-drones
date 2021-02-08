# Script to load Pybullet traj in interactive mode to check
import numpy as np

with open('../files/logs/save-flight-01.26.2021_16.55.41.920682.npy', 'rb') as file:
        timestamps = np.load(file)
        states = np.load(file)
        goal = np.load(file)