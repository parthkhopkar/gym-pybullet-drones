import math
import numpy as np
from gym import spaces
import pybullet as p
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from gym_pybullet_drones.utils.swarm_utils import system_edges, one_hot
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary

class GoalAviary(BaseMultiagentAviary):
    """Multi-agent RL problem: Go to goal location while avoiding obstacles."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.PID,
                 goal_pos = [.0, .0, .05],
                 obstacle_present = False,
                 obstacle_pos = [(1., 1., .5)],
                 noise=0.0):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        goal_pos: The 3D position of the goal in the environment

        """
        # SWARMS: Add next 2 lines to make PID control work
        self.DRONE_MODEL = drone_model
        self.NUM_DRONES = num_drones
        self.NUM_NODES = num_drones + len(obstacle_pos) + 1
        self.N_DIM = 2
        self.goal_pos = np.atleast_2d(np.array(goal_pos))
        self.goal_state_2d = np.hstack([goal_pos[:2], 0., 0.])
        self.obstacle_present = np.array(obstacle_present)
        self.obstacle_pos = obstacle_pos
        self.NUM_OBSTACLES = len(obstacle_pos)
        self.OBSTACLE_RADIUS = 0.5  # From sphere2.urdf
        self.COLLISION_BUFFER = 0.05
        self.obstacle_state_2d = np.vstack([np.hstack([pos[:2], 0., 0.]) for pos in obstacle_pos])
        self.edge_types = 4
        self.edges = system_edges(goals=1, obstacles=len(obstacle_pos), boids=self.NUM_DRONES)
        self.one_hot_edges = one_hot(self.edges, num_classes=self.edge_types)
        self.noise = noise
        self.ENV_SIZE = 5.0  # TODO Change according to Swarms training, currently acts as normalizer
        self.drone_distances = np.zeros((num_drones, num_drones))
        self.drone_collision_matrix = np.full((num_drones, num_drones), False)
        self.drone_obstacle_distances = np.zeros((num_drones, self.NUM_OBSTACLES))
        self.drone_obstacle_collision_matrix = np.full((num_drones, self.NUM_OBSTACLES), False)
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by ID in integer format,
            each containing info about all the nodes in the system and the edge connections

        """
        return spaces.Dict({i: spaces.Dict({'nodes': spaces.Box(low=-np.inf, high=np.inf, shape=(self.NUM_NODES, self.N_DIM*2)),
                                            'edges': spaces.MultiBinary([self.NUM_NODES, self.NUM_NODES, self.edge_types]),
                                           })
                            for i in range(self.NUM_DRONES)})
    ################################################################################
          
    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        # reward for dist from goal
        # reward for interagent dist and collision
        rewards = {}
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        # Update drone distance and collision matrices
        for i in range(self.NUM_DRONES):
            for j in range(self.NUM_DRONES):
                if i!= j:
                    self.drone_distances[i,j] = np.linalg.norm(states[i, 0:3] - states[j, 0:3])
                    self.drone_collision_matrix[i, j] = True if self.drone_distances[i,j] < (2*self.L + self.COLLISION_BUFFER) \
                                                        else False 

            for k in range(self.NUM_OBSTACLES):
                # Assume that obstacle is always at same height as drone and use only X,Y coord for calculation
                self.drone_obstacle_distances[i,k] = np.linalg.norm(states[i, 0:2] - self.obstacle_pos[k][0:2])
                self.drone_obstacle_collision_matrix[i, k] = True if self.drone_obstacle_distances[i, k] < (self.OBSTACLE_RADIUS + self.L + self.COLLISION_BUFFER) \
                                                            else False

        # Determine reward for each drone
        for i in range(self.NUM_DRONES):
            # Determine dist from goal without considering Z
            distance_to_goal = np.linalg.norm(states[i, 0:2] - self.goal_pos[0,0:2])
            goal_reward = 0.2 * (1 / (distance_to_goal / self.ENV_SIZE + 0.1) - 0.8)

            # Interagent distance reward
            agent_dist_reward = 0.0
            for j in range(self.NUM_DRONES):
                 agent_dist_reward += 0.05 * (0 - self.drone_distances[i, j] / self.ENV_SIZE) * \
                                      (1 - self.drone_collision_matrix[i, j]) - \
                                      10. * self.drone_collision_matrix[i, j]

            # Obstacle collision penalty
            obstacle_collision_penalty = 0.0
            for k in range(self.NUM_OBSTACLES):
                obstacle_collision_penalty += -10. * self.drone_obstacle_collision_matrix[i,k]

            
            rewards[i] = goal_reward + agent_dist_reward + obstacle_collision_penalty
        return rewards

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and 
            one additional boolean value for key "__all__".

        """
        # TODO: Is a drone done when it collides with something?
        done = {i: True if np.any(self.drone_collision_matrix[i,:]) or np.any(self.drone_obstacle_collision_matrix[i,:]) \
                else False for i in range(self.NUM_DRONES)}
        done['__all__'] = True in done.values() and False not in done.values()
        done['__any__'] = True if True in done.values() else False
        return done

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        # SWARMS: Added position and velocity return to info
        return {i: {'position': self.pos[i], 'velocity': self.vel[i]} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

    ################################################################################

    # Overrides `_computeObs()` in BaseMultiAgentAviary
    # TODO: Change observation space in BaseMultiAgentAviary
    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN: 
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            # return {   i   : self._clipAndNormalizeState(self._getDroneStateVector(i)) for i in range(self.NUM_DRONES) }
            ############################################################

            # Create noisy goal and state observations
            goal_state_noisy = self.goal_state_2d * (1.0 + np.random.uniform(low=-self.noise, high=self.noise))
            obstacle_state_noisy = self.obstacle_state_2d * (1.0 + np.random.uniform(low=-self.noise, high=self.noise))
            #### OBS SPACE OF SIZE 4
            obs_all = {}  # Actual observations w/o noise
            obs_all_noise = {}  # Actual observations w/ noise
            obs = {}  # Observations w/ noise and edges to be returned
            # Get all observations
            for i in range(self.NUM_DRONES):
                obs_i_20 = self._getDroneStateVector(i)
                obs_i_4 = np.hstack([obs_i_20[0:2], obs_i_20[10:12]]).reshape(4,)  # We need X Y Z VX VY VZ or X Y VX VY
                obs_all[i] = obs_i_4
                obs_all_noise[i] = obs_i_4 * (1.0 + np.random.uniform(low=-self.noise, high=self.noise))

            # Compute obs according to obs space w/ noise and add edges
            for i in range(self.NUM_DRONES):
                obs_i = np.vstack([obs_all[j] if j == i else obs_all_noise[j] for j in range(self.NUM_DRONES)])
                obs[i] = {'nodes': np.vstack((goal_state_noisy, obstacle_state_noisy, obs_i)),
                          'edges': self.one_hot_edges}
            
            return obs
            ############################################################
        else:
            print("[ERROR] in BaseMultiagentAviary._computeObs()")

    def _addObstacles(self):
        """Add sphere obstacles to environment at given positons and also
        goal rubber duck

        Overrides BaseMultiAgentAviary's method.
        """
        for obstacle_pos in self.obstacle_pos:
            p.loadURDF("sphere2.urdf",
                       obstacle_pos,
                       p.getQuaternionFromEuler([0, 0, 0]),
                       globalScaling = 1.,
                       useFixedBase=1,
                       physicsClientId=self.CLIENT
                       )
        p.loadURDF("duck_vhacd.urdf", self.goal_pos[0],  physicsClientId=self.CLIENT)