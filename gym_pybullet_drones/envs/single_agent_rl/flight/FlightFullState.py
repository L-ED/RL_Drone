from gym_pybullet_drones.envs.single_agent_rl import BaseRL
from gymnasium import spaces
import numpy as np
import pybullet as pb

from gym_pybullet_drones.vehicles import QuadCopter
from gym_pybullet_drones.devices import Camera, mpu6000, Barometer, FullState
from gym_pybullet_drones.utils.state import State
from torch import sigmoid
from copy import deepcopy
import torch

class FlightFullState(BaseRL):

    def __init__(
            self, 
            client=None, 
            drone=None, 
            control_system=None, 
            logger=None, 
            scene_objects=[], 
            visualize=False, 
            record=False, 
            realtime=False, 
            max_step = 200, 
            seed = 42, 
            rank = 0

        ):

        self.rank = rank
        self.set_seed(seed)

        self.elem_num = 17

        self.max_step = max_step

        self.max_g = 2*9.8
        self.max_ang_vel = 2 #10 

        self.alpha = 0.01
        # self.max_speed = deepcopy(drone.max_speed)/2
        self.command = np.array(
            [1, 0, 0, 1]
        )

        self.last_action = np.zeros(4)
        self.randomize = True

        if client is None:
            if visualize:
                client = pb.connect(pb.GUI)
            else:
                client = pb.connect(pb.DIRECT)

        if drone is None:

            sensors= [
                FullState(50),
                mpu6000(50)
            ]

            state = State()
            state.world.pos[2] += 0.1

            drone = QuadCopter(
                client=client,
                filename= 'custom.urdf',#'cf2x_cam.urdf',
                # sensors = [],
                sensors = sensors,
                state=state
            )

        super().__init__(client, drone, control_system, logger, scene_objects, visualize, record, realtime)
        

    def set_seed(self, seed):
        np.random.seed(seed)
        # torch.random.seed(seed)
        torch.random.manual_seed(seed)
        

    def normalize_observation_space(self):

        self.observation_space = spaces.Box(
            low=-1*np.ones((1, self.elem_num)),
            high=np.ones((1, self.elem_num)),
            dtype=np.float32
        )


    
    def preprocess_action(self, action):
        self.last_action = action.copy()
        return self.drone.max_rpm/(1+np.exp(-action))
    

    def reset_buffers(self):
        self.last_action = np.zeros(4)


    # def reset(self, seed=None, options=None):
    #     # action = self.create_initial_action()
    #     # obs = self.drone.step(action, self)
    #     obs, inf = super().reset()
    #     # print('IM HERE')
    #     return self.preprocess_observation(obs), inf



    def preprocess_observation(self, observation):

        pos = observation['FS_0'][0]
        ang = observation['FS_0'][1]
        proj_grav = observation['FS_0'][2]

        world_lin_vel = observation['FS_0'][3]
        world_ang_vel = observation['FS_0'][4]

        local_lin_vel = observation['FS_0'][5]
        local_ang_vel = observation['FS_0'][6]

        imu = observation['IMU_0'] 

        stats = [
            proj_grav,
            local_ang_vel,
            local_lin_vel, 
            self.command,
            self.last_action
        ]

        return np.concatenate(stats).reshape((1, self.elem_num))
        # return np.concatenate(stats).reshape((1, 12))
    

    def check_termination(self):
        
        term = False
        
        return term
    

    def check_truncation(self):
        trunc = False
        if self.step_idx > self.max_step:
            trunc = True
        return trunc
    

    def create_initial_state(self):
        state = super().create_initial_state()
        new_pos = np.array([0, 0, 20])
        command = np.random.rand(4)
        command[3] *= self.drone.max_speed/(2-self.alpha)
        command[:2] *= self.alpha
        command[:3] /= np.linalg.norm(command[:3])
        self.command = command
        # print(self.command)
        state.world.pos = new_pos
        return state
    

    def create_initial_action(self):
        return np.zeros(4)
    

    def reward(self):

        state = deepcopy(self.drone.state)

        vel = state.world.vel
        flight_mag = np.linalg.norm(vel)
        flight_dir = vel/flight_mag
        
        dir_reward = np.dot(flight_dir, self.command[:3])
        mag_diff = np.exp(-(flight_mag - self.command[3])**2)
        
        reward = dir_reward*mag_diff #+ dir_reward*0.3
        
        return reward


    def set_alpha(self, alpha):
        print('setting alpha to ', alpha)
        self.alpha = alpha