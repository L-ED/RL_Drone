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

class HoverFullState(BaseRL):

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

        self.elem_num = 16

        self.max_step = max_step

        self.max_g = 2*9.8
        self.max_ang_vel = 2 
        self.max_radius = 1

        self.target_pos = np.array([0, 0, 1])
        self.last_action = np.zeros(4)
        self.randomize = True
        self.validation = False

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

        max_disp = self.max_radius
        # max_vel = self.

        # print("OBS", 'rank', self.rank, observation)
        # pos, ang, vel, a_vel, acc, a_acc = observation['FS_0'] 
        pos = observation['FS_0'][0]
        ang = observation['FS_0'][1]
        proj_grav = observation['FS_0'][2]

        world_lin_vel = observation['FS_0'][3]
        world_ang_vel = observation['FS_0'][4]

        local_lin_vel = observation['FS_0'][5]
        local_ang_vel = observation['FS_0'][6]

        imu = observation['IMU_0'] 
        targ_disp = self.target_pos - pos

        stats = [
            proj_grav,
            local_ang_vel,
            local_lin_vel, 
            targ_disp,
            self.last_action
        ]


        # stats = [
        #     pos,
        #     ang,
        #     world_ang_vel,
        #     world_lin_vel, 
        #     # imu[:3],
        #     # imu[3:],
        #     # a_acc, 
        #     # acc,
        #     targ_disp
        # ]

        # for i in range(len(stats)):
        #     value = stats[i]
        #     value_norm = np.linalg.norm(value)
        #     if value_norm != 0:
        #         value = value/value_norm 
        #     stats[i] = value

        return np.concatenate(stats).reshape((1, self.elem_num))
        # return np.concatenate(stats).reshape((1, 12))
    

    def check_termination(self):
        
        term = False
        pos = np.copy(self.drone.state.world.pos)
        if pos[2] < 0.05:
            term = True

        pos[2] -= 1
        # is_out = sum(pos**2) > self.max_radius**2
        is_out = max(pos) > self.max_radius
        if is_out:
            term = True
        
        return term
    

    def check_truncation(self):
        trunc = False
        if self.step_idx > self.max_step:
            trunc = True
        return trunc and not self.validation
    

    def create_initial_state(self):
        state = super().create_initial_state()
        if self.randomize:
            new_pos = np.random.rand(3)
            # new_pos = np.zeros(3)*2
            new_pos[:2] -= 0.5
            new_pos = new_pos*self.max_radius
            new_pos[2] = max(new_pos[2], 0.2)
        else:
            new_pos = np.zeros(3)
            new_pos[2] = 0.2

        state.world.pos = new_pos
        return state
    

    def create_initial_action(self):
        return np.zeros(4)
    

    def reward(self):

        # safe_radius= 0.3
        safe_radius= self.max_radius

        state = deepcopy(self.drone.state)

        disp = self.target_pos - state.world.pos
        displ_dir = disp/np.linalg.norm(disp)

        displ_normalized = np.sum(disp**2)/(safe_radius)**2

        vel = state.world.vel
        flight_dir = vel/np.linalg.norm(vel)

        vel_normalized = np.sum(vel**2)/(self.drone.max_speed/2)**2

        closenes_reward = (1-displ_normalized)#*(1-vel_normalized)

        dir_reward = np.dot(flight_dir, displ_dir)

        # if pos[2] < 0.05

        if np.sum(disp**2)<0.2:
            # closenes_reward +=1
            dir_reward=1

        # print(state.world.ang_vel, state.local.ang_vel)
        angles_reward = np.exp(-np.linalg.norm(state.world.ang_vel)*0.3) 
        # print(closenes_reward)
        reward = closenes_reward*angles_reward
        # reward = closenes_reward + angles_reward + dir_reward*0.1

        
        return reward
