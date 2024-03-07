from gym_pybullet_drones.envs.single_agent_rl import BaseRL
from gymnasium import spaces
import numpy as np
import pybullet as pb

from gym_pybullet_drones.vehicles import QuadCopter
from gym_pybullet_drones.devices import Camera, mpu6000, Barometer
from gym_pybullet_drones.utils.state import State
from torch import sigmoid
from copy import deepcopy

class HoverIMU(BaseRL):

    def __init__(
            self, 
            client=None, 
            drone=None, 
            control_system=None, 
            logger=None, 
            scene_objects=[], 
            visualize=False, 
            record=False, 
            realtime=False
        ):

        self.max_g = 5*9.8
        self.max_ang_vel = 10 #10 
        self.max_radius = 2


        if client is None:
            if visualize:
                client = pb.connect(pb.GUI)
            else:
                client = pb.connect(pb.DIRECT)

        if drone is None:

            sensors= [
                mpu6000(), 
                Barometer(1e3)
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
        

    def normalize_observation_space(self):

        # sens_names = [
        #     name for name in self.observation_space.keys() 
        #     if "IMU" in name]

        # assert len(sens_names) == 1

        # self.imu_name = sens_names[0]
        # imu_obs_space = self.observation_space[self.imu_name]
        # self.observation_space = spaces.Box(
        #     low=-1*np.ones(imu_obs_space.shape),
        #     high=np.ones(imu_obs_space.shape),
        #     dtype=np.float32
        # )

        self.observation_space = spaces.Box(
            low=-1*np.ones((1, 7)),
            high=np.ones((1, 7)),
            dtype=np.float32
        )


    
    def preprocess_action(self, action):
        return self.drone.max_rpm/(1+np.exp(-action))
        

    def preprocess_observation(self, observation):
        # obs = None
        # for sens_name, sens_obs_space in observation.items():
        #     if "IMU" in sens_name:
        #         sens_obs_space = np.clip(sens_obs_space, -self.max_g, self.max_g)/self.max_g        
        #         obs = sens_obs_space
        # return obs
        imu = observation["IMU_0"]
        imu[:3] = np.clip(imu[:3], -self.max_g, self.max_g)/self.max_g
        imu[3:] = np.clip(imu[3:], -self.max_ang_vel, self.max_ang_vel)/self.max_ang_vel

        return np.concatenate((
            imu, 
            np.array([observation["Bar_0"]/(2*self.max_radius)])
        )).reshape((1, 7))
    

    def check_termination(self):
        
        term = False
        pos = np.copy(self.drone.state.world.pos)
        pos[2] -= 1
        is_out = sum(pos**2) > self.max_radius**2
        if is_out:
            term = True
        return term
    

    def create_initial_state(self):
        state = super().create_initial_state()
        # new_pos = np.random.rand(3)*2 - 2
        new_pos = np.zeros(3)*2
        new_pos[2] = max(new_pos[2], 0.2)
        state.world.pos = new_pos
        return state
    

    def create_initial_action(self):
        return np.zeros(4)
    

    def reward(self):

        state = deepcopy(self.drone.state)

        disp = np.array([0, 0, 1]) - state.world.pos
        displ_dir = disp/np.linalg.norm(disp)

        displ_normalized = np.sum(disp**2)/self.max_radius**2

        vel = state.world.vel
        flight_dir = vel/np.linalg.norm(vel)

        vel_normalized = np.sum(vel**2)/self.drone.max_speed**2

        closenes_reward = (1-displ_normalized)*(1-vel_normalized)

        # print('Position: ', pos, " Velocity: ", vel)
        # print('Displacement ', disp, "Disp_dir ", displ_dir)
        # print("Velocity dir ", flight_dir)

        # print(flight_dir, displ_dir)

        dir_reward = np.dot(flight_dir, displ_dir)
        angles_reward = -np.linalg.norm(state.world.ang_vel) 

        reward = dir_reward + angles_reward #+ closenes_reward 
        
        return reward
