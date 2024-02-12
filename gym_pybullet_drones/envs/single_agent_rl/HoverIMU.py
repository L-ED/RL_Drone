from gym_pybullet_drones.envs.single_agent_rl import BaseRL
from gymnasium import spaces
import numpy as np
import pybullet as pb

from gym_pybullet_drones.vehicles import QuadCopter
from gym_pybullet_drones.devices import Camera, mpu6000
from gym_pybullet_drones.utils.state import State


class HoverIMU(BaseRL):

    def __init__(
            self, 
            client=None, 
            drone=None, 
            control_system=None, 
            logger=None, 
            scene_objects=[], 
            visualize=True, 
            record=False, 
            realtime=False
        ):

        if client is None:
            client = pb.connect(pb.DIRECT)
        
        if drone is None:

            sensors= [
                mpu6000()
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

        sens_names = [
            name for name in self.observation_space.keys() 
            if "IMU" in name]

        assert len(sens_names) == 1

        self.imu_name = sens_names[0]
        imu_obs_space = self.observation_space[self.imu_name]
        self.observation_space = spaces.Box(
            low=-1*np.ones(imu_obs_space.shape),
            high=np.ones(imu_obs_space.shape),
            dtype=np.float32
        )

    
    def preprocess_action(self, action):
        return (action+1)*self.drone.max_rps/2
        

    def preprocess_observation(self, observation):
        max_g = 5*self.G
        obs = None
        for sens_name, sens_obs_space in observation:
            if "IMU" in sens_name:
                sens_obs_space = np.clip(sens_obs_space, -max_g, max_g)/max_g        
                obs = sens_obs_space
        return obs

    def check_termination(self):
        
        term = False
        pos = np.copy(self.drone.state.world.pos)
        pos[2] -= 1
        is_out = sum(pos**2) > 4
        if is_out:
            term = True
        return term
    

    def create_initial_state(self):
        state = super().create_initial_state()
        new_pos = np.random.rand(3)*4 - 2
        new_pos[2] = max(new_pos[2], 0.2)
        state.world.pos = new_pos
        return state
    

    def create_initial_action(self):
        return np.zeros(4)
    

    def reward(self):

        state = np.copy(self.drone.state)

        disp = np.array([0, 0, 1]) - state.world.pos
        displ_dir = disp/np.linalg.norm(disp)
        
        vel = state.world.vel
        flight_dir = vel/np.linalg.norm(vel)

        # print('Position: ', pos, " Velocity: ", vel)
        # print('Displacement ', disp, "Disp_dir ", displ_dir)
        # print("Velocity dir ", flight_dir)

        # p.addUserDebugLine(
        #     pos, pos+pos*2000000000
        # )

        # p.addUserDebugLine(
        #     np.array([0, 0, 1]), np.array([0, 0, 1])+np.array([0, 0, 2]), [0, 1, 0]
        # )


        # print(flight_dir, displ_dir)

        ang_cos = np.dot(flight_dir, displ_dir)
        # print("Cos between dirs", ang_cos)
        # print(ang_cos)

        reward = 100*ang_cos - np.linalg.norm(np.abs(vel)) - \
                10*np.sum(disp**2)- \
                10*np.linalg.norm(np.abs(state.world.rpy)) - \
                100*np.linalg.norm(np.abs(state.world.ang_vel))
        
        return reward
