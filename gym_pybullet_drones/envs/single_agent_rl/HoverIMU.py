from gym_pybullet_drones.envs.single_agent_rl import BaseRL
from gymnasium import spaces
import numpy as np

class HoverIMU(BaseRL):

    def __init__(self, client, drone, control_system=None, logger=None, scene_objects=[], visualize=True, record=False, realtime=False):
        super().__init__(client, drone, control_system, logger, scene_objects, visualize, record, realtime)
        

    def normalize_observation_space(self):
        new_obs_space = {}
        for sens_name, sens_obs_space in self.observation_space.items():
            if "IMU" in sens_name:
                new_obs_space[sens_name] = spaces.Box(
                    low=-1*np.ones(sens_obs_space.shape),
                    high=np.ones(sens_obs_space.shape),
                    dtype=np.float32
                )
            else:
                new_obs_space[sens_name] = sens_obs_space
        self.observation_space = spaces.Dict(new_obs_space)

    
    def preprocess_action(self, action):
        return (action+1)*self.drone.max_rps/2
        

    def preprocess_observation(self, observation):
        max_g = 5*self.G
        for sens_name, sens_obs_space in observation:
            if "IMU" in sens_name:
                sens_obs_space = np.clip(sens_obs_space, -max_g, max_g)/max_g


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
        new_pos[2] = max(new_pos[2], 0)
        state.world.pos = new_pos
        return state
    

    def create_initial_action(self):
        return np.zeros(4)