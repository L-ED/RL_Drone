from typing import Any, Dict, SupportsFloat, Tuple
from gym_pybullet_drones.envs import Base

from gymnasium import spaces
import numpy as np


class BaseRL(Base):
    def __init__(self, client, drone, control_system=None, logger=None, scene_objects=[], visualize=True, record=False, realtime=False):
        super().__init__(client, drone, control_system, logger, scene_objects, visualize, record, realtime)

        self.normalize_action_space()
        self.normalize_observation_space()


    def normalize_action_space(self):
        self.action_space = spaces.Box(
            low=-np.ones(4),
            high=np.ones(4),
            dtype=np.float32
        )
    

    def normalize_observation_space(self):
        pass

    def reset_buffers(self):
        pass

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        action = self.preprocess_action(action)
        # obs, reward, terminated, truncated, info = super().step(action)
        obs, reward, terminated, truncated, info = super().step(action)
        # done = terminated or truncated
        obs = self.preprocess_observation(obs)
        return obs, reward, terminated, truncated, info
    

    def reset(self, 
              seed=None,
              options=None):
        
        self.reset_buffers()
        seed = np.random.randint(0, 65000)
        np.random.seed(seed=seed)
        obs, inf =  super().reset()
        return self.preprocess_observation(obs), inf