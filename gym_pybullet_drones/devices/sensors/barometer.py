
from gym_pybullet_drones.devices import Device
import numpy as np
import pybullet as pb
from gymnasium import spaces

class Barometer(Device):
    name_base = 'Bar'

    def __init__(self, frequency, base=None) -> None:
        super().__init__(frequency, base)

        self.initial_height = None
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,)
            # low=-np.inf, high=np.inf, shape=(1,)
        )

    def make_obs(self):
        
        if self.initial_height is None:
            self.initial_height = self._base.state.world.pos[2]

        return self._base.state.world.pos[2] - self.initial_height
    
