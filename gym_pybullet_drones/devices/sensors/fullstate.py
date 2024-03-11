from gym_pybullet_drones.devices import Device
import numpy as np
import pybullet as pb
from gymnasium import spaces

class FullState(Device):
    name_base = 'FS'

    def __init__(self, frequency, base=None) -> None:
        super().__init__(frequency, base)

        # self.initial_pos = None
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,)
            # low=-np.inf, high=np.inf, shape=(1,)
        )

    def make_obs(self):
        return np.concatenate(
            (self._base.state.world.pos, self._base.state.world.rpy, self._base.state.world.vel, self._base.state.world.ang_vel))