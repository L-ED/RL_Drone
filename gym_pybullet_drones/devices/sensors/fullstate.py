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
            low=-np.inf, high=np.inf, shape=(18,)
            # low=-np.inf, high=np.inf, shape=(1,)
        )

    def make_obs(self):
        proj_grav = np.dot(
            self._base.state.R.T,
            np.array([0, 0, -9.8])
        )

        return (
            np.copy(self._base.state.world.pos), 
            np.copy(self._base.state.world.rpy), 
            np.copy(proj_grav), 
            np.copy(self._base.state.world.vel), 
            np.copy(self._base.state.world.ang_vel), 

            np.copy(self._base.state.local.vel), 
            np.copy(self._base.state.local.ang_vel), 
        )

        # return (
        #     np.copy(self._base.state.world.pos), 
        #     np.copy(proj_grav), 
        #     np.copy(self._base.state.world.vel), 
        #     np.copy(self._base.state.world.ang_vel), 
        #     np.copy(self._base.state.world.acc), 
        #     np.copy(self._base.state.world.ang_acc)
        # )