from gym_pybullet_drones.devices import Device

class IMU(Device):
    def make_obs(self):
        return (self._base.state.local.acc, self._base.state.local.ang_vel)
        