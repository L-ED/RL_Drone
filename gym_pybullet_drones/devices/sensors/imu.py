from gym_pybullet_drones.devices import Device
import numpy as np
import pybullet as pb
from gymnasium import spaces

class IMU(Device):
    name_base = 'IMU'

    def __init__(self, frequency, accel_params, gyro_params, base=None) -> None:
        super().__init__(frequency, base)

        self.G=9.8

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, 6)
        )

        accel_params['sample_time'] = 1/self.freq
        gyro_params['sample_time'] = 1/self.freq

        self.acc_noize = SensorNoizeModel(
            **accel_params
        )
        self.gyro_noize = SensorNoizeModel(
            **gyro_params
        )

    def make_obs(self):

        local_g = np.dot(
            self._base.state.R.T,
            np.array([0, 0, -self.G]))
        
        acc = self.acc_noize.step(self._base.state.local.acc + local_g)
        ang_vel = self.gyro_noize.step(self._base.state.local.ang_vel)
        
        return np.concatenate((acc, ang_vel))
    

# https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
'''
Sensor model assumed as:
    estimated_value = ideal_value + bias + noize 
    noize = noise_density*N(0, 1)/sqrt(sample_time)
    bias = bias_prev + random_walk*sqrt(sample_time)
    
'''

class SensorNoizeModel:
    def __init__(self, random_walk, tau, sample_time, bias_stability) -> None:
        
        sqrt_dt = np.sqrt(sample_time)
        self.bias_stability_norm = bias_stability / np.sqrt(tau)
        self.sigma_bias = self.bias_stability_norm * sqrt_dt
        self.sigma_rw = random_walk /sqrt_dt
        self.bias = np.zeros(3)
        self.turn_on_bias = np.zeros(3)  # assume calibration is done


    def step(self, value):
        noize = np.random.normal(0, 1, size=(3)) * self.sigma_rw + self.bias
        self.bias += np.random.normal(0, 1, size=(3)) * self.sigma_bias
        return value + noize 




def mpu6000(base=None): #spi(low) - 100kHZ, spi(high) - 1MHZ, SPI(readonly) - 20MHZ, I2C - 100/400 (kHZ) 
    G=9.8

    gyro = {
        'random_walk': 0.30/np.sqrt(3600.0)*np.pi/180,
        'tau':500,
        'bias_stability': 4.6/3600*np.pi/180
    }
    accel = {
        'random_walk': G / 1.0e3,
        'tau':800,
        'bias_stability': 36.0 * 1e-6 * G
    }

    return IMU(
        frequency= int(100*1e3),
        accel_params=accel,
        gyro_params=gyro,
        base=base
    )