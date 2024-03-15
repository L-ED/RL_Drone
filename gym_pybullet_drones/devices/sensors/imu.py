from gym_pybullet_drones.devices import Device
import numpy as np
import pybullet as pb
from gymnasium import spaces

class IMU(Device):
    name_base = 'IMU'

    def __init__(self, frequency, accel_params, gyro_params, base=None, verbose=False) -> None:
        super().__init__(frequency, base)

        self.G=9.8
        self.verbose = verbose

        self.observation_space = spaces.Box(
            # low=-np.inf, high=np.inf, shape=(1, 6)
            low=-np.inf, high=np.inf, shape=(6,)
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
        
        acc = self.acc_noize.step(
            self._base.state.local.acc #+ local_g
        )
        ang_vel = self.gyro_noize.step(self._base.state.local.ang_vel)
        
        if self.verbose:
            print((acc, ang_vel))

        return np.concatenate((acc, ang_vel))
    

# https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
'''
Sensor model assumed as:
    estimated_value = ideal_value + bias + noize 
    noize = noise_density*N(0, 1)/sqrt(sample_time)
    bias = bias_prev + random_walk*sqrt(sample_time)
    
'''

class SensorNoizeModel:
    def __init__(self, random_walk, sample_time, noize_density) -> None:
        
        sqrt_dt = np.sqrt(sample_time)
        self.sigma_bias = random_walk * sqrt_dt
        self.sigma_rw = noize_density /sqrt_dt
        self.bias = np.zeros(3)
        self.turn_on_bias = np.zeros(3)  # assume calibration is done


    def step(self, value):
        self.bias += np.random.normal(0, 1, size=(3)) * self.sigma_bias
        noize = np.random.normal(0, 1, size=(3)) * self.sigma_rw + self.bias
        return value + noize 




def mpu6000(base=None): #spi(low) - 100kHZ, spi(high) - 1MHZ, SPI(readonly) - 20MHZ, I2C - 100/400 (kHZ) 
    G=9.8

    # gyro = {
    #     'random_walk': 0.30/np.sqrt(3600.0)*np.pi/180,
    #     'tau':500,
    #     'bias_stability': 4.6/3600*np.pi/180
    # }
    # accel = {
    #     'random_walk': G / 1.0e3,
    #     'tau':800,
    #     'bias_stability': 36.0 * 1e-6 * G
    # }
    # https://www.researchgate.net/publication/335679582_Performance_Assessment_of_an_Ultra_Low-Cost_Inertial_Measurement_Unit_for_Ground_Vehicle_Navigation
    gyro = {
        'random_walk': 8.7*1e-5,
        'noize_density': 2.3*1e-5
    }
    accel = {
        'random_walk': 3.9*1e-3,
        'noize_density': 5*1e-4
    }

    return IMU(
        frequency= int(1e3),
        accel_params=accel,
        gyro_params=gyro,
        base=base
    )