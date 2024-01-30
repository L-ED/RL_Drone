from typing import Any, SupportsFloat, Tuple, Dict
import numpy as np 
import pybullet as pb
import gymnasium as gym
import pybullet_data

from gym_pybullet_drones.utils.state import State
from math import gcd

class Base(gym.Env):

    def __init__(self, client, drone, control_system, logger=None, scene_objects=[], visualize=True, record=False, realtime=False):
        G=9.8
        self.rho = 1.


        self.drone = drone
        self.client = client
        devs_freqs = drone.take_dev_freqs() 

        self.min_frequency = 240
        if len(devs_freqs)>0:
            self.min_frequency = self.find_lcm(
                drone.take_dev_freqs()#.append(control_system.frequency)
                )

        self.add_objects(scene_objects, drone)

        pb.setGravity(0, 0, -G, physicsClientId=self.client)
        pb.setRealTimeSimulation(realtime, self.client) # diesnt work in DIRECT
        if not realtime:
            pb.setTimeStep(1/self.min_frequency)
            self.timestep = 1/self.min_frequency
        self.step_idx=0
        self.timestemp = 0

        self.action_space = self.drone.actionSpace
        self.observation_space = self.drone.obsSpace


    def step(self, action: Any) -> Tuple[
            Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        
        pb.stepSimulation(self.client)
        # obs = self.drone.step(action, self)
        obs=None
        reward = self.reward()
        terminated = self.check_termination()
        truncated = self.check_truncation()
        info = self.compute_info()
        self.step_idx+=1
        self.timestemp+=self.timestep
        return obs, reward, terminated, truncated, info


    def reset(self):
        self.step_idx=0
        self.timestemp=0

        pb.resetSimulation(physicsClientId=self.CLIENT)
        state = self.create_initial_state()
        self.drone.reset_state(state)
        action = self.create_initial_action()
        obs = self.drone.step(action)
        return obs
    

    def render(self):
        
        state=self.drone.state
        wrld, loc = state.world, state.local

        msg = f"Step {self.step_idx} {self.timestemp}\n"+ \
        "WORLD FRAME\n"+\
        f"pos {wrld.pos} rpy {wrld.rpy}"+ \
        f"vel {wrld.vel} ang_vel {wrld.ang_vel}\n"+ \
        f"acc {wrld.acc} ang_acc {wrld.ang_acc}\n"+\
        f"force {wrld.force} torque {wrld.torque}\n"+\
        "LOCAL FRAME\n"+\
        f"vel {loc.vel} ang_vel {loc.ang_vel}\n"+ \
        f"acc {loc.acc} ang_acc {loc.ang_acc}\n"+\
        f"force {loc.force} torque {loc.torque}\n"

        print(msg)


    def close(self):
        pb.disconnect(self.client)


    def add_objects(self, scene_obj, vehicle):

        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        self.plane_id = pb.loadURDF("plane.urdf", physicsClientId=self.client)

        for obj in scene_obj:
            state = obj['state']
            pb.loadURDF(
                obj['file'],
                state.world.pos, 
                state.world.q,
                physicsClientId=self.client
            )

        vehicle.load_model()
    

    def reward(self):
        return 0
    

    def check_termination(self):
        return False
    
    
    def check_truncation(self):
        return False
    

    def compute_info(self):
        return None
    

    def create_initial_state(self):
        return State()
    

    def create_initial_action(self):
        return 
    
    def find_lcm(self, frequencies):

        lcm = frequencies.pop(-1)
        while len(frequencies)>0:
            freq = frequencies.pop(-1)
            lcm = lcm*freq//gcd(lcm, freq)

        return lcm