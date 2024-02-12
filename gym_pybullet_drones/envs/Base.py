from typing import Any, SupportsFloat, Tuple, Dict
import numpy as np 
import pybullet as pb
import gymnasium as gym
import pybullet_data

from gym_pybullet_drones.utils.state import State
from math import gcd

class Base(gym.Env):

    def __init__(self, client, drone, control_system=None, logger=None, scene_objects=[], visualize=True, record=False, realtime=False):

        self.G=9.8
        self.rho = 1.
        self.drone = drone
        self.control_system = control_system
        self.client = client

        self.find_env_frequency(realtime)

        self.add_objects(scene_objects, drone)

        pb.setGravity(0, 0, -self.G, physicsClientId=self.client)
        pb.setRealTimeSimulation(realtime, self.client) # diesnt work in DIRECT

        self.step_idx=0
        self.timestemp = 0

        self.action_space = self.drone.actionSpace
        self.observation_space = self.drone.obsSpace

        self.X_AX = -1
        self.Y_AX = -1
        self.Z_AX = -1

        if visualize:
            self.showDroneLocalAxes()
            self.drone.visualize_sensors()


    def step(self, action: Any) -> Tuple[
            Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        
        pb.stepSimulation(self.client)
        obs = self.drone.step(action, self)
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
        return obs, None
    

    def render(self):
        
        state=self.drone.state
        wrld, loc = state.world, state.local

        msg = f"Step {self.step_idx} {np.round(self.timestep*self.step_idx, 4)}\n"+ \
        "WORLD FRAME\n"+\
        f"pos {np.round(wrld.pos, 4)} rpy {np.round(wrld.rpy, 4)} vel {np.round(wrld.vel, 4)} ang_vel {np.round(wrld.ang_vel, 4)}\n"+ \
        f"acc {np.round(wrld.acc, 4)} ang_acc {np.round(wrld.ang_acc, 4)} force {np.round(wrld.force, 4)} torque {np.round(wrld.torque, 4)}\n"+\
        "LOCAL FRAME\n"+\
        f"vel {np.round(loc.vel, 4)} ang_vel {np.round(loc.ang_vel, 4)}\n"+ \
        f"acc {np.round(loc.acc, 4)} ang_acc {np.round(loc.ang_acc, 4)} force {np.round(loc.force, 4)} torque {np.round(loc.torque, 4)}\n"

        print(msg)


    def close(self):
        pb.disconnect(self.client)


    def add_objects(self, scene_obj, vehicle):

        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        self.plane_id = pb.loadURDF("plane.urdf", physicsClientId=self.client)

         # """
        # p.loadURDF("samurai.urdf",
        #            physicsClientId=self.CLIENT
        #            )
        pb.loadURDF("duck_vhacd.urdf",
                   [-.5, -.5, .05],
                   pb.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.client
                   )
        pb.loadURDF("cube_no_rotation.urdf",
                   [10.5, -2.5, .5],
                   pb.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.client
                   )
        pb.loadURDF("sphere2.urdf",
                   [0, 2, .5],
                   pb.getQuaternionFromEuler([0,0,0]),
                   physicsClientId=self.client
                   )

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
    

    def showDroneLocalAxes(self):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        AXIS_LENGTH = 2*self.drone.L
        print("AXS", AXIS_LENGTH)
        self.X_AX = pb.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[AXIS_LENGTH, 0, 0],
            lineColorRGB=[1, 0, 0],
            parentObjectUniqueId=self.drone.ID,
            parentLinkIndex=-1,
            replaceItemUniqueId=int(self.X_AX),
            physicsClientId=self.client
        )

        self.Y_AX = pb.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, AXIS_LENGTH, 0],
            lineColorRGB=[0, 1, 0],
            parentObjectUniqueId=self.drone.ID,
            parentLinkIndex=-1,
            replaceItemUniqueId=int(self.Y_AX),
            physicsClientId=self.client
        )

        self.Z_AX = pb.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0, AXIS_LENGTH],
            lineColorRGB=[0, 0, 1],
            parentObjectUniqueId=self.drone.ID,
            parentLinkIndex=-1,
            replaceItemUniqueId=int(self.Z_AX),
            physicsClientId=self.client
        )

    
    def find_env_frequency(self, realtime):

        devs_freqs = self.drone.take_dev_freqs() 
        if self.control_system is not None:
            devs_freqs.append(self.control_system.freq)

        self.min_frequency = 240
        if len(devs_freqs)>0:
            self.min_frequency = self.find_lcm(devs_freqs)

        if not realtime:
            pb.setTimeStep(1/self.min_frequency)
            self.timestep = 1/self.min_frequency

        print("Frequency ", self.min_frequency)
        self.drone.set_sensor_ticks(self.min_frequency)
        