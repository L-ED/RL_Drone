from typing import Any, SupportsFloat, Tuple, Dict
import numpy as np 
import pybullet as pb
import gymnasium as gym


class Base(gym.Env):

    def __init__(self, drone, control_system, logger=None, scene_objects=None, visualize=True, record=False, realtime=False):
        G=9.8

        self.drone = drone
        self.min_frequency = find_lcm(
            drone.frequency, control_system.frequency)

        if visualize:
            self.client = pb.connect(pb.GUI)
        else:
            self.client = pb.connect(pb.DIRECT)

        self.add_objects(scene_objects, drone)

        pb.setGravity(0, 0, -G, physicsClientId=self.client)
        pb.setRealTimeSimulation(realtime, self.client) # diesnt work in DIRECT
        if not realtime:
            pb.setTimeStep(1/self.min_frequency)
        self.step=0


    def step(self, action: Any) -> Tuple[
            Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        
        self.step+=1
        pb.stepSimulation(self.client)
        obs = self.drone.step(action)
        reward = self.reward()
        terminated = self.check_termination()
        truncated = self.check_truncation()
        info = self.comute_info()
        return obs, reward, terminated, truncated, info


    def reset(self):
        self.step=0
        pb.resetSimulation(physicsClientId=self.CLIENT)
        state = self.create_initial_state()
        self.drone.reset_state(state)
        action = self.create_initial_action()
        obs = self.drone.step(action)
        return obs
    

    def render(self):
        
        state=self.drone.state
        wrld, loc = state.world, state.local

        msg = f"Step {self.step}\n"+ \
        "WORLD FRAME\n"+\
        f"pos {wrld.pos} rpy {wrld.rpy}"+ \
        f"vel {wrld.vel} ang_vel {wrld.ang_vel}\n"+ \
        f"acc {wrld.acc} ang_acc {wrld.ang_acc}\n"+\
        f"force {wrld.force} torque {wrld.torque}\n"+\
        "LOCAL FRAME\n"
        f"pos {loc.pos} rpy {loc.rpy}"+ \
        f"vel {loc.vel} ang_vel {loc.ang_vel}\n"+ \
        f"acc {loc.acc} ang_acc {loc.ang_acc}\n"+\
        f"force {loc.force} torque {loc.torque}\n"

        print(msg)