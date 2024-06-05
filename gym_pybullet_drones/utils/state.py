from collections import namedtuple
import numpy as np
import pybullet as pb
from dataclasses import dataclass, field
from functools import cache

# Inner_state = namedtuple(
#     "Inner_state", 
#     ['force', 'torque', 'vel', 'acc', 'ang_vel', 'ang_acc'],
#     defaults=[np.array([0, 0, 0])]*6)
# Outer_State = namedtuple(
#     "Outer_State", 
#     ('pos', 'rpy')+ Inner_state._fields, 
#     defaults=[np.array([0, 0, 0])]*8)


@dataclass
class Inner_state:
    force: np.array = field(default_factory= lambda: np.array([0, 0, 0], dtype=np.float64))
    thrust: np.array = field(default_factory= lambda: np.array([0, 0, 0, 0], dtype=np.float64))
    torque: np.array = field(default_factory= lambda: np.array([0, 0, 0], dtype=np.float64))
    vel: np.array = field(default_factory= lambda:np.array([0, 0, 0], dtype=np.float64))
    acc: np.array = field(default_factory= lambda:np.array([0, 0, 0], dtype=np.float64))
    ang_vel: np.array = field(default_factory= lambda:np.array([0, 0, 0], dtype=np.float64))
    ang_acc: np.array = field(default_factory= lambda:np.array([0, 0, 0], dtype=np.float64))


@dataclass
class Outer_State(Inner_state):
    pos: np.array = field(default_factory= lambda:np.array([0, 0, 0], dtype=np.float64))
    rpy: np.array = field(default_factory= lambda: np.array([0, 0, 0], dtype=np.float64))
    qtr: np.array = field(default_factory= lambda: np.array([0, 0, 0, 0], dtype=np.float64))


class State:

    def __init__(self) -> None:
        self.world = Outer_State()
        self.local = Inner_state()

    @property
    def R(self):
        return self.compute_R(self.world.qtr)  
    
    # @cache
    def compute_R(self, quaternion): 
        return np.array(
                pb.getMatrixFromQuaternion(
                    quaternion
            )
        ).reshape((3,3))
    