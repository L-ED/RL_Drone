from collections import namedtuple
import numpy as np
import pybullet as pb
from dataclasses import dataclass, field

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
    force: np.array = field(default_factory= lambda: np.array([0, 0, 0]))
    torque: np.array = field(default_factory= lambda: np.array([0, 0, 0]))
    vel: np.array = field(default_factory= lambda:np.array([0, 0, 0]))
    acc: np.array = field(default_factory= lambda:np.array([0, 0, 0]))
    ang_vel: np.array = field(default_factory= lambda:np.array([0, 0, 0]))
    ang_acc: np.array = field(default_factory= lambda:np.array([0, 0, 0]))


@dataclass
class Outer_State(Inner_state):
    pos: np.array = field(default_factory= lambda:np.array([0, 0, 0]))
    rpy: np.array = field(default_factory= lambda: np.array([0, 0, 0]))


class State:

    def __init__(self) -> None:
        self.world = Outer_State()
        self.local = Inner_state()
        self.R = None
        self.T = None

    def create_R_matrix(self):

        self.R = np.array(pb.getMatrixFromQuaternion(
            self.world.qtr
        )).reshape((3,3))
        self.inv_R = self.R.T
    