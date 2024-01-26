from collections import namedtuple
import numpy as np
import pybullet as pb

Inner_state = namedtuple(
    "Inner_state", 
    ['force', 'torque', 'vel', 'acc', 'ang_vel', 'ang_acc'],
    defaults=[np.array([0, 0, 0])]*6)
Outer_State = namedtuple(
    "State", 
    ('pos', 'ang')+ Inner_state._fields, 
    defaults=[np.array([0, 0, 0])]*8)

class State:

    def __init__(self) -> None:
        self.world = Outer_State()
        self.local = Inner_state()
        self.R = None
        self.T = None

    def create_R_matrix(self):

        self.R = np.array(pb.getMatrixFromQuaternion(
            pb.getQuaternionFromEuler(self.world.qtr)
        )).reshape((3,3))
        self.inv_R = self.R.T
    