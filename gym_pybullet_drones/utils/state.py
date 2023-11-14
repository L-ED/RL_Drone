from collections import namedtuple
import numpy as np

Inner_state = namedtuple(
    "Inner_state", 
    ['forces', 'torques', 'vel', 'acc', 'ang_vel', 'ang_acc'],
    defaults=[np.array([0, 0, 0])]*6)
Outer_State = namedtuple(
    "State", 
    ('pos', 'ang')+ Inner_state._fields, 
    defaults=[np.array([0, 0, 0])]*8)

class State:

    def __init__(self) -> None:
        self.world = Outer_State()
        self.local = Inner_state()

    