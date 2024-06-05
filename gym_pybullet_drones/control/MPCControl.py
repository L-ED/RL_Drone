
import do_mpc
from casadi import vertcat

from typing import Any


class MPC_Control:

    def __init__(self, base) -> None:
        self._base = base

    def __call__(self, observation) -> Any:
        
    def create_model(self):

        model = do_mpc.model.Model()

        u1 = 