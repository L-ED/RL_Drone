


from typing import Any


class MPC_Control:

    def __init__(self, base) -> None:
        self._base = base

    def __call__(self, observation) -> Any:
        
