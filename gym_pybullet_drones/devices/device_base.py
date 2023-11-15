

from typing import Any
from dataclasses import dataclass

@dataclass
class Device:


    def __init__(self, frequency, base=None) -> None:
        self.freq = frequency
        self.last_obs = None
        self._base = base

    
    def __call__(self, timestemp) -> Any:
        if self._base is None:
            ValueError("Set base")
        if timestemp%self.freq == 0:
            self.last_obs = self.make_obs()

        return self.last_obs
    

    def make_obs(self):
        pass

    @property
    def base(self, base):
        self._base = base
