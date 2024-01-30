

from typing import Any
from dataclasses import dataclass

@dataclass
class Device:


    def __init__(self, frequency, base=None) -> None:
        self.freq = frequency
        self.last_obs = None
        self._base = base
        self._name = None
    
    def __call__(self, timestemp) -> Any:
        if self._base is None:
            ValueError("Set base")
        if timestemp%self.freq == 0:
            self.last_obs = self.make_obs()

        return self.last_obs
    

    def make_obs(self):
        pass

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, base):
        self._base = base

    @base.deleter
    def base(self):
        del self._base

    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @name.deleter
    def name(self):
        del self._name