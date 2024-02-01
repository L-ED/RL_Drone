

from typing import Any
from dataclasses import dataclass

@dataclass
class Device:

    def __init__(self, frequency, base=None) -> None:
        self.freq = frequency
        self.last_obs = None
        self._base = base
        self._name = None
        self.timestemp=1/frequency
        self.counter = 0


    def __call__(self, timestemp) -> Any:
        if self._base is None:
            ValueError("Set base")
        
        # timestemp = round(timestemp, 10)

        # print(timestemp, self.timestemp, timestemp%self.timestemp)
        # if timestemp%self.timestemp == 0:
        if self.counter == 0:
            self.last_obs = self.make_obs()

        self.counter = (self.counter+1)%self.tick

        return self.last_obs
    
    def set_tick(self, env_freq):
        self.tick = env_freq//self.freq


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

    def visualize(self):
        pass