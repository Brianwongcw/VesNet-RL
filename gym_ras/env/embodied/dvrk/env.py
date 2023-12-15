import gym
from gym_ras.env.embodied.base_env import BaseEnv
import numpy as np
from gym_ras.tool.common import *

class dVRKEnv(BaseEnv):
    def __init__(self,
        task,
        **kwargs,
        ):
        if task in "needle_pick":
            from gym_ras.env.embodied.dvrk.needle_pick import NeedlePick
            client = NeedlePick(**kwargs)
        else:
            raise Exception("Not support")
        super().__init__(client)

    def reset(self):
        self._init_var()
        return self.client.reset()

    def step(self,action):
        obs, reward, done, info = self.client.step(action)
        self.timestep+=1
        return obs, reward, done, info

    def render(self, **kwargs): #['human', 'rgb_array', 'mask_array']
        return self.client.render()
    
    def get_oracle_action(self,**kwargs):
        return self.client.get_oracle_action()

    def __getattr__(self, name):
        """__getattr__ is only invoked if the attribute wasn't found the usual ways."""
        if name[0] == "_":
            raise Exception("cannot find {}".format(name))
        else:
            return getattr(self.client, name)

    def _init_var(self):
        self.timestep = 0

    @property
    def reward_dict(self):
        return self.client.reward_dict