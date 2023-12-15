from gym_ras.env.wrapper.base import BaseWrapper
import gym
import numpy as np


class FSM(BaseWrapper):
    """ finite state machine 
    """

    def __init__(self, env,**kwargs
                 ):
        super().__init__(env)
        self._states = ["in_progress","done_success","done_fail","progress_fail"]
    
    def reset(self):
        obs = self.env.reset()
        fsm_state = "in_progress"
        obs["fsm_state"] = self._states.index(fsm_state)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        fsm_state = info["fsm"]
        # print(self.unwrapped.reward_dict)
        reward = self.unwrapped.reward_dict[fsm_state]
        info['is_success'] = fsm_state =="done_success"
        if fsm_state in ["done_success","done_fail"]:
            done = True
        else:
            done = False
        obs["fsm_state"] = self._states.index(fsm_state)
        return obs, reward, done, info

    @property
    def observation_space(self):
        obs = {k:v for k, v in self.env.observation_space.items()}
        obs['fsm_state'] = gym.spaces.Box(low=0, 
                                      high=len(self._states)-1, shape=(1,),dtype=np.float32)
        return gym.spaces.Dict(obs)