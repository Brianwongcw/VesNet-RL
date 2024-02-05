import numpy as np
import torch
import cv2
import math
from Vessel_3d import Vessel_3d_sim
from collections import deque
from gym_ras.env.wrapper.base import BaseWrapper
import imutils
import gym

class VesselEnv():
    def __init__(self,configs,num_channels=3):
        self.configs=configs
        self.num_envs=len(configs)
        self.vessels=[]
        for config in configs:
            self.vessels.append(Vessel_3d_sim(config,probe_width=313))
        
        self.reward_window=deque(maxlen=num_channels+1)
        self.area_window=deque(maxlen=num_channels+1)
        
        self.z_size=[0,2*math.pi]
        self.actions=[(0,0,0), (50, 0, 0), (-50, 0, 0), (0, 50, 0), (0, -50, 0), (0, 0, math.pi/18), (0, 0, -math.pi/18)]

        self.num_actions=len(self.actions)
        self.num_channels=num_channels
        self.actions_all=[]
        self.action_his=deque(maxlen=self.num_channels)
        for _ in range(self.num_channels):
            self.action_his.append(-1)
        self.pose_his=deque(maxlen=self.num_channels)
        for _ in range(self.num_channels):
            self.pose_his.append(0)
    
    def terminate_decision(self):
        if len(self.contours)<1:
            return False
        areas=[cv2.contourArea(c) for c in self.contours]
        max_area_index=np.argmax(areas)
        
        c=self.contours[max_area_index]
        area=areas[max_area_index]
        rect = cv2.minAreaRect(c)
        box = cv2.cv.Boxpoints() if imutils.is_cv2()else cv2.boxPoints(rect)
        box = np.int0(box)
        box = np.clip(box, 0, 255)
        width=np.linalg.norm(box[0]-box[1])
        height=np.linalg.norm(box[1]-box[2])
        box_area=int(width*height)
        self.estimated_diameter.append(min(width,height))
        if areas[max_area_index]<5000:
            return False
        
        # terminate_coef=(box_area-area)/box_area
        # if terminate_coef<0.15 and max(width,height) >250 and min(width,height)>(np.mean(self.estimated_diameter)-10):
        #     return True
        # else:
        #     return False
        
        terminate_coef=(box_area-area)/box_area
        if max(width,height) >250 and min(width,height)>(np.mean(self.estimated_diameter)-10) and terminate_coef<0.01:
            return True
        else:
            return False
        

    def reward_func(self):
        self.vessel_area=len(np.where(self.image>0.9)[0])
        max_area=self.vessels[self.cur_env].r*256/self.vessels[self.cur_env].size_3d[2]*256*2
        reward_vessel=(self.vessel_area-self.vessels[self.cur_env].threshold)/(max_area-self.vessels[self.cur_env].threshold)
        reward_dis=1-abs(self.pos[1]-self.vessels[self.cur_env].c[0])/(self.vessels[self.cur_env].probe_width/2+self.vessels[self.cur_env].r)
        return 0.7*reward_vessel+0.3*reward_dis

    def step(self,action_int):
        action=self.actions[action_int]
        
        new_pos=np.array([int(self.pos[0]+action[0]*np.cos(self.pos[2])-action[1]*np.sin(self.pos[2])),int(self.pos[1]+action[0]*np.sin(self.pos[2])+action[1]*np.cos(self.pos[2])),self.pos[2]+action[2]])
        
        if self.vessels[self.cur_env].check_mask(new_pos[0:2],new_pos[2]) and self.vessels[self.cur_env].vessel_existance(new_pos[0:2],new_pos[2]):
            self.pos=new_pos
            self.action_his.append(action_int)
            self.actions_all.append(action_int)
            self.pose_his.append(self.pos[2])
            reward_extra=-0.01
        else:
            self.pos=self.pos
            self.action_his.append(action_int)
            self.actions_all.append(-1)
            self.pose_his.append(self.pos[2])
            reward_extra=-0.1
            
        self.image,_,_=self.vessels[self.cur_env].get_slicer(self.pos[0:2],self.pos[2])

        # self.image = cv2.resize(self.image, (64,64),interpolation=cv2.INTER_NEAREST)

        # import pdb;pdb.set_trace()
        # self.image = np.stack((self.image,)*3, axis=-1)
        
        self.uint_img = np.array(self.image).astype('uint8')
        
        self.contours,_ = cv2.findContours(self.uint_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        cur_reward=self.reward_func()


        state=self.image

        self.state.append(state)
        img = np.array(state * 255).astype('uint8')
        img = np.stack((img, img,img), axis=-1)
        self.reward_window.append(cur_reward)
        self.vessel_area=len(np.where(self.image>0.9)[0])
        self.area_window.append(self.vessel_area)
        done=False
        
        reward=self.reward_window[-1]-self.reward_window[-2]
        # reward=-self.reward_window[-1]+self.reward_window[-2]

        if cur_reward>0.9 and self.actions_all[-1]!=-1:
            if np.mean(self.reward_window)>0.95 and len(self.actions_all)>4:
                done=True
                reward=5
            else:
                reward=1

        # done=self.terminate_decision()
        
        self.area_changes=np.array(self.area_window)[1:]-np.array(self.area_window)[:-1]
        
        return {'image':(np.array(img))}, reward+reward_extra, done, {'action_his':np.array(self.action_his),'area_change':self.area_changes/1000}
    


    def reset(self,randomVessel = False,randomStart=True):
        self.first_increase=True
        self.actions_all=[]
        self.estimated_diameter=deque(maxlen=10)
        for _ in range(self.reward_window.maxlen):
            self.reward_window.append(0)
        for _ in range(self.area_window.maxlen):
            self.area_window.append(0)
        for _ in range(self.action_his.maxlen):
            self.action_his.append(-1)
        for _ in range(self.pose_his.maxlen):
            self.pose_his.append(0)
        if randomVessel:
            self.cur_env=np.random.randint(self.num_envs)
        else:
            self.cur_env=0
        if randomStart:
            self.pos=self.vessels[self.cur_env].get_vertical_view(self.vessels[self.cur_env].size_3d[0]//2)
            while not (self.vessels[self.cur_env].check_mask(self.pos[0:2],self.pos[2])  and self.vessels[self.cur_env].vessel_existance(self.pos[0:2],self.pos[2])):
                    self.pos=self.vessels[self.cur_env].get_vertical_view_p(np.random.randint(self.vessels[self.cur_env].x_min+20,self.vessels[self.cur_env].x_max-20))        
        
        # import pdb;pdb.set_trace()
        self.state=deque(maxlen=self.num_channels)
        for _ in range(self.state.maxlen):
            self.state.append(np.zeros([256,256]))
   
        self.image,self.poi,_=self.vessels[self.cur_env].get_slicer(self.pos[0:2],self.pos[2])
        # self.image = cv2.resize(self.image, (64,64),interpolation=cv2.INTER_NEAREST)
        state=self.image
        
        self.state.append(state)
        img = np.array(state * 255).astype('uint8')
        img = np.stack((img, img,img), axis=-1)
        self.vessel_area=len(np.where(self.image>0.9)[0])
        self.area_window.append(self.vessel_area)
        self.area_changes=np.array(self.area_window)[1:]-np.array(self.area_window)[:-1]
        # img = np.stack((img, img, img), axis = -1)

        return {'image': img}

    @property
    def action_space(self):
        return self.actions
    
    @property
    def observation_space(self):
        obs = {}
        obs['image'] = gym.spaces.Box(0, 255, (256, 256, 3), np.uint8)
        return gym.spaces.Dict(obs)


def create_configs_rand(num):
    configs=[]
    r_min=30
    r_max=75
    for i in range(num):
        offset=np.random.rand()*np.pi/2
        size_3d=[750,700,450]
        r=np.random.randint(r_min+(r_max-r_min)*i/num,r_min+(r_max-r_min)*(i+1)/num)
        c_x=350
        c_y=np.random.randint(50+r,225)
        c=[c_x,c_y]
        config=(c,r,size_3d,offset)
        configs.append(config)
    return configs


class DiscreteAction(BaseWrapper):
    def __init__(self, env, 
                 action_scale=0.2,
                 **kwargs):
        super().__init__(env)
        self._action_dim =  4
        assert self._action_dim == 4 # onlys support x,y,z,yaw,gripper
        self._action_list = []
        self._action_strs = ['still', 'x_neg', 'x_pos', 'y_neg','y_pos', 'rot_neg','rot_pos']
        self._action_prim = {'still':0, 'x_neg':1, 'x_pos':2, 'y_neg':3,'y_pos':4, 'rot_neg':5,'rot_pos':6} # store discrete action primitives
        self._action_idx = [0,1,2,3,4,5,6]
        self._action_discrete_n = len(self._action_idx)
        
    @property
    def action_space(self):
        return gym.spaces.Discrete(self._action_discrete_n)
        
    def step(self, action):
        # import pdb;pdb.set_trace()
        _action = self._action_prim[self._action_strs[self._action_idx[action]]]
        # print(_action, self._is_gripper_close)
        obs, reward, done, info  =  self.env.step(_action)
            
        return obs, reward, done, info
    
    
    def get_oracle_action(self):
        action = self.env.get_oracle_action()
        # print(action)
        ref = np.zeros(action.shape)
        if self._is_gripper_close:
            ref[-1] = -1
        else:
            ref[-1] = 1

        _err = action - ref
        # print("err",_err, self._is_gripper_close)
        if np.abs(_err)[-1] >= 1: # gripper toggle 
             _action = 8
             return _action
        else:
            _err[-1] = 0
            _index = np.argmax(np.abs(_err)) 
            _direction = _err[_index] > 0
            if _direction:
                _action = _index * 2 + 1
            else:
                _action = _index * 2
        return _action