import numpy as np
import torch
import cv2
import math
import torchvision.transforms as transforms
from Vessel_3d import Vessel_3d_sim, Vessel_3d
from collections import deque
from Env import Env_multi_sim_img_test
import imutils
import gym
import pdb

class VesselEnv():
    def __init__(self,configs,num_channels=4):
        self.configs=configs
        self.num_envs=len(configs)
        self.vessels=[]
        for config in configs:
            self.vessels.append(Vessel_3d_sim(config,probe_width=313))
        
        self.reward_window=deque(maxlen=num_channels+1)
        self.area_window=deque(maxlen=num_channels+1)
        
        self.z_size=[0,2*math.pi]
        self.actions=((0,0,0), (50, 0, 0), (-50, 0, 0), (0, 50, 0), (0, -50, 0), (0, 0, math.pi/90), (0, 0, -math.pi/90))
        # self.observation_space={'spaces':[1]}

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
        if max(width,height) >250 and min(width,height)>(np.mean(self.estimated_diameter)-10) and terminate_coef<0.03:
            return True
        else:
            return False
    
    def step(self,action_int):
        action=self.actions[action_int]
        
        pdb.set_trace()
        new_pos=np.array([int(self.pos[0]+action[0]*np.cos(self.pos[2])-action[1]*np.sin(self.pos[2])),int(self.pos[1]+action[0]*np.sin(self.pos[2])+action[1]*np.cos(self.pos[2])),self.pos[2]+action[2]])
        
        if self.vessels[self.cur_env].check_mask(new_pos[0:2],new_pos[2]) and self.vessels[self.cur_env].vessel_existance(new_pos[0:2],new_pos[2]):
            self.pos=new_pos
            self.action_his.append(action_int)
            self.actions_all.append(action_int)
            self.pose_his.append(self.pos[2])
        else:
            self.pos=self.pos
            self.action_his.append(action_int)
            self.actions_all.append(-1)
            self.pose_his.append(self.pos[2])
            
        self.image,_,_=self.vessels[self.cur_env].get_slicer(self.pos[0:2],self.pos[2])
        
        self.uint_img = np.array(self.image).astype('uint8')
        
        self.contours,_ = cv2.findContours(self.uint_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        state=self.image

        self.state.append(state)
        self.vessel_area=len(np.where(self.image>0.9)[0])
        self.area_window.append(self.vessel_area)
        done=False
        

        done=self.terminate_decision()
            
        self.area_changes=np.array(self.area_window)[1:]-np.array(self.area_window)[:-1]
        
        return (np.array(self.state),np.array(self.action_his),self.area_changes/1000), done
    


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
        
        self.state=deque(maxlen=self.num_channels)
        for _ in range(self.state.maxlen):
            self.state.append(np.zeros([256,256]))
   
        self.image,self.poi,_=self.vessels[self.cur_env].get_slicer(self.pos[0:2],self.pos[2])
        state=self.image
        
        self.state.append(state)
        self.vessel_area=len(np.where(self.image>0.9)[0])
        self.area_window.append(self.vessel_area)
        self.area_changes=np.array(self.area_window)[1:]-np.array(self.area_window)[:-1]

        return (np.array(self.state),np.array(self.action_his),self.area_changes/1000)

    @property
    def action_space(self):
        return self.actions
    @property
    def observation_space(self):
        pdb.set_trace()
        obs = {}
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
