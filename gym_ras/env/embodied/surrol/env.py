import gym
from gym_ras.env.embodied.base_env import BaseEnv
import pybullet as p
from surrol.utils.pybullet_utils import get_link_pose
import numpy as np
from gym_ras.tool.common import *
from pathlib import Path
import colorsys

class SurrolEnv(BaseEnv):
    def __init__(self,
        task,
        pybullet_gui=False,
        cid=-1,
        cam_width = 600,
        cam_height = 600,
        mask_background_id=-3,
        depth_remap_range= 0.2,
        depth_remap_range_noise= 0.0,
        depth_remap_center_noise= 0.0,
        cam_target_noise=0,
        cam_distance_noise=0,
        cam_yaw_noise=0,
        cam_pitch_noise=0,
        cam_roll_noise=0,
        cam_up_axis_noise=0,
        background_texture_dir="",
        random_obj_vis=True,
        cam_mode="rgbdm",
        cam_num=1,
        cam1_to_cam2_pose = [0.0,.0,.0, .0,.0,.0],
        realistic_depth=False,
        **kwargs,
        ):
        self._cam_width = cam_width
        self._cam_height = cam_height
        
        self._mask_obj = ["psm1", "stuff"]
        if task == "needle_pick":
            from gym_ras.env.embodied.surrol.needle_pick import NeedlePickMod
            # print(kwargs)
            client = NeedlePickMod(render_mode="human" if pybullet_gui else "rgb_array", cid=cid,**kwargs)
        elif task == "gauze_retrieve":
            from gym_ras.env.embodied.surrol.gauze_retrieve import GauzeRetrieveMod
            client = GauzeRetrieveMod(render_mode="human" if pybullet_gui else "rgb_array", cid=cid,**kwargs)
        elif task == "peg_transfer":
            from gym_ras.env.embodied.surrol.peg_transfer import PegTransferMod
            client = PegTransferMod(render_mode="human" if pybullet_gui else "rgb_array", cid=cid,**kwargs)                 
        else:
            raise Exception("Not support")
        

        super().__init__(client)
        self._random_obj_vis = random_obj_vis
        self._view = self.client._view_param
        self._project = self.client._proj_param
        self._mask_background_id = mask_background_id
        self._view["depth_remap_range"] = depth_remap_range
        self._view["depth_remap_range_noise"] = depth_remap_range_noise
        self._view["depth_remap_center_noise"] = depth_remap_center_noise
        self._view["target_noise"] = cam_target_noise
        self._view["distance_noise"] = cam_distance_noise
        self._view["yaw_noise"] = cam_yaw_noise
        self._view["pitch_noise"] = cam_pitch_noise
        self._view["yaw_noise"] = cam_yaw_noise
        self._view["roll_noise"] = cam_roll_noise
        self._view["up_axis_noise"] = cam_up_axis_noise
        self._texture_dir = {
                    "tray": Path(__file__).resolve().parent.parent.parent.parent / "asset" \
                                if background_texture_dir is "" else background_texture_dir,
                             }
        self._texture_extension = ["png", "jpeg", "jpg"]
        assert cam_mode in ["rgbdm", "rgbm", "rgbm_2"]
        self._cam_mode = cam_mode
        self._cam_num = cam_num
        self.seed = 0
        self._view_matrix = [None,] * cam_num
        self._cam1_to_cams_pose = [[0.0,0,0, 0,0,0], cam1_to_cam2_pose]
        self._proj_matrix = [None,] * cam_num
        self._depth_remap_range = [None,] * cam_num
        self._realistic_depth = realistic_depth

    @property
    def reward_dict(self, ):
        return self.client._reward_dict
    
    def _reset_cam(self, cam_id):
        # add_noise_fuc = lambda x, low, high: np.array(x) + self._cam_pose_rng.uniform(low, high)
        workspace_limits = self.client.workspace_limits1
        target_pos = [workspace_limits[0].mean(),  workspace_limits[1].mean(),  workspace_limits[2][0]]
        roll = self._view["roll"]
        pitch = self._view["pitch"]
        yaw = self._view["yaw"]
        distance = self._view["distance"]
        _dis_noise = self._cam_pose_rng.uniform(-self._view["distance_noise"], self._view["distance_noise"])
        _roll_noise = self._cam_pose_rng.uniform(-self._view["roll_noise"], self._view["roll_noise"])
        _pitch_noise = self._cam_pose_rng.uniform(-self._view["pitch_noise"], self._view["pitch_noise"])
        _yaw_noise = self._cam_pose_rng.uniform(-self._view["yaw_noise"], self._view["yaw_noise"]) 
        _cam_target_noise_x = self._cam_pose_rng.uniform(-self._view["target_noise"], self._view["target_noise"])
        _cam_target_noise_y = self._cam_pose_rng.uniform(-self._view["target_noise"], self._view["target_noise"])
        _up_axis_noise = self._cam_pose_rng.uniform(-self._view["up_axis_noise"], self._view["up_axis_noise"])
        _depth_remap_range_noise = self._cam_pose_rng.uniform(0, self._view["depth_remap_range_noise"])
        _depth_remap_center_noise = self._cam_pose_rng.uniform(-self._view["depth_remap_center_noise"], self._view["depth_remap_center_noise"])
        T1 = getT(target_pos, [roll, pitch+_pitch_noise, yaw+_yaw_noise],  rot_type="euler")
        T2 = getT([0,0,distance+_dis_noise], [0,0,0],  rot_type="euler")
        T3 = getT([0,0,0], [0,0,_roll_noise],  rot_type="euler")
        T = TxT([T1,T2])
        # _T_M=T[0:3,0:3]
        # _T_p=T[0:3,3]
        target_pos[0]+=_cam_target_noise_x
        target_pos[1]+=_cam_target_noise_y
        T_cam = TxT([T, getT(self._cam1_to_cams_pose[cam_id][:3], 
                                self._cam1_to_cams_pose[cam_id][3:], rot_type="euler", euler_convension="xyz", euler_Degrees=True)])
        self._view_matrix[cam_id] = p.computeViewMatrix(cameraEyePosition=T_cam[0:3,3].tolist(),
                    cameraTargetPosition=target_pos,
                    cameraUpVector=[0,np.sin(np.radians(_up_axis_noise)),np.cos(np.radians(_up_axis_noise))],
        )


        self._proj_matrix[cam_id] = p.computeProjectionMatrixFOV(fov=self._project["fov"],
                                            aspect=float(self._cam_width) / self._cam_height,
                                            nearVal=self._project["nearVal"],
                                            farVal=self._project["farVal"])
        _dis = distance+_dis_noise
        _center = _dis + _depth_remap_center_noise
        _range = self._view["depth_remap_range"]+_depth_remap_range_noise
        _low = _center - _range
        _high = _center + _range
        self._depth_remap_range[cam_id] = (min(_low,_dis-self._view["depth_remap_range"]), max(_high,_dis+self._view["depth_remap_range"]))

    def get_oracle_action(self):
        return self.client.get_oracle_action(self.client._get_obs())

    def _get_obj_pose(self, obj_id:str, link_index:int):
        # assert obj_id in self.obj_ids
        return get_link_pose(self.client.keyobj_ids[obj_id], link_index)
        
    def reset(self):
        self._init_vars()
        for j in range(self._cam_num):
            self._reset_cam(j)
        _ = self.client.reset()
        if self._random_obj_vis:
            self._random_background_obj_vis()
        obs = {}
        obs["robot_prio"], obs["gripper_state"] = self._get_prio_obs()
        return obs

    def step(self, action):
        self.timestep +=1
        if (not self.skip) or (self.step_func_prv is None):

            _prio, _ = self._get_prio_obs()
            _is_out = self._check_new_action(_prio, action[:3])
            obs, reward, done, info =self.client.step(action)
            obs["robot_prio"], obs["gripper_state"] = self._get_prio_obs()
            if _is_out:
                info["fsm"] = "progress_fail"
            self.step_func_prv = obs, reward, done, info
        
        return self.step_func_prv
    
    def render(self,): 
        imgs = {}
        for j in range(self._cam_num):
            postfix = "_"+str(j+1) if j!=0 else ""
            (_, _, px, depth, mask) = p.getCameraImage(width=self._cam_width,
                                        height=self._cam_height,
                                        viewMatrix=self._view_matrix[j],
                                        projectionMatrix=self._proj_matrix[j],
                                        shadow=1,
                                        lightDirection=(10, 0, 10),
                                        renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
            px = px[:,:,:3] # the 4th channel is alpha
            imgs["rgb"+postfix] = px

            if self._cam_mode.find("d")>=0:
                far = self._project["farVal"]
                near = self._project["nearVal"]
                depth = far * near / (far - (far - near) * depth) 
                depth = np.uint8(np.clip(self._scale(depth, self._depth_remap_range[j][0], self._depth_remap_range[j][1], 0, 255), 0, 255)) # to image
                imgs["depth"+postfix] = depth
                
                if self._realistic_depth:
                    _mask = None
                    for k,v  in self.client.no_depth_link_ids.items():
                        # print(k,v)
                        _mask_no = self._decode_mask(mask, self.client.keyobj_ids[k], v)
                        _mask = _mask_no if _mask is None else _mask or _mask_no
                    # print("jjj+++++++++++++++++++++++", np.sum(_mask))
                    _mask = np.invert(_mask)

                    _depth = np.zeros(depth.shape, dtype=np.uint8)
                    _depth[_mask] = depth[_mask]
                    imgs["depth"+postfix] = _depth



            if self._cam_mode.find("m")>=0:
                masks = {}
                for _mask_obj  in self._mask_obj:
                    _obj_id = self.client.keyobj_ids[_mask_obj]
                    _obj_link_id = self.client.keyobj_link_ids[_mask_obj]
                    masks[_mask_obj] = self._decode_mask(mask, _obj_id, _obj_link_id)

                
                imgs["mask"+postfix] = masks
        # print(imgs.keys())
        return imgs
    def _decode_mask(self, mask_in, mask_obj, mask_links):
        obj_mask = np.bitwise_and(mask_in,((1 << 24) - 1)) # ref: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/segmask_linkindex.py
        link_mask = (np.right_shift(mask_in,24)) - 1
        obj_mask[mask_in<=0] = self._mask_background_id  
        link_mask[mask_in<=0] = self._mask_background_id

        _obj_id = mask_obj
        _obj_link_id = mask_links
        decode_mask =  (obj_mask == _obj_id) &  (self._mask_or(link_mask, _obj_link_id))
        return decode_mask

        
    def _get_prio_obs(self):
        # norm by workspace 
        obs = self.client._get_robot_state(idx=0) # for psm1
        tip_pos =obs[:3]
        gripper_state = obs[6]
        return tip_pos, gripper_state

    @property
    def workspace_limit(self,):
        return self.client.workspace_limits1

    def _check_new_action(self, state, action):
        pos = self.client.action2pos(action)
        new_state = state + pos
        ws = self.workspace_limit
        _low = ws[:,0]
        _high = ws[:,1]
        is_out_ws =  np.any(new_state < _low) or (np.any(new_state > _high))
        return is_out_ws

    @staticmethod
    def _scale(input, old_min, old_max, new_min, new_max):
        out = (input-old_min)/(old_max-old_min)*(new_max-new_min) + new_min
        return out

    @property
    def observation_space(self):
        obs = {}
        obs['gripper_state'] = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        prio, _ = self._get_prio_obs()
        ws = self.workspace_limit
        _low = ws[:,0]
        _high = ws[:,1]
        obs['robot_prio'] = gym.spaces.Box(_low, _high, dtype=np.float32)
        return gym.spaces.Dict(obs)


    @property
    def action_space(self):
        return self.client.action_space

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.client.seed(seed)
        self._init_rng(seed)


    def _change_obj_vis(self,obj_id, obj_link_id, rgb_list=[1,1,1,1], texture_dir=None):
        # newmetal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal2.jpg"))
        if texture_dir is not None:
            textid = p.loadTexture(texture_dir)
            p.changeVisualShape(obj_id, obj_link_id, rgbaColor=rgb_list, textureUniqueId=textid)
        else:
            p.changeVisualShape(obj_id, obj_link_id, rgbaColor=rgb_list,)
    
    def _random_background_obj_vis(self,):
        _obj_id = {}
        _obj_id.update(self.client.background_obj_ids)
        _obj_id.update(self.client.keyobj_ids)
        _link_id = {}
        _link_id.update(self.client.background_obj_link_ids)
        _link_id.update(self.client.keyobj_link_ids)
        for k in self.client.random_vis_key:
            for link in _link_id[k]:
                color_type = self.client.random_color_range[0]
                if k in self.client.random_color_range[1]:
                    _range = self.client.random_color_range[1][k]
                else:
                    _range = self.client.random_color_range[1]["default"]
                _val = self._background_vis_rng.uniform(_range[0], _range[1])
                _rgb = _val.tolist()
                if color_type == "hsv":
                    _new_val = colorsys.hsv_to_rgb(_val[0],_val[1],_val[2])
                    _rgb = list(_new_val)
                    _rgb.append(_val[-1])
                if not k in self._texture_dir:
                    texture_dir = None
                else:
                    files = []
                    for ext in self._texture_extension:
                        files.extend(sorted(Path(self._texture_dir[k]).glob("*."+ext)))
                    texture_dir = str(files[self._background_vis_rng.randint(len(files))])
                self._change_obj_vis(_obj_id[k], link, _rgb,texture_dir=texture_dir)
            
   
    
    def _init_rng(self, seed):
        cam_pose_seed = np.uint32(seed+1)
        depth_remap_seed = np.uint32(seed+2)
        background_vis_seed = np.uint32(seed+3)
        print("cam_pose_seed:", cam_pose_seed)
        print("depth_remap_seed:", depth_remap_seed)
        self._cam_pose_rng = np.random.RandomState(cam_pose_seed)
        self._depth_remap_rng = np.random.RandomState(depth_remap_seed)
        self._background_vis_rng = np.random.RandomState(background_vis_seed)

    @staticmethod
    def _mask_or(mask, ids):
        x = None
        for _id in ids:
            out = mask == _id
            x = out if x is None else x | out
        return x




if __name__ == "__main__":
    env = SurrolEnv(name="needle_pick")
