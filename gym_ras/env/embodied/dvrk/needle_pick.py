import gym
from gym_ras.env.embodied.dvrk.rgbd_cam import RGBD_CAM
from gym_ras.env.embodied.dvrk.oracle_input import OracleInput
from gym_ras.tool.config import load_yaml
import numpy as np
import yaml
class NeedlePick(gym.Env):
    def __init__(self,
                arm_names=["PSM1",],
                rgbd_device="rs435",
                oracle_device="ds4",
                ws_x=[-0.1,0.1],
                ws_y=[-0.1,0.1],
                ws_z=[-0.24,0],
                psm_reset_q=[0,0,0.12, 0,0,0],
                psm_open_gripper_deg=40,
                psm_init_gripper_quat=[7.07106781e-01,  7.07106781e-01, 0, 0],
                psm_init_pose_low_gripper = [-0.5,-0.5,-0.5,-0.9],
                psm_init_pose_high_gripper = [0.5,0.5,0.5,0.9],
                psm_max_step_pos=0.01,
                psm_max_step_rot=20,
                cam_image_height = 600,
                cam_image_width = 600,
                cam_depth_remap_center=None,
                cam_depth_remap_range=None,
                cam_segment_tool="",
                cam_segment_model_dir="",
                dvrk_cal_file='',
                ):
        self._arm_names = arm_names
        self._arms = {}
        self._seed = 0




        for name in arm_names:
            if name in ["PSM1", "PSM2"]:
                psm_args = {
                    "arm_name":name,
                    "ws_x":ws_x,
                    "ws_y":ws_y,
                    "ws_z":ws_z,
                    "action_mode":'yaw',
                    "reset_q":psm_reset_q,
                    "open_gripper_deg":psm_open_gripper_deg,
                    "init_gripper_quat":psm_init_gripper_quat,
                    "init_pose_low_gripper": psm_init_pose_low_gripper,
                    "init_pose_high_gripper": psm_init_pose_high_gripper,
                    "max_step_pos":psm_max_step_pos,
                    "max_step_rot":psm_max_step_rot,
                    }

                if dvrk_cal_file!='':
                    add_args = load_yaml(dvrk_cal_file)
                    psm_args.update(add_args)
                from gym_ras.env.embodied.dvrk.psm import SinglePSM
                self._arms[name] = SinglePSM(
                                    **psm_args
                                    )   
            else:
                raise NotImplementedError
        self._cam_device = RGBD_CAM(device=rgbd_device,
                                image_height = cam_image_height,
                                image_width = cam_image_width,
                                depth_remap_center=cam_depth_remap_center,
                                depth_remap_range=cam_depth_remap_range,
                                segment_tool = cam_segment_tool,
                                segment_model_dir = cam_segment_model_dir,
                                )
        self._oracle_device = OracleInput(device=oracle_device)
    def render(self):
        return self._cam_device.render()

    def get_oracle_action(self,):
        return self._oracle_device.get_oracle_action()

    def step(self, action):
        _psm = self._arms[self._arm_names[0]]
        _psm.step(action)
        obs = _psm.get_obs()
        reward = 0
        done=False
        info={"fsm": "in_progress"}
        return obs, reward, done, info
        
    def reset(self,):
        _psm = self._arms[self._arm_names[0]]
        _psm.reset_pose()
        _psm.move_gripper_init_pose()
        return _psm.get_obs()

    @property
    def observation_space(self):
        space = self._arms[self._arm_names[0]].obs_space
        obs = {}
        obs['gripper_state'] = gym.spaces.Box(space['gripper_state'][0], space['gripper_state'][1], (1,), dtype=np.float32)
        ws = space['tip_pos']
        _low = ws[:,0]
        _high = ws[:,1]
        obs['robot_prio'] = gym.spaces.Box(_low, _high, dtype=np.float32)
        return gym.spaces.Dict(obs)

    @property
    def workspace_limit(self):
        return self._arms[self._arm_names[0]].workspace_limit

        
    @property
    def action_space(self,):
        if len(self._arm_names) == 1:
            low_high = self._arms[self._arm_names[0]].act_space
        else:
            raise NotImplementedError
        return gym.spaces.Box(low=low_high[0],high=low_high[1])
    def __del__(self):
        for k, v in self._arms.items():
            del v

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        _seed = seed
        for k,v in self._arms.items():
            _seed -=1
            v.seed = _seed

    @property
    def reward_dict(self):
        return {"done_success": 0, 
        "done_fail":0, 
        "in_progress":0, 
        "progress_fail":0}



if __name__ == "__main__":
    env = NeedlePick()
    obs = env.reset()

    # for i in range(10):
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)

    from gym_ras.env.wrapper import Visualizer
    env =Visualizer(env,update_hz=100)
    _ = env.reset()
    img = env.render()
    img_break = env.cv_show(imgs=img)   
    for i in range(20):
        img = env.render()
        # action = env.action_space.sample()
        action = env.get_oracle_action()
        obs, reward, done, info = env.step(action)
        img_break = env.cv_show(imgs=img)
        if img_break:
            break

