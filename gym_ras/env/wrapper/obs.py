from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
import gym 
import cv2

class OBS(BaseWrapper):
    """ specify observation """

    def __init__(self, env,
                    image_obs_key=["dsa"],
                    vector_obs_key=["gripper_state", "fsm_state", "robot_prio"],
                    direct_map_key=["fsm_state"],
                    direct_render_key=["rgb"],
                    is_vector2image=True,
                    image_resize=[64,64],
                    cv_interpolate="area",
                    **kwargs,
                 ):
        super().__init__(env,)
        self._image_obs_key = image_obs_key
        self._vector_obs_key = vector_obs_key
        self._is_vector2image = is_vector2image
        self._image_resize = image_resize
        self._direct_map_key = direct_map_key
        self._cv_interpolate = cv_interpolate
        self._direct_render_key = direct_render_key

        obs = self.reset()
        self._obs_shape = {k: v.shape if isinstance(v, np.ndarray) else None for k,v in obs.items()}
    def reset(self,):
        obs = self.env.reset()
        return self._get_obs(obs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._get_obs(obs)
        return obs, reward, done, info
    
    def _get_obs_low_high(self):
        low_high_tuple = lambda x: (x.low, x.high, x.shape)
        get_bound = lambda x, shape: x if x.shape[0] == shape else x*np.ones(shape)
        _list = [low_high_tuple(self.env.observation_space[k]) for k in self._vector_obs_key]
        if len(_list) == 1:
            k = _list[0]
            _low = get_bound(k[0], k[2])
            _high = get_bound(k[1], k[2])
        else:
            _low = np.concatenate(tuple([get_bound(k[0], k[2]) for k in _list]),axis=0)
            _high = np.concatenate(tuple([get_bound(k[1], k[2]) for k in _list]),axis=0)
        np_arr_func = lambda x: x if isinstance(x,np.ndarray) else np.array([x])
        return np_arr_func(_low), np_arr_func(_high)

    def _get_obs(self, _obs):
        obs = {}
        for v in self._direct_map_key:
            obs[v] = _obs[v]
        _img = self.env.render()
        if "depth" in _img:
            _img["depth"] = np.stack([_img["depth"]]*3, axis=2)
        for v in self._direct_render_key:
            obs[v] = _img[v]
            if self._image_resize[0]>0:
                obs[v] = cv2.resize(obs[v],  
                        tuple(self._image_resize),
                        interpolation={"nearest": cv2.INTER_NEAREST,
                                    "linear": cv2.INTER_LINEAR,
                                    "area": cv2.INTER_AREA,
                                    "cubic": cv2.INTER_CUBIC,}[self._cv_interpolate])

        np_arr_func = lambda x: x if isinstance(x,np.ndarray) else np.array([x])
        if len(self._image_obs_key) >0:
            obs["image"] = _img[self._image_obs_key[0]] \
                 if len(self._image_obs_key) ==1 \
                    else np.concatenate(tuple([_img[k] for k in self._image_obs_key]),axis=2)
            if self._image_resize[0]>0:
                obs["image"] = cv2.resize(obs["image"], 
                        tuple(self._image_resize),
                        interpolation={"nearest": cv2.INTER_NEAREST,
                                    "linear": cv2.INTER_LINEAR,
                                    "area": cv2.INTER_AREA,
                                    "cubic": cv2.INTER_CUBIC,}[self._cv_interpolate])
                    
        if len(self._vector_obs_key) >0:
            obs["vector"] = _obs[self._vector_obs_key[0]] \
                 if len(self._vector_obs_key) ==1 \
                    else np.concatenate(tuple([np_arr_func(_obs[k]) for k in self._vector_obs_key]), axis=0)
            _low, _high = self._get_obs_low_high()
            obs["vector"] = self._scale(obs["vector"], _low, _high, -np.ones(_low.shape), np.ones(_low.shape))
            if self._is_vector2image:
                obs["image"] = self._vector2image(obs["image"], obs["vector"])
                obs.pop('vector', None)

        return obs

    def _vector2image(self,image_in, vector,fill_channel=0, encode_method="pixel"):
        image = np.copy(image_in)  
        _value_norm = self._scale(vector, old_min=-1, old_max=1, new_min=0.0, new_max=255.0)
        _value_norm = _value_norm.astype(np.uint8)
        if encode_method == "column":
            _value_norm = np.tile(_value_norm, (image.shape[0], 1))
            _value_norm = np.transpose(_value_norm)
            image[0:_value_norm.shape[0], :, fill_channel] = _value_norm
        elif encode_method == "pixel":
            _shape = image.shape
            _x = np.reshape(image[:,:,fill_channel], (-1))
            _x[:_value_norm.shape[0]] = _value_norm
            _x = np.reshape(_x, image.shape[:2])
            image[:,:, fill_channel] = _x[:,:]
        # print(_value_norm)
        return image
    
    @staticmethod
    def _scale(_input, old_min,old_max,new_min,new_max):
        _in = _input
        _in = np.divide(_input-old_min,old_max-old_min)
        _in = np.multiply(_in,new_max-new_min) + new_min
        return _in
    
    @property
    def observation_space(self):
        obs = {}
        for v in self._direct_map_key:
            obs[v] = self.env.observation_space[v]
        for v in self._direct_render_key:
            obs[v] = gym.spaces.Box(0, 255, self._obs_shape[v], 
                                                dtype=np.uint8)

        if "image" in self._obs_shape:
            obs['image'] = gym.spaces.Box(0, 255, self._obs_shape["image"], 
                                                        dtype=np.uint8)

        if "vector" in self._obs_shape and (not self._is_vector2image):
            _low, _high = self._get_obs_low_high()
            obs['vector'] = gym.spaces.Box(-1, 1, self._obs_shape["vector"], 
                                                        dtype=np.float32)
                                                    
        return gym.spaces.Dict(obs)