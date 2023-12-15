"""
refer to domain randomization tech: https://arxiv.org/pdf/2208.04171.pdf
"""

from gym_ras.env.wrapper.base import BaseWrapper
import numpy as np
import gym
import cv2
from copy import deepcopy
class ImageNoise(BaseWrapper):
    def __init__(self, env,
                pns_noise_amount=0.05,
                pns_noise_balance=0.5,
                gaussian_blur_kernel=3,
                gaussian_blur_sigma=0.8,
                cutoff_circle_r_low=0.05,
                cutoff_circle_r_high=0.1,
                cutoff_circle_num_low=2,
                cutoff_circle_num_high=8,
                cutoff_rec_w_low=0.05,
                cutoff_rec_w_high=0.1,
                cutoff_rec_h_low=0.05,
                cutoff_rec_h_high=0.1,
                cutoff_rec_num_low=3,
                cutoff_rec_num_high=10,
                cutoff_line_w_low=0.5,
                cutoff_line_w_high=1,
                cutoff_line_h_low=0.9,
                cutoff_line_h_high=1,
                cutoff_line_num_low=3,
                cutoff_line_num_high=10,
                cutoff_mask_min=0.2,
                 **kwargs):
        super().__init__(env)
        self._pns_noise_amount = pns_noise_amount
        self._pns_noise_balance = pns_noise_balance
        self._gaussian_blur_kernel = gaussian_blur_kernel
        self._gaussian_blur_sigma = gaussian_blur_sigma
        self._image_noise_rng = np.random.RandomState(0)
        self._circle = {}
        self._rec = {}
        self._line = {}
        self._circle["radius"] = [cutoff_circle_r_low, cutoff_circle_r_high]
        self._circle["num"] = [cutoff_circle_num_low, cutoff_circle_num_high]
        self._rec["width"] = [cutoff_rec_w_low, cutoff_rec_w_high]
        self._rec["height"] = [cutoff_rec_h_low, cutoff_rec_h_high]
        self._rec["num"] = [cutoff_rec_num_low, cutoff_rec_num_high]

        self._line["width"] = [cutoff_line_w_low, cutoff_line_w_high]
        self._line["height"] = [cutoff_line_h_low, cutoff_line_h_high]
        self._line["num"] = [cutoff_line_num_low, cutoff_line_num_high]
        self._cutoff_mask_min = cutoff_mask_min

    def render(self,):
        image = self.env.render()
        img = self._post_process(image)
        return img


    def _post_process(self, img):
        img["rgb"] = self._add_pepper_and_salt_nosie(img["rgb"])
        img["depth"] = self._add_pepper_and_salt_nosie(img["depth"])

        img["rgb"] = self._add_gaussian_blur(img["rgb"])
        img["depth"] = self._add_gaussian_blur(img["depth"])
        
        loop = 0
        while True:
            _img = deepcopy(img)
            loop +=1
            # print(loop)
            mask_amount = {k: np.sum(v) for k,v in _img["mask"].items()}

            # cirlce cutoff
            _num = {}
            _num["circle"] = self._image_noise_rng.randint(low=self._circle["num"][0], 
                                                 high=self._circle["num"][1]+1,)
            _num["rectangle"] = self._image_noise_rng.randint(low=self._circle["num"][0], 
                                                 high=self._rec["num"][1]+1,)
            _num["line"] = self._image_noise_rng.randint(low=self._line["num"][0], 
                                                 high=self._line["num"][1]+1,)
            
            for k,v in _num.items():
                for _ in range(v):
                    _img = self._cutoff(_img, cutoff_type=k)

            # # retangle cutoff
            # _num = self._image_noise_rng.randint(low=self._cutoff_circle["num_low"], 
            #                                      high=self._cutoff_circle["num_high"]+1,)
            # for _ in range(_num):
            #     _img = self._cutoff(_img, cutoff_type="rectangle")

            amount_check = True
            for k, v in _img["mask"].items():
                if np.sum(v) < mask_amount[k]*self._cutoff_mask_min:
                     amount_check = False
            if amount_check:
                break
        img = _img

        return img

    def _add_gaussian_blur(self, img):
        img = deepcopy(img)
        return cv2.GaussianBlur(img,(self._gaussian_blur_kernel,self._gaussian_blur_kernel), self._gaussian_blur_sigma, self._gaussian_blur_sigma)

    def _add_pepper_and_salt_nosie(self,image):
        image = deepcopy(image)
        s_vs_p = self._pns_noise_balance
        amount = self._pns_noise_amount
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    
    def _draw_rec(self, image, cx, cy, width, height, angle, rgb_list):
        rot_rectangle = ((np.int(cy*image.shape[1]),np.int(cx*image.shape[0]), ), 
                    (np.int(height*image.shape[1]), np.int(width*image.shape[0])), np.int(angle*180))
        box = cv2.boxPoints(rot_rectangle) 
        box = np.int0(box) #Convert into integer values
        rgb_list = np.uint8(np.array(rgb_list)*255).tolist()
        rgb_list.reverse()
        image = cv2.drawContours(image,[box],contourIdx=0,color=tuple(rgb_list),thickness=cv2.FILLED)
        return image

    def _draw_line(self, image, cx, cy, width, height, angle, rgb_list, ratio_w2h=0.01):
        self._draw_rec(image, cx, cy, width*ratio_w2h, height, angle, rgb_list)
        return image

    def _draw_circle(self, image, cx, cy, radius, rgb_list):
        cx = np.int(cx*image.shape[0])
        cy = np.int(cy*image.shape[1])
        radius = np.int(min(image.shape[0], image.shape[1])*radius/2)
        rgb_list = np.uint8(np.array(rgb_list)*255).tolist()
        rgb_list.reverse()
        image = cv2.circle(image, (cy,cx), radius, color=tuple(rgb_list),thickness=cv2.FILLED)
        return image

    def _cutoff(self, img, cutoff_type="circle"):
        img = deepcopy(img)
        w, h, c = img['rgb'].shape
        
        args = {}
        args["cx"] = self._image_noise_rng.uniform(low=0.0, high=1.0,)
        args["cy"] = self._image_noise_rng.uniform(low=0.0, high=1.0,)
        args["rgb_list"] =  self._image_noise_rng.uniform(low=0.0, high=1.0, size=3,).tolist()
        if cutoff_type=="circle":
            args["radius"] = self._image_noise_rng.uniform(low=self._circle["radius"][0], high=self._circle["radius"][1])
            _call = getattr(self, "_draw_circle")
            
        elif cutoff_type=="rectangle":
            args["width"] = self._image_noise_rng.uniform(low=self._rec["width"][0], high=self._rec["width"][1]) 
            args["height"] = self._image_noise_rng.uniform(low=self._rec["height"][0], high=self._rec["height"][1])
            args["angle"] = self._image_noise_rng.uniform(low=0, high=1)
            _call = getattr(self, "_draw_rec")
        elif cutoff_type=="line":
            args["width"] = self._image_noise_rng.uniform(low=self._line["width"][0], high=self._line["width"][1]) 
            args["height"] = self._image_noise_rng.uniform(low=self._line["height"][0], high=self._line["height"][1])
            args["angle"] = self._image_noise_rng.uniform(low=0, high=1)
            _call = getattr(self, "_draw_line")

        for k in ["rgb", "depth"]:
            img[k] = _call(image=img[k],**args)

        _mask_dict = {}
        args.update({"rgb_list": [0,0,0]})
        for k, v in img["mask"].items():
            _bool_mat = v.copy()
            _mat = np.zeros(_bool_mat.shape, dtype=np.uint8)
            _mat[_bool_mat] = 1

            _mat = _call(image=_mat,**args)
            _mat = _mat!=0
            _mask_dict[k] = _mat
        img.update({"mask": _mask_dict})
        return img
    
    def _init_rng(self, seed):
        image_noise_seed = np.uint32(seed)
        print("image_noise_seed:", image_noise_seed)
        self._image_noise_rng = np.random.RandomState(image_noise_seed)
        if self.env.is_wrapper:
            self.env._init_rng(self.env.seed)
