from gym_ras.env.wrapper.base import BaseWrapper
from gym_ras.tool.img_tool import CV2_Visualizer


class Visualizer(BaseWrapper):
    """ render image with GUI """
    def __init__(self, 
                    env, 
                    update_hz=-1, 
                    render_dir="/tmp/gym-ras/render/", 
                    vis_channel=[0, 1, 2], 
                    is_gray=False,
                    gui_shape=[600,600],
                    cv_interpolate="area",
                    **kwargs,
                     ):
        super().__init__(env)
        
        self._visualizer = CV2_Visualizer(update_hz=update_hz, 
                    render_dir=render_dir, 
                    vis_channel=vis_channel, 
                    is_gray=is_gray,
                    gui_shape=gui_shape,
                    cv_interpolate=cv_interpolate,)


    def cv_show(self, imgs,):
        return self._visualizer.cv_show(imgs,)

    def __del__(self):
        del self._visualizer