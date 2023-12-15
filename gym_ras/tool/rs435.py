import rospy
from rospy.numpy_msg import numpy_msg
from gym_ras.tool.ros_tool import ensure_sub_topic, safe_init_ros_node
from gym_ras.tool.common import scale_arr
from sensor_msgs.msg import Image
import numpy as np
import time
class RS435_ROS_Engine():
    def __init__(self,
    image_height = 600,
    image_width = 600,
    depth_remap_center=None,
    depth_remap_range=None,
                    ):
        print("init RS435_ROS_Engine....", end="")
        safe_init_ros_node("gym_ras")
        self._sub_rgb = rospy.Subscriber(ensure_sub_topic('/camera/color/image_raw'), numpy_msg(Image), self._sub_rgb_cb)
        self._sub_depth = rospy.Subscriber(ensure_sub_topic('/camera/aligned_depth_to_color/image_raw'), numpy_msg(Image), self._sub_depth_cb)
        self._data = {"rgb": None, "point_cloud": None}
        time.sleep(1)
        self.depth_remap_center = None
        self.depth_remap_range = None
        self._image_height = image_height
        self._image_width = image_width
        print("finish")
        
    def _sub_rgb_cb(self, data):
        img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self._data["rgb"] = img
    def _sub_depth_cb(self, data):
        img = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width, -1)
        self._data["point_cloud"] = img[:,:,0]
    def get_image(self,):
        _s = self._data["rgb"].shape
        _w = np.int(np.clip(_s[0] -self._image_width, a_min=0, a_max=_s[0]) / 2)
        _h = np.int(np.clip(_s[1] -self._image_height,a_min=0, a_max=_s[1]) / 2)
        img = {}
        img["rgb"] = self._data["rgb"][_w:-_w+1,_h:-_h+1,:]
  
        img["point_cloud"] = self._data["point_cloud"][_w:-_w+1,_h:-_h+1]
        if self.depth_remap_center is None:
            _high = np.max(img["point_cloud"])/1000
            _low = np.min(img["point_cloud"])/1000
            self.depth_remap_center = (_high+_low)/2
            self.depth_remap_range = (_high-_low)/2
        _high =  self.depth_remap_center+self.depth_remap_range/2
        _low =  self.depth_remap_center-self.depth_remap_range/2
        img["depth"] = np.uint8(scale_arr(img["point_cloud"]/1000,_low,_high,0,255))

        
        return img


if __name__ == '__main__':
    engine = RS435_ROS_Engine()
    

    from gym_ras.tool.img_tool import CV2_Visualizer
    visualizer = CV2_Visualizer( update_hz=60)
    # img = {"rgb":rgb, "depth": depth}
    is_quit = False
    while not is_quit:
        img = engine.get_image()
        is_quit = visualizer.cv_show(img)
