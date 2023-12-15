from pathlib import Path
class RGBD_CAM():
    def __init__(self,
                    device="rs435",
                    image_height = 600,
                    image_width = 600,
                    depth_remap_center=None,
                    depth_remap_range=None,
                    segment_tool="",
                    segment_model_dir="",
                    ):
        self._image_height = image_height
        self._image_width = image_width
        self._depth_remap_center = depth_remap_center
        self._depth_remap_range = depth_remap_range
        if device == "rs435":
            from gym_ras.tool.rs435 import RS435_ROS_Engine
            self._device = RS435_ROS_Engine(image_height = image_height,
                                        image_width = image_width,
                                        depth_remap_center=depth_remap_center,
                                        depth_remap_range=depth_remap_range,)
        else:
            raise NotImplementedError
        
        if segment_tool == "detectron":
            from gym_ras.tool.seg_tool import DetectronPredictor
            assert segment_model_dir != ""
            _dir = Path(segment_model_dir)
            self._segment = DetectronPredictor(cfg_dir=str(_dir / "segment.yaml"),
                                            model_dir=str(_dir / "model_best.pth"),)
        elif segment_tool == "":
            self._segment = None
        else:
            raise NotImplementedError()
    def render(self,):
        img = self._device.get_image()
        img["mask"] = {}
        if self._segment is not None:
            masks = self._segment.predict(img['rgb'])
            if len(masks)>0:
                img.update({"mask": {self.segment_id_map[k]: v[0] for k,v in masks.items()}})
        return img
    @property
    def segment_id_map(self,):
        return {0: "psm1", 1: "stuff"}
