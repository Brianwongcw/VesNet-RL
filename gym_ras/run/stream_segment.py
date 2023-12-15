from gym_ras.tool.img_tool import CV2_Visualizer
from gym_ras.tool.rs435 import RS435_ROS_Engine
from gym_ras.tool.seg_tool import DetectronPredictor
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', type=str, required=True)
args = parser.parse_args()


_dir = Path(args.modeldir)

predictor = DetectronPredictor(model_dir= str(_dir / "model_best.pth"),
                    cfg_dir=str(_dir / 'segment.yaml'))
engine = RS435_ROS_Engine()
visualizer = CV2_Visualizer( update_hz=10)
# img = {"rgb":rgb, "depth": depth}
is_quit = False
while not is_quit:
    img = engine.get_image()
    masks = predictor.predict(img['rgb'])
    if len(masks)>0:
        img.update({"mask": {str(k): v[0] for k,v in masks.items()}})
    visualizer.cv_show(img)
    is_quit = visualizer.cv_show(img)
    
    # print(results)
