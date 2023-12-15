from gym_ras.tool.img_tool import CV2_Visualizer
from gym_ras.tool.rs435 import RS435_ROS_Engine
from gym_ras.tool.seg_tool import DetectronPredictor
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, default="./data/sample_image")
args = parser.parse_args()


engine = RS435_ROS_Engine()
visualizer = CV2_Visualizer( update_hz=10,
                            render_dir=args.savedir)
# img = {"rgb":rgb, "depth": depth}
is_quit = False
while not is_quit:
    img = engine.get_image()
    visualizer.cv_show(img)
    is_quit = visualizer.cv_show(img)
    
    # print(results)
