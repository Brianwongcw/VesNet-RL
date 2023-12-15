from gym_ras.api import make_env
from gym_ras.env.wrapper import Visualizer
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-p',type=int)
parser.add_argument('--repeat',type=int, default=1)
parser.add_argument('--action',type=str, default="3")
parser.add_argument('--yaml-dir', type=str, default="./gym_ras/config.yaml")
parser.add_argument('--yaml-tag', type=str, nargs='+', default=[])
parser.add_argument('--env-tag', type=str, nargs='+', default=[])
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

env, env_config = make_env(tags=args.env_tag, seed=args.seed)
if env_config.embodied_name == "dVRKEnv":
    env =  Visualizer(env, update_hz=100)
else:
    env =  Visualizer(env)
for _ in tqdm(range(args.repeat)):
    done = False
    obs = env.reset()
    # print("obs:", obs)
    while not done:
        # action = env.action_space.sample()
        # print(action)
        print("==========step", env.timestep, "===================")
        if any(i.isdigit() for i in args.action):
            action = int(args.action)
        elif args.action == "random":
            action = env.action_space.sample()
        elif args.action == "oracle":
            action = env.get_oracle_action()
        else:
            raise NotImplementedError
        print("step....")
        obs, reward, done, info = env.step(action)
        print_obs = obs.copy()
        print_obs = {k: v.shape if k in ["image","rgb","depth"] else v for k,v in print_obs.items()}
        print_obs = [str(k)+ ":" +str(v) for k,v in print_obs.items()]
        print(" | ".join(print_obs))
        print("reward:", reward, "done:", done,)
    
        # print("reward:", reward, "done:", done, "info:", info, "step:", env.timestep, "obs_key:", obs.keys(), "fsm_state:", obs["fsm_state"])
        # print("observation space: ", env.observation_space)
        img = env.render()
        # print(img.keys())
        # print(img)
        img.update({"image": obs['image']})
        if "dsa" in img:
            obs.update({"dsa": img["dsa"]})
        # print(action)
        
        if info['is_success']:
            break
        img_break = env.cv_show(imgs=img)
        if img_break:
            break
    if img_break:
        break