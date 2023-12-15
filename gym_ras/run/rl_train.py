import sys
sys.path.append('/home/bmt-brian/Brian/code/VesNet-RL/')
from VesselEnv import VesselEnv, create_configs_rand
from gym_ras.api import make_env
from gym_ras.env.wrapper import Visualizer
from gym_ras.tool.config import Config, load_yaml
import argparse
from pathlib import Path
from datetime import datetime



parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=str, default="dreamerv2")
parser.add_argument('--baseline-tag', type=str, nargs='+', default=[])
parser.add_argument('--env-tag', type=str, nargs='+', default=[])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--logdir', type=str, default="./log")
parser.add_argument('--reload-dir', type=str, default="")
parser.add_argument('--online-eval', action='store_true')
parser.add_argument('--online-eps', type=int, default=10)
parser.add_argument('--visualize', action="store_true")
args = parser.parse_args()

if args.online_eval:
    assert args.reload_dir!=""



# import pdb;pdb.set_trace()
if args.reload_dir=="":
    env_config = create_configs_rand(1)
    env = VesselEnv(configs=env_config)
    method_config_dir =  Path(".") 
    method_config_dir = method_config_dir / 'gym_ras' / 'config' / str(args.baseline + ".yaml")
    if not method_config_dir.is_file():
        raise NotImplementedError("baseline not implement")
    yaml_dict = load_yaml(method_config_dir)
    yaml_config = yaml_dict["default"].copy()
    baseline_config = Config(yaml_config)
    # print(train_config)
    for tag in args.baseline_tag:
        baseline_config = baseline_config.update(yaml_dict[tag])

    _env_name = "ras"
    _baseline_name = baseline_config.baseline_name
    if len(args.baseline_tag)!=0:
        _baseline_name += "-" + "-".join(args.baseline_tag)

    if len(args.env_tag)!=0:
        _env_name += "-" + "-".join(args.env_tag)


    logdir = str(Path(args.logdir) / str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+'@'+_env_name +'@'+ _baseline_name ))
    logdir = Path(logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    if args.baseline == "DreamerfD":
        baseline_config = baseline_config.update({
         'bc_dir': str(logdir + '/train_episodes/oracle'),
        'logdir': str(logdir),})
    elif args.baseline == "dreamerv2":
        baseline_config = baseline_config.update({
        'logdir': str(logdir),})
    
    baseline_config.save(str(logdir / "baseline_config.yaml"))
    # env_config.save(str(logdir / "env_config.yaml"))


else:
    reload_dir = Path(args.reload_dir)
    yaml_dict = load_yaml(str(reload_dir / "baseline_config.yaml"))
    baseline_config = Config(yaml_dict)

    yaml_dict = load_yaml(str(reload_dir / "env_config.yaml"))
    env_config = Config(yaml_dict)

    env_config = create_configs_rand(1)
    env = VesselEnv(configs=env_config)
    logdir = reload_dir
    baseline_config = baseline_config.update({"logdir": str(logdir)})
    baseline_config.save(str(logdir / "baseline_config.yaml"))
    env_config.save(str(logdir / "env_config.yaml"))

if args.visualize:
    env = Visualizer(env)
    
if baseline_config.baseline_name == "dreamerv2":
    import pdb; pdb.set_trace()
    from gym_ras.rl import train_dreamerv2
    train_dreamerv2.train(env, baseline_config, )
