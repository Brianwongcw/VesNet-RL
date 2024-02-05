import sys
sys.path.append('/home/bmt-brian/Brian/code/vesnet/')
from VesselEnv import VesselEnv, DiscreteAction, create_configs_rand
import ext.dreamerv2.dreamerv2.api as dv2

config = dv2.defaults.update({
    'task': 'us_vessel',
    'logdir': './logdir/vessel_64',
    'log_every': 2e3,
    'train_every': 10,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
    'time_limit':200,
}).parse_flags()

ves_config=create_configs_rand(1)
env = VesselEnv(configs=ves_config)
env = DiscreteAction(env)
dv2.train(env, config)
