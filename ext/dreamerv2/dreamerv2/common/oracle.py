import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from . import dists
from . import tfutils
class OracleAgent:

  def __init__(self, act_space, logprob=False, env=None):
    self.act_space = act_space['action']
    self.logprob = logprob
    self.env = env

  def __call__(self, obs, state=None, mode=None):
    # print("call get oracle")
    _action = self.env.get_oracle_action()
    if hasattr(self.act_space, 'n'):
      _a = np.zeros(self.act_space.n)
      _a[_action] = 1
      _dist = dists.OneHotDist(probs=tf.convert_to_tensor(_a, np.float32))
    else:
      dist = tfd.Normal(tf.convert_to_tensor(_action, np.float32), 0)
      _dist = tfd.Independent(dist, 1)
    action = _dist.sample(len(obs['is_first']))
    output = {'action': action}
    if self.logprob:
      output['logprob'] = _dist.log_prob(action)
    return output, None