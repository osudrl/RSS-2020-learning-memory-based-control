import torch
import hashlib
import os
import numpy as np
from collections import OrderedDict

def create_logger(args):
  from torch.utils.tensorboard import SummaryWriter
  """Use hyperparms to set a directory to output diagnostic files."""

  arg_dict = args.__dict__
  assert "logdir" in arg_dict, \
    "You must provide a 'logdir' key in your command line arguments."

  # sort the keys so the same hyperparameters will always have the same hash
  arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

  # remove seed so it doesn't get hashed, store value for filename
  # same for logging directory
  if 'seed' in arg_dict:
    seed = str(arg_dict.pop("seed"))
  else:
    seed = None
  
  task_name = 'cassie'
  if 'policy' in arg_dict:
    task_name = os.path.normpath(arg_dict.pop('policy')).split(os.path.sep)
    task_name = '-'.join([x for x in task_name[-4:-1]])

  logdir = str(arg_dict.pop('logdir'))

  # get a unique hash for the hyperparameter settings, truncated at 10 chars
  if seed is None:
    arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6]
  else:
    arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6] + '-seed' + seed

  logdir     = os.path.join(logdir, task_name)
  output_dir = os.path.join(logdir, arg_hash)

  # create a directory with the hyperparm hash as its name, if it doesn't
  # already exist.
  os.makedirs(output_dir, exist_ok=True)

  # Create a file with all the hyperparam settings in plaintext
  info_path = os.path.join(output_dir, "experiment.info")
  file = open(info_path, 'w')
  for key, val in arg_dict.items():
      file.write("%s: %s" % (key, val))
      file.write('\n')

  logger = SummaryWriter(output_dir, flush_secs=0.1)
  print("Logging to " + str(output_dir))

  logger.taskname = task_name
  logger.dir = output_dir
  logger.arg_hash = arg_hash
  return logger

def train_normalizer(policy, min_timesteps, max_traj_len=1000, noise=0.5):
  with torch.no_grad():
    env = env_factory(policy.env_name)()
    env.dynamics_randomization = False

    total_t = 0
    while total_t < min_timesteps:
      state = env.reset()
      done = False
      timesteps = 0

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      while not done and timesteps < max_traj_len:
        if noise is None:
          action = policy.forward(state, update_norm=True, deterministic=False).numpy()
        else:
          action = policy.forward(state, update_norm=True).numpy() + np.random.normal(0, noise, size=policy.action_dim)

        state, _, done, _ = env.step(action)
        timesteps += 1
        total_t += 1

def eval_policy(policy, env, episodes=1, max_traj_len=400, verbose=True, visualize=False):
  with torch.no_grad():
    steps = 0
    ep_returns = []
    for _ in range(episodes):
      env.dynamics_randomization = False
      state = torch.Tensor(env.reset())
      done = False
      traj_len = 0
      ep_return = 0

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      while not done and traj_len < max_traj_len:
        action = policy(state)
        next_state, reward, done, _ = env.step(action.numpy())
        if visualize:
          env.render()
        state = torch.Tensor(next_state)
        ep_return += reward
        traj_len += 1
        steps += 1
      ep_returns += [ep_return]

  return np.mean(ep_returns)
  
def env_factory(dynamics_randomization, verbose=False, **kwargs):
    from functools import partial

    """
    Returns an *uninstantiated* environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 

    """
    from cassie.cassie import CassieEnv

    if verbose:
      print("Created cassie env with arguments:")
      print("\tdynamics randomization: {}".format(dynamics_randomization))
    return partial(CassieEnv, dynamics_randomization=dynamics_randomization)
