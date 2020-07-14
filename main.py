if __name__ == "__main__":
  import sys, argparse, time, os
  parser = argparse.ArgumentParser()
  print("RSS 2020: Learning Memory-Based Control for Human-Scale Bipedal Locomotion")
  print("\tJonah Siekmann, Srikar Valluri, Jeremy Dao, Lorenzo Bermillo, Helei Duan, Alan Fern, Jonathan Hurst")

  if len(sys.argv) < 2:
    print("Usage: python main.py [option]", sys.argv)
    print("\t potential options are: 'ppo', 'extract', 'eval', 'cassie'")
    exit(1)

  option = sys.argv[1]
  sys.argv.remove(sys.argv[1])

  if option == 'eval':
    from util import eval_policy
    import torch

    model = sys.argv[1]
    sys.argv.remove(sys.argv[1])

    parser.add_argument("--traj_len", default=300, type=int)
    args = parser.parse_args()

    model = torch.load(model)

    eval_policy(model, max_traj_len=args.traj_len, visualize=True, verbose=True)
    exit()

  if option == 'cassie':
    from cassie.udp import run_udp

    policies = sys.argv[1:]

    run_udp(policies)
    exit()

  if option == 'extract':
    from algos.extract import run_experiment

    parser.add_argument("--policy", "-p", default=None,           type=str)
    parser.add_argument("--layers",       default="256,256",      type=str) 
    parser.add_argument("--logdir",       default='logs/extract', type=str)

    parser.add_argument("--workers",      default=4,              type=int)
    parser.add_argument("--points",       default=5000,           type=int)
    parser.add_argument("--batch_size",   default=16,             type=int)
    parser.add_argument("--epochs",       default=500,            type=int)

    parser.add_argument("--lr",           default=1e-5,           type=float)
    args = parser.parse_args()
    if args.policy is None:
      print("Please provide a --policy argument.")
      exit(1)
    run_experiment(args)
    exit()


  # Options common to all RL algorithms.
  elif option == 'ppo':
    """
      Utility for running Proximal Policy Optimization.

    """
    from algos.ppo import run_experiment
    parser.add_argument("--timesteps",          default=1e6,           type=float) # timesteps to run experiment for
    parser.add_argument('--discount',           default=0.99,          type=float) # the discount factor
    parser.add_argument('--std',                default=0.13,          type=float) # the fixed exploration std
    parser.add_argument("--a_lr",               default=1e-4,          type=float) # adam learning rate for actor
    parser.add_argument("--c_lr",               default=1e-4,          type=float) # adam learning rate for critic
    parser.add_argument("--eps",                default=1e-6,          type=float) # adam eps
    parser.add_argument("--kl",                 default=0.02,          type=float) # kl abort threshold
    parser.add_argument("--grad_clip",          default=0.05,          type=float) # gradient norm clip

    parser.add_argument("--batch_size",         default=64,            type=int)   # batch size for policy update
    parser.add_argument("--epochs",             default=3,             type=int)   # number of updates per iter
    parser.add_argument("--workers",            default=4,             type=int)   # how many workers to use for exploring in parallel
    parser.add_argument("--seed",               default=0,             type=int)   # random seed for reproducibility
    parser.add_argument("--traj_len",           default=1000,          type=int)   # max trajectory length for environment
    parser.add_argument("--prenormalize_steps", default=10000,         type=int)   # number of samples to get normalization stats 
    parser.add_argument("--sample",             default=5000,          type=int)   # how many samples to do every iteration

    parser.add_argument("--layers",             default="128,128",     type=str)   # hidden layer sizes in policy
    parser.add_argument("--save_actor",         default=None,          type=str)   # where to save the actor (default=logdir)
    parser.add_argument("--save_critic",        default=None,          type=str)   # where to save the critic (default=logdir)
    parser.add_argument("--logdir",             default="./logs/ppo/", type=str)   # where to store log information
    parser.add_argument("--nolog",              action='store_true')               # store log data or not.
    parser.add_argument("--recurrent",          action='store_true')               # recurrent policy or not
    parser.add_argument("--randomize",          action='store_true')               # randomize dynamics or not
    args = parser.parse_args()

    run_experiment(args)

  elif option == 'pca':
    from algos.pca import run_pca
    import torch
    model = sys.argv[1]
    sys.argv.remove(sys.argv[1])


    model = torch.load(model)

    run_pca(model)
    exit()

  else:
    print("Invalid option '{}'".format(option))
