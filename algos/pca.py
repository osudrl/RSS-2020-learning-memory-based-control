import ray
import torch
import torch.nn as nn
import numpy as np
import locale, os, time, sys
from pathlib import Path

from util import env_factory
import matplotlib.pyplot as plt

def get_hiddens(policy):
  hiddens = []
  if hasattr(policy, 'hidden'):
    hiddens += [h.data for h in policy.hidden]
  
  if hasattr(policy, 'cells'):
    hiddens += [c.data for c in policy.cells]

  if hasattr(policy, 'latent'):
    hiddens += [l for l in policy.latent]

  return torch.cat([layer.view(-1) for layer in hiddens]).numpy()

def run_pca(policy):
  max_traj_len = 1000
  from sklearn.decomposition import PCA
  with torch.no_grad():
    env = env_factory(False)()
    state = env.reset()

    done = False
    timesteps = 0
    eval_reward = 0

    if hasattr(policy, 'init_hidden_state'):
      policy.init_hidden_state()

    mems = []
    while not done and timesteps < max_traj_len:

      env.speed      = 0.5

      action = policy.forward(torch.Tensor(state)).numpy()
      state, reward, done, _ = env.step(action)
      env.render()
      eval_reward += reward
      timesteps += 1

      memory = get_hiddens(policy)
      mems.append(memory)

    data = np.vstack(mems)

    pca = PCA(n_components=2)

    fig = plt.figure()
    plt.axis('off')
    base = (0.05, 0.05, 0.05)

    components = pca.fit_transform(data)

    x = components[:,0]
    y = components[:,1]
    c = []
    for i in range(len(x)):
      c.append(np.hstack([base, (len(x) - i/2) / len(x)]))

    plt.scatter(x, y, color=c, s=0.8)
    plt.show()
    plt.close()
