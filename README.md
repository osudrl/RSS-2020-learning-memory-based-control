# Learning Memory-Based Control for Human-Scale Bipedal Locomotion

## Purpose

This repo is intended to serve as a foundation with which you can reproduce the results of the experiments detailed in our RSS 2020 paper, [Learning Memory-Based Control for Human-Scale Bipedal Locomotion](https://arxiv.org/abs/2006.02402).

## First-time setup
This repo requires [MuJoCo 2.0](http://www.mujoco.org/). We recommend that you use Ubuntu 18.04.

You will probably need to install the following packages:
```bash
pip3 install --user torch numpy ray tensorboard
sudo apt-get install -y curl git libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev net-tools unzip vim wget xpra xserver-xorg-dev patchelf
```

If you don't already have it, you will need to install MuJoCo. You will also need to obtain a license key `mjkey.txt` from the [official website](https://www.roboti.us/license.html). You can get a free 30-day trial if necessary.

```bash
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
mkdir ~/.mujoco
mv mujoco200_linux ~/.mujoco/mujoco200
cp [YOUR KEY FILE] ~/.mujoco/mjkey.txt
```

You will need to create an environment variable `LD_LIBRARY_PATH` to allow mujoco-py to find your mujoco directory. You can add it to your `~/.bashrc` or just enter it into the terminal every time you wish to use mujoco.
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
```

## Reproducing experiments

### Basics

To train a policy in accordance with the hyperparameters used in the paper, execute this command:

```bash
python3 main.py ppo --batch_size 64 --sample 50000 --epochs 8 --traj_len 300 --timesteps 60000000 --discount 0.95 --workers 56 --recurrent --randomize --layers 128,128 --std 0.13 --logdir LOG_DIRECTORY
```

To train a FF policy, simply remove the `--recurrent` argument. To train without dynamics randomization, remove the `--randomize` argument.


### Logging details / Monitoring live training progress
Tensorboard logging is enabled by default. After initiating an experiment, your directory structure would look like this:

```
logs/
├── [algo]
│     └── [New Experiment Logdir]
```

To see live training progress, run ```$ tensorboard --logdir=logs``` then navigate to ```http://localhost:6006/``` in your browser
