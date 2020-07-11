from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import CassieTrajectory

from math import floor

import numpy as np 
import os
import random

import pickle

class CassieEnv:
  def __init__(self, dynamics_randomization=False):
    self.sim = CassieSim("./cassie/cassiemujoco/cassie.xml")
    self.vis = None

    self.dynamics_randomization = dynamics_randomization

    state_est_size = 46
    clock_size     = 2
    speed_size     = 1

    self.observation_space = np.zeros(state_est_size + clock_size + speed_size)
    self.action_space      = np.zeros(10)

    dirname = os.path.dirname(__file__)

    traj_path = os.path.join(dirname, "trajectory", "stepdata.bin")

    self.trajectory = CassieTrajectory(traj_path)

    self.P = np.array([100,  100,  88,  96,  50]) 
    self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

    self.u = pd_in_t()

    self.cassie_state = state_out_t()

    self.simrate = 60 # simulate X mujoco steps with same pd target
                      # 60 brings simulation from 2000Hz to roughly 30Hz
    self.time    = 0  # number of time steps in current episode
    self.phase   = 0  # portion of the phase the robot is in
    self.counter = 0  # number of phase cycles completed in episode

    # NOTE: a reference trajectory represents ONE phase cycle
    self.phaselen = floor(len(self.trajectory) / self.simrate) - 1

    # see include/cassiemujoco.h for meaning of these indices
    self.pos_idx   = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
    self.vel_idx   = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
    self.offset    = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
    self.speed     = 0
    self.phase_add = 1

    # Record default dynamics parameters
    self.default_damping = self.sim.get_dof_damping()
    self.default_mass = self.sim.get_body_mass()
    self.default_ipos = self.sim.get_body_ipos()

  def step_simulation(self, action):

      target = action + self.offset
      
      self.u = pd_in_t()
      for i in range(5):
          # TODO: move setting gains out of the loop?
          # maybe write a wrapper for pd_in_t ?
          self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
          self.u.rightLeg.motorPd.pGain[i] = self.P[i]

          self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
          self.u.rightLeg.motorPd.dGain[i] = self.D[i]

          self.u.leftLeg.motorPd.torque[i]  = 0 # Feedforward torque
          self.u.rightLeg.motorPd.torque[i] = 0 

          self.u.leftLeg.motorPd.pTarget[i]  = target[i]
          self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

          self.u.leftLeg.motorPd.dTarget[i]  = 0
          self.u.rightLeg.motorPd.dTarget[i] = 0

      self.cassie_state = self.sim.step_pd(self.u)

  def step(self, action):
      for _ in range(self.simrate):
          self.step_simulation(action)

      height = self.sim.qpos()[2]

      self.time  += 1
      self.phase += self.phase_add

      if self.phase > self.phaselen:
          self.phase = 0
          self.counter += 1

      # Early termination
      done = not(height > 0.4 and height < 3.0)

      reward = self.compute_reward()

      if reward < 0.3:
          done = True

      return self.get_full_state(), reward, done, {}

  def reset(self):
      self.phase = random.randint(0, self.phaselen)
      self.time = 0
      self.counter = 0

      qpos, qvel = self.get_ref_state(self.phase)

      self.sim.set_qpos(qpos)
      self.sim.set_qvel(qvel)

      # Randomize dynamics:
      if self.dynamics_randomization:
          damp = self.default_damping
          weak_factor = 0.5
          strong_factor = 1.5
          pelvis_damp_range = [[damp[0], damp[0]], 
                               [damp[1], damp[1]], 
                               [damp[2], damp[2]], 
                               [damp[3], damp[3]], 
                               [damp[4], damp[4]], 
                               [damp[5], damp[5]]]                 # 0->5

          hip_damp_range = [[damp[6]*weak_factor, damp[6]*strong_factor],
                            [damp[7]*weak_factor, damp[7]*strong_factor],
                            [damp[8]*weak_factor, damp[8]*strong_factor]]  # 6->8 and 19->21

          achilles_damp_range = [[damp[9]*weak_factor,  damp[9]*strong_factor],
                                 [damp[10]*weak_factor, damp[10]*strong_factor], 
                                 [damp[11]*weak_factor, damp[11]*strong_factor]] # 9->11 and 22->24

          knee_damp_range     = [[damp[12]*weak_factor, damp[12]*strong_factor]]   # 12 and 25
          shin_damp_range     = [[damp[13]*weak_factor, damp[13]*strong_factor]]   # 13 and 26
          tarsus_damp_range   = [[damp[14], damp[14]]]             # 14 and 27
          heel_damp_range     = [[damp[15], damp[15]]]                           # 15 and 28
          fcrank_damp_range   = [[damp[16]*weak_factor, damp[16]*strong_factor]]   # 16 and 29
          prod_damp_range     = [[damp[17], damp[17]]]                           # 17 and 30
          foot_damp_range     = [[damp[18]*weak_factor, damp[18]*strong_factor]]   # 18 and 31

          side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
          damp_range = pelvis_damp_range + side_damp + side_damp
          damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

          hi = 1.3
          lo = 0.7
          m = self.default_mass
          pelvis_mass_range      = [[lo*m[1],  hi*m[1]]]  # 1
          hip_mass_range         = [[lo*m[2],  hi*m[2]],  # 2->4 and 14->16
                                    [lo*m[3],  hi*m[3]], 
                                    [lo*m[4],  hi*m[4]]] 

          achilles_mass_range    = [[lo*m[5],  hi*m[5]]]  # 5 and 17
          knee_mass_range        = [[lo*m[6],  hi*m[6]]]  # 6 and 18
          knee_spring_mass_range = [[lo*m[7],  hi*m[7]]]  # 7 and 19
          shin_mass_range        = [[lo*m[8],  hi*m[8]]]  # 8 and 20
          tarsus_mass_range      = [[lo*m[9],  hi*m[9]]]  # 9 and 21
          heel_spring_mass_range = [[lo*m[10], hi*m[10]]] # 10 and 22
          fcrank_mass_range      = [[lo*m[11], hi*m[11]]] # 11 and 23
          prod_mass_range        = [[lo*m[12], hi*m[12]]] # 12 and 24
          foot_mass_range        = [[lo*m[13], hi*m[13]]] # 13 and 25

          side_mass = hip_mass_range + achilles_mass_range \
                      + knee_mass_range + knee_spring_mass_range \
                      + shin_mass_range + tarsus_mass_range \
                      + heel_spring_mass_range + fcrank_mass_range \
                      + prod_mass_range + foot_mass_range

          mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
          mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

          delta_y_min, delta_y_max = self.default_ipos[4] - 0.07, self.default_ipos[4] + 0.07
          delta_z_min, delta_z_max = self.default_ipos[5] - 0.04, self.default_ipos[5] + 0.04
          com_noise = [0, 0, 0] + [np.random.uniform(-0.25, 0.06)] + [np.random.uniform(delta_y_min, delta_y_max)] + [np.random.uniform(delta_z_min, delta_z_max)] + list(self.default_ipos[6:])

          self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
          self.sim.set_body_mass(np.clip(mass_noise, 0, None))
          self.sim.set_body_ipos(com_noise)
      else:
          self.sim.set_dof_damping(self.default_damping)
          self.sim.set_body_mass(self.default_mass)
          self.sim.set_body_ipos(self.default_ipos)

      self.sim.set_const()

      self.cassie_state = self.sim.step_pd(self.u)
      self.speed        = np.random.uniform(-0.15, 0.8)

      return self.get_full_state()

  def compute_reward(self):
      qpos = np.copy(self.sim.qpos())
      qvel = np.copy(self.sim.qvel())

      ref_pos, _ = self.get_ref_state(self.phase)

      joint_error       = 0
      com_error         = 0
      orientation_error = 0
      spring_error      = 0

      # each joint pos
      weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]
      for i, j in enumerate(self.pos_idx):
          target = ref_pos[j]
          actual = qpos[j]

          joint_error += 30 * weight[i] * (target - actual) ** 2

      forward_diff = np.abs(qvel[0] - self.speed)
      if forward_diff < 0.05:
         forward_diff = 0

      y_vel = np.abs(qvel[1])
      if y_vel < 0.03:
        y_vel = 0

      straight_diff = np.abs(qpos[1])
      if straight_diff < 0.05:
        straight_diff = 0

      actual_q = qpos[3:7]
      target_q = [1, 0, 0, 0]
      orientation_error = 5 * (1 - np.inner(actual_q, target_q) ** 2)

      # left and right shin springs
      for i in [15, 29]:
          target = ref_pos[i]
          actual = qpos[i]

          spring_error += 1000 * (target - actual) ** 2      
      
      reward = 0.000 + \
               0.300 * np.exp(-orientation_error) + \
               0.200 * np.exp(-joint_error) +       \
               0.200 * np.exp(-forward_diff) +      \
               0.200 * np.exp(-y_vel) +             \
               0.050 * np.exp(-straight_diff) +     \
               0.050 * np.exp(-spring_error)

      return reward

  # get the corresponding state from the reference trajectory for the current phase
  def get_ref_state(self, phase=None):
      if phase is None:
          phase = self.phase

      if phase > self.phaselen:
          phase = 0

      pos = np.copy(self.trajectory.qpos[phase * self.simrate])

      ###### Setting variable speed  #########
      pos[0] *= self.speed
      pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter * self.speed
      ######                          ########

      # setting lateral distance target to 0
      # regardless of reference trajectory
      pos[1] = 0

      vel = np.copy(self.trajectory.qvel[phase * self.simrate])
      vel[0] *= self.speed

      return pos, vel

  def get_full_state(self):
      qpos = np.copy(self.sim.qpos())
      qvel = np.copy(self.sim.qvel()) 

      ref_pos, _ = self.get_ref_state(self.phase + self.phase_add)

      clock = [np.sin(2 * np.pi *  self.phase / self.phaselen),
               np.cos(2 * np.pi *  self.phase / self.phaselen)]
      
      ext_state = np.concatenate((clock, [self.speed]))

      # Use state estimator
      robot_state = np.concatenate([
          [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height], # pelvis height
          self.cassie_state.pelvis.orientation[:],                                   # pelvis orientation
          self.cassie_state.motor.position[:],                                       # actuated joint positions
          self.cassie_state.pelvis.translationalVelocity[:],                         # pelvis translational velocity
          self.cassie_state.pelvis.rotationalVelocity[:],                            # pelvis rotational velocity 
          self.cassie_state.motor.velocity[:],                                       # actuated joint velocities
          self.cassie_state.pelvis.translationalAcceleration[:],                     # pelvis translational acceleration
          self.cassie_state.joint.position[:],                                       # unactuated joint positions
          self.cassie_state.joint.velocity[:]                                        # unactuated joint velocities
      ])

      return np.concatenate([robot_state, ext_state])

  def render(self):
      if self.vis is None:
          self.vis = CassieVis(self.sim, "./cassie/cassiemujoco/cassie.xml")

      return self.vis.draw(self.sim)
