import time
import torch
import pickle
import platform

import sys
import datetime

import select, termios, tty

from cassie.cassiemujoco.cassieUDP import *
from cassie.cassiemujoco.cassiemujoco_ctypes import *

import numpy as np

import math
import numpy as np

def inverse_quaternion(quaternion):
	result = np.copy(quaternion)
	result[1:4] = -result[1:4]
	return result

def quaternion_product(q1, q2):
	result = np.zeros(4)
	result[0] = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
	result[1] = q1[0]*q2[1]+q2[0]*q1[1]+q1[2]*q2[3]-q1[3]*q2[2]
	result[2] = q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
	result[3] = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
	return result

def rotate_by_quaternion(vector, quaternion):
	q1 = np.copy(quaternion)
	q2 = np.zeros(4)
	q2[1:4] = np.copy(vector)
	q3 = inverse_quaternion(quaternion)
	q = quaternion_product(q2, q3)
	q = quaternion_product(q1, q)
	result = q[1:4]
	return result

def quaternion2euler(quaternion):
	w = quaternion[0]
	x = quaternion[1]
	y = quaternion[2]
	z = quaternion[3]
	ysqr = y * y

	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))

	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))

	result = np.zeros(3)
	result[0] = X * np.pi / 180
	result[1] = Y * np.pi / 180
	result[2] = Z * np.pi / 180

	return result

def euler2quat(z=0, y=0, x=0):

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    result =  np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])
    if result[0] < 0:
    	result = -result
    return result

def check_stdin():
  return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def run_udp(args):
  from util.env import env_factory

  policy = torch.load(args.policy)
  #policy.eval()

  env = env_factory(policy.env_name)()
  if not env.state_est:
    print("This policy was not trained with state estimation and cannot be run on the robot.")
    raise RuntimeError

  print("This policy is: {}".format(policy.__class__.__name__))
  time.sleep(1)

  time_log   = [] # time stamp
  input_log  = [] # network inputs
  output_log = [] # network outputs 
  state_log  = [] # cassie state
  target_log = [] #PD target log

  clock_based = env.clock
  no_delta = env.no_delta

  u = pd_in_t()
  for i in range(5):
      u.leftLeg.motorPd.pGain[i] = env.P[i]
      u.leftLeg.motorPd.dGain[i] = env.D[i]
      u.rightLeg.motorPd.pGain[i] = env.P[i]
      u.rightLeg.motorPd.dGain[i] = env.D[i]

  if platform.node() == 'cassie':
      cassie = CassieUdp(remote_addr='10.10.10.3', remote_port='25010',
                              local_addr='10.10.10.100', local_port='25011')
  else:
      cassie = CassieUdp() # local testing

  print('Connecting...')
  y = None
  while y is None:
      cassie.send_pd(pd_in_t())
      time.sleep(0.001)
      y = cassie.recv_newest_pd()

  received_data = True
  t = time.monotonic()
  t0 = t

  print('Connected!\n')

  action = 0
  # Whether or not STO has been TOGGLED (i.e. it does not count the initial STO condition)
  # STO = True means that STO is ON (i.e. robot is not running) and STO = False means that STO is
  # OFF (i.e. robot *is* running)
  sto = True
  sto_count = 0

  orient_add = 0

  # We have multiple modes of operation
  # 0: Normal operation, walking with policy
  # 1: Start up, Standing Pose with variable height (no balance)
  # 2: Stop Drop and hopefully not roll, Damping Mode with no P gain
  operation_mode = 0
  standing_height = 0.7
  MAX_HEIGHT = 0.8
  MIN_HEIGHT = 0.4
  D_mult = 1  # Reaaaaaally bad stability problems if this is pushed higher as a multiplier
                   # Might be worth tuning by joint but something else if probably needed
  phase = 0
  counter = 0
  phase_add = 1
  speed = 0

  max_speed = 2
  min_speed = -1
  max_y_speed = 0.0
  min_y_speed = 0.0

  old_settings = termios.tcgetattr(sys.stdin)

  try:
    tty.setcbreak(sys.stdin.fileno())

    while True:
      t = time.monotonic()

      tt = time.monotonic() - t0

      # Get newest state
      state = cassie.recv_newest_pd()

      if state is None:
          print('Missed a cycle!                ')
          continue	

      if platform.node() == 'cassie':

        # Radio control
        orient_add -= state.radio.channel[3] / 60.0

        # Reset orientation on STO
        if state.radio.channel[8] < 0:
            orient_add = quaternion2euler(state.pelvis.orientation[:])[2]

            # Save log files after STO toggle (skipping first STO)
            if sto is False:
                #log(sto_count)
                sto_count += 1
                sto = True
                # Clear out logs
                time_log   = [] # time stamp
                input_log  = [] # network inputs
                output_log = [] # network outputs
                state_log  = [] # cassie state
                target_log = [] #PD target log

            if hasattr(policy, 'init_hidden_state'):
              print("RESETTING HIDDEN STATES TO ZERO!")
              policy.init_hidden_state()

        else:
            sto = False

        if state.radio.channel[15] < 0 and hasattr(policy, 'init_hidden_state'):
            print("(TOGGLE SWITCH) RESETTING HIDDEN STATES TO ZERO!")
            policy.init_hidden_state()

        # Switch the operation mode based on the toggle next to STO
        if state.radio.channel[9] < -0.5: # towards operator means damping shutdown mode
            operation_mode = 2
        elif state.radio.channel[9] > 0.5: # away from the operator means reset states
          operation_mode = 1
          standing_height = MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT)*0.5*(state.radio.channel[6] + 1)
        else:                               # Middle means normal walking
          operation_mode = 0

        curr_max = max_speed / 2
        speed_add = (max_speed / 2) * state.radio.channel[4]
        speed = max(min_speed, state.radio.channel[0] * curr_max + speed_add)
        speed = min(max_speed, state.radio.channel[0] * curr_max + speed_add)

        print('\tCH5: ' + str(state.radio.channel[5]))
        phase_add = 1 # + state.radio.channel[5]
      else:
        # Automatically change orientation and speed
        tt = time.monotonic() - t0

        if check_stdin():
          c = sys.stdin.read(1)
          if c == 'w':
            speed += 0.1
          if c == 's':
            speed -= 0.1
          if c == 'a':
            orient_add -= 0.1
          if c == 'd':
            orient_add += 0.1
          if c == 'r':
            speed = 0.5
            orient_add = 0


        speed = max(min_speed, speed)
        speed = min(max_speed, speed)

      #------------------------------- Normal Walking ---------------------------
      if operation_mode == 0:
          #print("speed: {:3.2f} | orientation {:3.2f}".format(speed, orient_add), end='\r')
          print("\tspeed: {:3.2f} | orientation {:3.2f}".format(speed, orient_add))
          
          # Reassign because it might have been changed by the damping mode
          for i in range(5):
              u.leftLeg.motorPd.pGain[i] = env.P[i]
              u.leftLeg.motorPd.dGain[i] = env.D[i]
              u.rightLeg.motorPd.pGain[i] = env.P[i]
              u.rightLeg.motorPd.dGain[i] = env.D[i]

          clock = [np.sin(2 * np.pi *  phase / 27), np.cos(2 * np.pi *  phase / 27)]
          quaternion = euler2quat(z=orient_add, y=0, x=0)
          iquaternion = inverse_quaternion(quaternion)
          new_orient = quaternion_product(iquaternion, state.pelvis.orientation[:])
          if new_orient[0] < 0:
              new_orient = -new_orient
          new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
              
          ext_state = np.concatenate((clock, [speed]))
          robot_state = np.concatenate([
                  [state.pelvis.position[2] - state.terrain.height], # pelvis height
                  new_orient,                                     # pelvis orientation
                  state.motor.position[:],                        # actuated joint positions

                  new_translationalVelocity[:],                   # pelvis translational velocity
                  state.pelvis.rotationalVelocity[:],             # pelvis rotational velocity 
                  state.motor.velocity[:],                        # actuated joint velocities

                  state.pelvis.translationalAcceleration[:],      # pelvis translational acceleration
                  
                  state.joint.position[:],                        # unactuated joint positions
                  state.joint.velocity[:]                         # unactuated joint velocities
          ])
          RL_state = np.concatenate([robot_state, ext_state])
          
          #pretending the height is always 1.0
          #RL_state[0] = 1.0
          
          # Construct input vector
          torch_state = torch.Tensor(RL_state)
          torch_state = policy.normalize_state(torch_state, update=False)

          if no_delta:
            offset = env.offset
          else:
            offset = env.get_ref_state(phase=phase)

          action = policy(torch_state)
          env_action = action.data.numpy()
          target = env_action + offset

          # Send action
          for i in range(5):
              u.leftLeg.motorPd.pTarget[i] = target[i]
              u.rightLeg.motorPd.pTarget[i] = target[i+5]
          cassie.send_pd(u)

          # Logging
          if sto == False:
              time_log.append(time.time())
              state_log.append(state)
              input_log.append(RL_state)
              output_log.append(env_action)
              target_log.append(target)
      #------------------------------- Start Up Standing ---------------------------
      elif operation_mode == 1:
          print('Startup Standing. Height = ' + str(standing_height))
          #Do nothing
          # Reassign with new multiplier on damping
          for i in range(5):
              u.leftLeg.motorPd.pGain[i] = 0.0
              u.leftLeg.motorPd.dGain[i] = 0.0
              u.rightLeg.motorPd.pGain[i] = 0.0
              u.rightLeg.motorPd.dGain[i] = 0.0

          # Send action
          for i in range(5):
              u.leftLeg.motorPd.pTarget[i] = 0.0
              u.rightLeg.motorPd.pTarget[i] = 0.0
          cassie.send_pd(u)

      #------------------------------- Shutdown Damping ---------------------------
      elif operation_mode == 2:

          print('Shutdown Damping. Multiplier = ' + str(D_mult))
          # Reassign with new multiplier on damping
          for i in range(5):
              u.leftLeg.motorPd.pGain[i] = 0.0
              u.leftLeg.motorPd.dGain[i] = D_mult*env.D[i]
              u.rightLeg.motorPd.pGain[i] = 0.0
              u.rightLeg.motorPd.dGain[i] = D_mult*env.D[i]

          # Send action
          for i in range(5):
              u.leftLeg.motorPd.pTarget[i] = 0.0
              u.rightLeg.motorPd.pTarget[i] = 0.0
          cassie.send_pd(u)

      #---------------------------- Other, should not happen -----------------------
      else:
          print('Error, In bad operation_mode with value: ' + str(operation_mode))
      
      # Measure delay
      # Wait until next cycle time
      while time.monotonic() - t < 60/2000:
          time.sleep(0.001)
      print('\tdelay: {:6.1f} ms'.format((time.monotonic() - t) * 1000))

      # Track phase
      phase += phase_add
      if phase >= 28:
          phase = 0
          counter += 1
  finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
