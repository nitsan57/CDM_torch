from os import POSIX_FADV_NOREUSE
import numpy as np
import torch

def calc_returns(rewards, dones, discount_factor):
  """works with rewards vector which consitst of many epidsodes"""
  curr = 0
  device = rewards.device
  returns = torch.zeros_like(rewards, device=device)
  num_rewards = len(rewards)
  for i,r in enumerate(torch.flip(rewards, dims=(0,))):
    reveres_idx = num_rewards - i -1
    if dones[reveres_idx]:
      curr = 0
    curr = r + discount_factor * curr
    returns[reveres_idx] = curr # append in reverse
  return returns


def calc_gaes(rewards, values, dones, discount_factor=0.99, decay=0.95):
    """
    works with rewards vector which consitst of many epidsodes
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    gamma = discount_factor
    device = rewards.device

    next_values = torch.cat([values[1:], torch.zeros(1, device=device)])
    # deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]
    deltas = rewards + gamma * next_values * (1 - dones) - values

    gaes = torch.zeros_like(deltas, device=device)
    gaes[-1] = deltas[-1]

    for i in reversed(range(len(deltas)-1)):

      gaes[i] = deltas[i] + gamma * decay * gaes[i+1] * (1 - dones[i])

    return gaes



class ExperienceReplay:
#TODO: support continous infinte env by removeing long seeing states without dones
    def __init__(self, capacity, obs_shape, batch_size=64, continous_mem = False):
      self.batch_size = batch_size
      self.obs_shape = obs_shape
      self.capacity = capacity
      self.init_buffers()
      self.continous_mem = continous_mem
      # self.continous_idx = 0
      #TODO: np.unique(self.buffers,axis=0) remove recurrent states


    def __len__(self):
        return self.curr_size


    def append(self, curr_obs, action, reward, done, next_obs):
        num_samples = len(curr_obs)

        free_space = self.capacity - self.curr_size
        if num_samples > self.capacity:
          self.curr_size = 0
          curr_obs = curr_obs[:self.capacity]
          action = action[:self.capacity]
          reward = reward[:self.capacity]
          done = done[:self.capacity]
          next_obs = next_obs[:self.capacity]
          num_samples = self.capacity
          done[-1] = True


        elif self.__len__() + num_samples > self.capacity and not self.continous_mem:         
            dones = np.where(self.all_buffers[self.dones_index]==True)[0]
            relevant_index = np.where(dones > num_samples - free_space)[0][0]
            done_index = dones[relevant_index]


            for i in range(len(self.all_buffers)):
                self.all_buffers[i][:done_index+1] = 0
                self.all_buffers[i] = np.roll(self.all_buffers[i], -done_index -1, axis=0)
           
            self.curr_size -= (done_index+1)


        self.all_buffers[self.states_index][self.curr_size:self.curr_size+num_samples] = curr_obs
        self.all_buffers[self.actions_index][self.curr_size:self.curr_size+num_samples] = action
        self.all_buffers[self.reward_index][self.curr_size:self.curr_size+num_samples] = reward
        self.all_buffers[self.dones_index][self.curr_size:self.curr_size+num_samples] = done
        self.all_buffers[self.next_states_index][self.curr_size:self.curr_size+num_samples] = next_obs

        self.curr_size +=num_samples
        

    def init_buffers(self):
        
        self.curr_size = 0
        states_buffer = np.zeros((self.capacity, *self.obs_shape),dtype=np.float32)
        actions_buffer = np.zeros((self.capacity),dtype=np.int64)
        reward_buffer = np.zeros((self.capacity),dtype=np.float32)
        dones_buffer = np.zeros((self.capacity), dtype=np.uint8)
        next_states_buffer = np.zeros((self.capacity, *self.obs_shape),dtype=np.float32)
        self.all_buffers = [states_buffer, actions_buffer, reward_buffer, dones_buffer, next_states_buffer]
        self.states_index = 0
        self.actions_index = 1
        self.reward_index = 2
        self.dones_index = 3
        self.next_states_index = 4


    def clear(self):
        self.init_buffers()


    def get_last_episodes(self, num_episodes, orderd=False):
        """return all last episode samples, or specified num samples"""
        episode_indices = [0]
        episode_indices.extend(np.where(self.all_buffers[self.dones_index]==True)[0])

        
        assert len(episode_indices) >= num_episodes , "requested more episodes then actual stored in mem"
        num_samples = self.curr_size - episode_indices[-num_episodes -1] -1 #exclude first done indice

        return self.get_last_samples(num_samples, orderd)


    def get_last_samples(self, num_samples=None, orderd=False):
        """return all last episode samples, or specified num samples"""
        if num_samples is None:
            "return last episode"
            dones = np.where(self.all_buffers[self.dones_index]==True)[0]
            if len(dones) > 1:
                last_done = dones[-2] +1 #exclude the latest done sample
            else:
                last_done = 0 # from 0 index to last done(which is also the first..)

            if orderd:
                last_samples = [buff[last_done+1:self.curr_size] for buff in self.all_buffers]
            else:
                random_indices = np.random.choice(list(range(last_done+1, self.curr_size, 1)),self.curr_size-last_done-1, replace=False)
                last_samples = [buff[random_indices] for buff in self.all_buffers]
        else:
            if orderd:
                last_samples = [buff[self.curr_size-num_samples:self.curr_size] for buff in self.all_buffers]
            else:
                random_indices = np.random.choice(list(range(self.curr_size - num_samples, self.curr_size, 1)),num_samples, replace=False)
                last_samples = [buff[random_indices] for buff in self.all_buffers]
        return last_samples
        

    def get_buffers_at(self, indices):
        buffers_at = (buff[indices] for buff in self.all_buffers)
        return buffers_at
        

    def sample_random_batch(self, sample_size):
      indices = np.random.choice(self.curr_size, sample_size,replace=False)
      return self.get_buffers_at(indices)


# https://www.tutorialspoint.com/multiprocessing-in-python

from multiprocessing import Process, Pipe
from collections import namedtuple
from time import sleep
import numpy as np

def worker(env, conn, idx):
  proc_running = True
  env.reset()

  while proc_running:
    cmd, msg = conn.recv()

    if (cmd == "step"):
      next_state, reward, done, _ = env.step(msg)
      if done:
        next_state = env.reset()
      conn.send((next_state, reward, done, _))

    elif (cmd == "reset"):
      next_state = env.reset() 
      conn.send(next_state) 

    elif (cmd == "close"):
      proc_running = False
      conn.close()

    else:
      raise Exception("Command not implemented")

class ParallelEnv:
  def __init__(self, env, num_envs):

    self.num_envs = num_envs
    self.process = namedtuple("Process", field_names=["proc", "connection"])
    self.env = env
    self.comm = []
    for idx in range(self.num_envs):
        parent_conn, worker_conn = Pipe()
        proc = Process(target=worker, args=(self.env, worker_conn, idx))
        proc.start()
        self.comm.append(self.process(proc, parent_conn))

  def reset(self):
    [ p.connection.send(("reset", "")) for p in self.comm] 
    res = [ p.connection.recv() for p in self.comm]
    return res

  def step(self, actions):    
    # send actions to envs
    [ p.connection.send(("step", action)) for i, p, action in zip(range(self.num_envs),self.comm, actions)]
    
    
    # Receive response from envs.
    res = [ p.connection.recv() for p in self.comm]
    next_states, rewards, dones, _ = zip(*res)
    rewards = np.array(rewards)
    dones = np.array(dones)
    
    return next_states, rewards, dones, np.array(_)

  def close_procs(self):
    [ p.connection.send(("close", "")) for p in self.comm]