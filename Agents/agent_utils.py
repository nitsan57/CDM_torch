
import numpy as np
import torch 

def calc_reward_to_go(rewards, discount_factor):
    curr = 0
    device = rewards.device
    rewards_to_go = torch.zeros_like(rewards, device=device)
    num_rewards = len(rewards)
    for i,r in enumerate(torch.flip(rewards, dims=(1,))):
        curr = r + discount_factor * curr
        rewards_to_go[num_rewards- i -1] = r # append in reverse

    return rewards_to_go

class ExperienceReplay:

    def __init__(self, capacity, obs_shape, full_episodes=False):
        self.obs_shape = obs_shape
        self.capacity = capacity
        self.init_buffers()
        self.episodes_indices = []


    def __len__(self):
        return self.curr_size


    def append(self, curr_obs, action, reward, done, next_obs):
        if len(self.states_buffer) > self.capacity:
            first_done = np.where(self.dones_buffer==True)[0][0] +1
            self.states_buffer[first_done:] = 0
            self.actions_buffer[first_done:] = 0
            self.reward_buffer[first_done:] = 0
            self.dones_buffer[first_done:] = 0
            self.next_states_buffer[first_done:] = 0
            self.curr_size -= first_done+1


        self.states_buffer[self.curr_size:self.curr_size+1] = curr_obs
        self.actions_buffer[self.curr_size:self.curr_size+1] = action
        self.reward_buffer[self.curr_size:self.curr_size+1] = reward
        self.dones_buffer[self.curr_size:self.curr_size+1] = done
        self.next_states_buffer[self.curr_size:self.curr_size+1] = next_obs
        self.curr_size +=1
        

    def init_buffers(self):
        self.curr_size = 0
        self.states_buffer = np.zeros((self.capacity, *self.obs_shape),dtype=np.float32)
        self.actions_buffer = np.zeros((self.capacity),dtype=np.int64)
        self.reward_buffer = np.zeros((self.capacity),dtype=np.float32)
        self.dones_buffer = np.zeros((self.capacity), dtype=np.uint8)
        self.next_states_buffer = np.zeros((self.capacity, *self.obs_shape),dtype=np.float32)


    def clear(self):
        self.init_buffers()


    def get_last_samples(self, num_samples):
        states = self.states_buffer[-num_samples:]
        actions = self.actions_buffer[-num_samples:]
        rewards = self.reward_buffer[-num_samples:]
        dones = self.dones_buffer[-num_samples:]
        next_states = self.next_states_buffer[-num_samples:]
        return states, actions, rewards, dones, next_states
        

    def get_buffers_at(self, indices):
        states = self.states_buffer[indices]
        actions = self.actions_buffer[indices]
        rewards = self.reward_buffer[indices]
        dones = self.dones_buffer[indices]
        next_states = self.next_states_buffer[indices]
        return states, actions, rewards, dones, next_states


    def sample_orderd_episodes_batch(self, max_batch_size=None):
        """
        return orderd experiance full episodes samples - good for some RL algorithms
        max_batch_size = maximum samples to return, if there is a longer episode the sample size will exceed max batch size
        """
        max_batch_size = len(self.states_buffer) if max_batch_size is None else max_batch_size
        indices = -1
        dones = np.array(self.dones_buffer)
        dones_indices = np.where(dones==True)
        if dones_indices[0] > max_batch_size:
            indices = range(dones_indices[0])
        else:
            for j,idx in enumerate(dones_indices):
                if idx >= max_batch_size:
                    indices = range(dones_indices[j-1])
                    break

        return self.get_buffers_at(indices)


    def sample_random_batch(self, batch_size=32):
      indices = np.random.choice(len(self.states_buffer), batch_size,replace=False)
      return self.get_buffers_at(indices)