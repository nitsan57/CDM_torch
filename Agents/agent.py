from numpy.random.mtrand import random_sample
from tqdm import tqdm
from abc import ABC, abstractmethod
from .agent_utils import ExperienceReplay
import numpy as np
import torch 

class RL_Agent(ABC):
    TRAIN=0
    EVAL = 1
    def __init__(self, obs_shape,max_mem_size, batch_size, device='cpu') -> None:
        super().__init__()
        self.act_mode = self.TRAIN
        self.experience = ExperienceReplay(max_mem_size, obs_shape, full_episodes=False)
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.device = device
        self.curr_collect_step = 0
        self.collect_before_update = self.batch_size // 4


    # @abstractmethod
    def save_agent(self,):
        raise NotImplementedError


    # @abstractmethod
    def load_agent(self,):
        raise NotImplementedError


    def set_train_mode(self):
        self.act_mode = self.TRAIN


    def set_eval_mode(self):
        self.act_mode = self.EVAL


    def train_episodial(self, env, n_episodes):
        self.set_train_mode()
        pbar = tqdm(range(n_episodes), leave=False)
        curr_training_steps = 0
        for i in pbar:
            self.curr_collect_step = len(self.experience)
            reward_vector = self.collect_episode_obs(env)
            num_steps_collected = len(reward_vector)
            curr_training_steps +=num_steps_collected
            
            desciption = f"episode {i}, R:{np.sum(reward_vector)}, total_steps:{curr_training_steps}"
            pbar.set_description(desciption)

            self.curr_collect_step += len(reward_vector) # how many steps where collected before update policy
            if len(self.experience) < self.batch_size or self.curr_collect_step < self.collect_before_update:
                #collect more sampels if not enough..
                continue
            self.curr_collect_step  = 0
            observations, actions, rewards,dones, next_observations = self.get_experiences(random_samples=True)
            self.update_policy(observations, actions, rewards, dones, next_observations)

        
    def train_n_steps(self, env, n_steps, max_episode_len=None):
        self.set_train_mode()
        curr_training_steps = 0
        all_episodes_rewards = []
        pbar = tqdm(total=n_steps, leave=False)
        episode = 0
        while curr_training_steps < n_steps:
            self.curr_collect_step = len(self.experience)
            reward_vector = self.collect_episode_obs(env, max_episode_len)
            episode +=1
            num_steps_collected = len(reward_vector)
            # all_episodes_rewards.append(np.sum(reward_vector))
            curr_training_steps +=num_steps_collected
            pbar.update(num_steps_collected)
            desciption = f"episode {episode}, R:{np.sum(reward_vector)}, total_steps:{curr_training_steps}"
            pbar.set_description(desciption)

            self.curr_collect_step += len(reward_vector) # how many steps where collected before update policy
            if len(self.experience) < self.batch_size or self.curr_collect_step < self.collect_before_update:
                #collect more samples if not enough..
                continue
            self.curr_collect_step  = 0
            observations, actions, rewards, dones, next_observations = self.get_experiences(random_samples=True)
            self.update_policy(observations, actions, rewards, dones, next_observations)
        pbar.close()


    def collect_episode_obs(self, env, max_episode_len = None):
        done = False
        if max_episode_len:
            episode_len_exceeded = lambda x: x > max_episode_len
        else:
            episode_len_exceeded = lambda x: False

        observation = env.reset()
        episode_steps = 0 
        episode_rewards = []
        while not done:
            action = self.act(observation)
            next_obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            episode_steps +=1
            if episode_len_exceeded(episode_steps):
                # add done=True is episode len exceeds
                done = True
                self.experience.append(observation, action, reward, done, next_obs)
                break
            self.experience.append(observation, action, reward, done, next_obs)

        return episode_rewards


    def get_experiences(self, random_samples=True):
        if random_samples:
            latest_experience_batch = self.experience.get_last_samples(self.collect_before_update)
            random_experience_batch = self.experience.sample_random_batch(self.batch_size - self.collect_before_update)
            observations, actions, rewards, dones, next_observations = random_experience_batch
            latest_observations, latest_actions, latest_rewards, latest_dones, latest_next_observations = latest_experience_batch
            observations = np.concatenate([observations, latest_observations])
            actions = np.concatenate([actions, latest_actions])
            rewards = np.concatenate([rewards, latest_rewards])
            dones = np.concatenate([dones, latest_dones])
            next_observations = np.concatenate([next_observations, latest_next_observations])
        else:
            observations, actions, rewards, dones, next_observations = self.experience.sample_orderd_episodes_batch(self.batch_size)

        observations = torch.tensor(observations).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_observations = torch.tensor(next_observations).to(self.device)
        return observations, actions, rewards, dones, next_observations


    @abstractmethod
    def update_policy(self, observations, actions, rewards, dones, next_observations):
        """Get batched samples and update policy"""
        raise NotImplementedError


    @abstractmethod
    def act(self):
        raise NotImplementedError
