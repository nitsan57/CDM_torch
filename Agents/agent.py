from numpy.core.numeric import indices
from torch.autograd.grad_mode import F
from tqdm import tqdm
from abc import ABC, abstractmethod
from .agent_utils import ExperienceReplay
import numpy as np
import torch 
import functools
import operator
from .agent_utils import ParallelEnv

class RL_Agent(ABC):
    TRAIN=0
    EVAL = 1
    def __init__(self, obs_shape, max_mem_size, batch_size, num_parallel_envs=None,rnn=False, device='cpu') -> None:
        """if num_parallel_envs is none it will use batch size"""
        super().__init__()
        self.rand_perm = True
        self.update_policy = self.update_policy_reg
        self.rnn = rnn
        if rnn:
            self.rand_perm = False
            self.update_policy = self.update_policy_rnn
        self.num_parallel_envs = batch_size if num_parallel_envs is None else num_parallel_envs
        assert self.num_parallel_envs <= batch_size, f"please provide batch_size>= num_parallel_envs current: {batch_size}, {num_parallel_envs},"

        self.eval_mode = self.EVAL
        self.experience = ExperienceReplay(max_mem_size, obs_shape)
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.device = device
        self.curr_collect_step = 0
        #if return log probs, and save original user choice, since in eval mode we do not return log probs


    # @abstractmethod
    def save_agent(self,):
        raise NotImplementedError


    # @abstractmethod
    def load_agent(self,):
        raise NotImplementedError


    def set_train_mode(self):
        self.eval_mode = self.TRAIN


    def set_eval_mode(self):
        self.eval_mode = self.EVAL

    def train_episodial(self, env, n_episodes):
        return self._train_n_iters(env, n_episodes, True)

    def train_n_steps(self, env, n_steps, max_episode_len=None):
        return self._train_n_iters(env, n_steps, episodes=False, max_episode_len=max_episode_len)


    def _train_n_iters(self, env, n_iters, episodes=False,max_episode_len=None):
        """General train function, if episodes is true- each iter is episode, otherwise train steps"""
        self.set_train_mode()
        env = ParallelEnv(env, self.num_parallel_envs)
        pbar = tqdm(total=n_iters, leave=False)
        curr_training_steps = 0
        train_rewards = []
        self.curr_collect_step = len(self.experience)
        if episodes:
            to_update_idx = 0
        else:
            to_update_idx = 1
        
        i = 0 
        ep_number = 0
        while i < n_iters:
            rewards_vector = self.collect_episode_obs_parallel(env, max_episode_len) #self.collect_episode_obs(env, max_episode_len)
            num_steps_collected = 0
            for r in rewards_vector:
                train_rewards.append(np.sum(r))
                num_steps_collected += len(r)
            
            collect_info = [self.num_parallel_envs, num_steps_collected]
            curr_training_steps +=num_steps_collected
            
            desciption = f"episode {ep_number}, R:{np.round(np.mean(train_rewards[-self.batch_size:]), 2):08}, total_steps:{curr_training_steps}"
            pbar.set_description(desciption)

            pbar.update(collect_info[to_update_idx])
            i +=collect_info[to_update_idx]
            ep_number +=self.num_parallel_envs


            self.curr_collect_step += num_steps_collected # how many steps where collected before update policy
            if len(self.experience) < self.batch_size:
                #collect more sampels if not enough..
                continue
            self.curr_collect_step  = 0
            
            self.update_policy()
            self.reset_rnn_hidden()
        env.close_procs()
        pbar.close()
        return train_rewards


    def get_experiences(self, random_samples):
        """Get a mix of samples, including all last episode- makes sure we dont miss any seen states"""
        if random_samples:
            latest_experience_batch = self.experience.get_last_episodes(self.num_parallel_envs)
            last_episode_len = len(latest_experience_batch[0])
            #50% last episodes, 50% random
            random_experience_batch = self.experience.sample_random_batch(last_episode_len)
            observations, actions, rewards, dones, next_observations = random_experience_batch
            latest_observations, latest_actions, latest_rewards, latest_dones, latest_next_observations = latest_experience_batch
            rand_perm = torch.randperm(len(observations))
            observations = np.concatenate([observations, latest_observations])[rand_perm]
            actions = np.concatenate([actions, latest_actions])[rand_perm]
            rewards = np.concatenate([rewards, latest_rewards])[rand_perm]
            dones = np.concatenate([dones, latest_dones])[rand_perm]
            next_observations = np.concatenate([next_observations, latest_next_observations])[rand_perm]
            
        else:
            observations, actions, rewards, dones, next_observations = self.experience.get_last_episodes(self.num_parallel_envs, True)

        
        observations = torch.tensor(observations).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        next_observations = torch.tensor(next_observations).to(self.device)
        return observations, actions, rewards, dones, next_observations

    # def get_rnn_exp(self, random_samples):
    #     observations, actions, rewards, dones, next_observations = self.get_experiences(False)
    #     env_indices = np.zeros(self.num_parallel_envs, dtype=np.int32)
    #     done_indices = torch.where(dones == True)[0].cpu().numpy().astype(np.int32)
    #     env_indices[1:] = done_indices[:-1]
        # seq_lens = np.sort(done_indices - env_indices)[::-1].copy()
        # import pdb
        # pdb.set_trace()
        # unpadded_input = [observations[curr_idx:next_idx] for curr_idx, next_idx in zip(env_indices, done_indices)]
        # unpadded_input.sort(reverse=True,key=lambda x: x.size())


        # padded_seq_batch = torch.nn.utils.rnn.pad_sequence(unpadded_input, batch_first=True)

        # packed_seq_batch = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True)
        # packed_seq_batch.batch_sizes

        # return packed_seq_batch


        # env_indices[1:] = done_indices[:-1]
        # alive_indices = np.ones_like(env_indices, dtype=bool)
        
        # progress_indices = env_indices.T
        # relevant_indices = progress_indices
        # while any(alive_indices):
        #     yield observations[relevant_indices], actions[relevant_indices], rewards[relevant_indices], dones[relevant_indices], next_observations[relevant_indices]
        #     progress_indices +=1

        #     alive_indices = (1 - (progress_indices > done_indices)).astype(bool)
        #     relevant_indices = progress_indices[alive_indices]
            # progress_indices = progress_indices[alive_indices]
            # pdb.set_trace()       
    
    def get_rnn_exp(self, random_samples):
        observations, actions, rewards, dones, next_observations = self.get_experiences(False)
        env_indices = np.zeros(self.num_parallel_envs, dtype=np.int32)
        done_indices = torch.where(dones == True)[0].cpu().numpy()
        env_indices[1:] = done_indices[:-1]
        alive_indices = np.ones_like(env_indices, dtype=bool)
        last_lived = alive_indices
        
        progress_indices = env_indices.T
        relevant_indices = progress_indices
        while any(alive_indices):
            curr_obs = observations[relevant_indices]
            next_obs = next_observations[relevant_indices]
     
            yield torch.unsqueeze(curr_obs, 1), actions[relevant_indices], rewards[relevant_indices], dones[relevant_indices], torch.unsqueeze(next_obs, 1)
            progress_indices +=1

            last_lived = alive_indices
            alive_indices = (1 - (progress_indices > done_indices)).astype(bool)
            relevant_indices = progress_indices[alive_indices]
            rnn_mask = [item for i,item in enumerate(alive_indices) if last_lived[i] ]

            self.update_rnn_indices(rnn_mask)

    @abstractmethod
    def update_policy_reg(self):
        """Get batched samples and update policy"""
        raise NotImplementedError

    @abstractmethod
    def update_policy_rnn(self):
        """Get batched samples and update policy"""
        raise NotImplementedError

    @abstractmethod
    def act(self, observations, num_obs=1):
        raise NotImplementedError

    def return_correct_actions_dim(self, selected_actions, num_obs):
        if num_obs == 1 and self.eval_mode:
            """non batch case, probably eval mode action"""
            selected_actions = selected_actions[0]
        return selected_actions


    def collect_episode_obs_parallel(self, env, max_episode_len = None):
        parallel_envs = env

        if max_episode_len:
            episode_len_exceeded = lambda x: x > max_episode_len
        else:
            episode_len_exceeded = lambda x: False

        observations = [[item] for item in parallel_envs.reset()]
        env_dones = np.array([False for i in range(self.num_parallel_envs)])

        latest_observations = np.array(([observations[i][-1] for i in range(self.num_parallel_envs)]))

        rewards = [[] for i in range(self.num_parallel_envs)]
        actions = [[] for i in range(self.num_parallel_envs)]
        next_observations = [[] for i in range(self.num_parallel_envs)]
        dones = [[] for i in range(self.num_parallel_envs)]

        max_episode_steps = 0 

        while not all(env_dones):
            relevant_indices = np.where(env_dones == False)[0].astype(np.int32)
            if self.rnn:
                # import pdb
                ## only for rnn alone no pack:# 
                latest_observations = latest_observations.reshape(self.num_parallel_envs, 1, *latest_observations[-1].shape)
                ## latest_observations = [torch.from_numpy(x) for x in latest_observations]
                # seq_lens = np.ones(self.num_parallel_envs)
                # padded_seq_batch = torch.nn.utils.rnn.pad_sequence(latest_observations, batch_first=True)
                # padded_seq_batch= padded_seq_batch.reshape((self.num_parallel_envs, 1, *latest_observations[-1].shape))
                # latest_observations = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True)


            current_actions = self.act(latest_observations, self.num_parallel_envs)
            #TODO DEBUG 
            # allways use all envs to step, even some envs are done already
            next_obs, reward, done, info = parallel_envs.step(current_actions)
            for i in relevant_indices:
                actions[i].append(current_actions[i])
                next_observations[i].append(next_obs[i])
                rewards[i].append(reward[i])
                dones[i].append(done[i])
                env_dones[i] = done[i]

                max_episode_steps +=1
                if done[i]:
                    continue

                if episode_len_exceeded(max_episode_steps):
                    done = True
                    dones[i][-1] = done
                    env_dones[i] = done
                    continue

                observations[i].append(next_obs[i])
            latest_observations = np.array(([observations[i][-1] for i in range(self.num_parallel_envs)]))
        if self.rnn:
            self.reset_rnn_hidden()
        observations = functools.reduce(operator.iconcat, observations, [])
        actions = functools.reduce(operator.iconcat, actions, [])
        rewards_x = functools.reduce(operator.iconcat, rewards, [])
        dones = functools.reduce(operator.iconcat, dones, [])
        next_observations = functools.reduce(operator.iconcat, next_observations, [])
        self.experience.append(observations, actions, rewards_x, dones, next_observations)
        return rewards

    @abstractmethod
    def reset_rnn_hidden(self,):
        """if agent uses rnn, this callback is called in many places so please impliment it"""
        raise NotImplementedError

    @abstractmethod
    def update_rnn_indices(self,indices):
        """if agent uses rnn, this callback is called in many places so please impliment it"""
        raise NotImplementedError

    # def collect_episode_obs(self, env, max_episode_len = None):
    #     done = False
    #     if max_episode_len:
    #         episode_len_exceeded = lambda x: x > max_episode_len
    #     else:
    #         episode_len_exceeded = lambda x: False

    #     observation = env.reset()
    #     episode_steps = 0 
    #     train_rewards = [[]]
    #     while not done:
    #         action = self.act(observation[np.newaxis, :])
    #         next_obs, reward, done, info = env.step(action)
    #         train_rewards[0].append(reward)
    #         if episode_len_exceeded(episode_steps):
    #             # add done=True is episode len exceeds
    #             done = True
    #             self.experience.append(np.expand_dims(observation, axis=0), action, np.expand_dims(np.array(reward), axis=0), np.expand_dims(np.array(done), axis=0), next_obs[np.newaxis, :])
    #             break
    #         self.experience.append(np.expand_dims(observation, axis=0), action, np.expand_dims(np.array(reward), axis=0), np.expand_dims(np.array(done), axis=0), next_obs[np.newaxis, :])
    #         episode_steps +=1
    #         observation = next_obs


    #     return train_rewards

    # def _train_n_iters(self, env, n_iters, episodes=False,max_episode_len=None):
    #     """General train function, if episodes is true- each iter is episode, otherwise train steps"""
    #     self.set_train_mode()
    #     pbar = tqdm(total=n_iters, leave=False)
    #     curr_training_steps = 0
    #     train_rewards = []
    #     self.curr_collect_step = len(self.experience)
    #     if episodes:
    #         to_update_idx = 0
    #     else:
    #         to_update_idx = 1
        
    #     p = self.batch_size

    #     for i in range(1,n_iters+1,1):

    #         rewards_vector = self.collect_episode_obs(env, max_episode_len) #self.collect_episode_obs(env, max_episode_len) #self.collect_episode_obs_parallel(env, max_episode_len) #self.collect_episode_obs(env, max_episode_len)
    #         num_episodes_collected = len(rewards_vector)
    #         num_steps_collected = 0
    #         for r in rewards_vector:
    #             train_rewards.append(np.sum(r)//num_episodes_collected)

    #             num_steps_collected += np.sum(len(r))
    #         collect_info = [num_episodes_collected, num_steps_collected]
    #         curr_training_steps +=num_steps_collected
            
    #         desciption = f"episode {i}, R:{np.round(train_rewards[-1], 2):08}, total_steps:{curr_training_steps}"
    #         pbar.set_description(desciption)

    #         pbar.update(collect_info[to_update_idx])

    #         self.curr_collect_step += num_steps_collected # how many steps where collected before update policy
    #         if len(self.experience) < self.batch_size:
    #             #collect more sampels if not enough..
    #             continue
    #         self.curr_collect_step  = 0
    #         observations, actions, rewards,dones, next_observations = self.get_experiences()
    #         self.update_policy(observations, actions, rewards, dones, next_observations)
    #     pbar.close()

    #     return train_rewards

    # def get_experiences(self):
    #     """Get a mix of samples, including all last episode- makes sure we dont miss any seen states"""
    #     if self.random_experience_samples:
    #         latest_experience_batch = self.experience.get_last_samples()
    #         last_episode_len = len(latest_experience_batch[0])
    #         #50% last episode, 50% random
    #         random_experience_batch = self.experience.sample_random_batch(last_episode_len)
    #         observations, actions, rewards, dones, next_observations = random_experience_batch
    #         latest_observations, latest_actions, latest_rewards, latest_dones, latest_next_observations = latest_experience_batch
    #         observations = np.concatenate([observations, latest_observations])
    #         actions = np.concatenate([actions, latest_actions])
    #         rewards = np.concatenate([rewards, latest_rewards])
    #         dones = np.concatenate([dones, latest_dones])
    #         next_observations = np.concatenate([next_observations, latest_next_observations])
            
    #     else:
    #         observations, actions, rewards, dones, next_observations = self.experience.get_last_samples(True, self.batch_size)

    #     observations = torch.tensor(observations).to(self.device)
    #     actions = torch.tensor(actions).to(self.device)
    #     rewards = torch.tensor(rewards).to(self.device)
    #     dones = torch.tensor(dones).to(self.device)
    #     next_observations = torch.tensor(next_observations).to(self.device)

    #     return observations, actions, rewards, dones, next_observations
