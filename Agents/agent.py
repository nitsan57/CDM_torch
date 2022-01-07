from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from abc import ABC, abstractmethod
from .agent_utils import ExperienceReplay, ObsShapeWraper, ObsWraper
import numpy as np
import functools
import operator
from .agent_utils import ParallelEnv
import torch
from functools import wraps
from time import time
import torch.nn.functional as F
from torch.distributions import Categorical


class RL_Agent(ABC):
    TRAIN = 0
    EVAL = 1

    def __init__(self, obs_shape, n_actions, max_mem_size, batch_size, num_parallel_envs=None, model=None, lr=0.0001, device='cpu') -> None:
        """if num_parallel_envs is none it will use batch size"""
        super().__init__()
        self.rand_perm = True
        self.update_policy = self.update_policy_reg
        self.model = model
        self.rnn = model.is_rnn
        if self.rnn:
            self.update_policy = self.update_policy_rnn
        self.num_parallel_envs = batch_size if num_parallel_envs is None else num_parallel_envs
        assert self.num_parallel_envs <= batch_size, f"please provide batch_size>= num_parallel_envs current: {batch_size}, {num_parallel_envs},"

        self.eval_mode = self.EVAL
        self.obs_shape = ObsShapeWraper(obs_shape)
        self.experience = ExperienceReplay(max_mem_size, self.obs_shape)
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.device = device
        self.n_actions = n_actions
        self.lr = lr
        self.env = None
        self.store_entropy = False
        # self.init_entropy_buffer()
        # if return log probs, and save original user choice, since in eval mode we do not return log probs


    def calc_entropy_from_vec(self, batched_vector):
        logits = F.log_softmax(batched_vector,1)
        return self.calc_entropy_from_logits(logits)


    def calc_entropy_from_logits(self, logits):
        entropy = Categorical(logits= logits).entropy().detach().cpu().numpy()
        return entropy


    def get_stored_entropy(self,):
        return self.stored_entropy


    def init_entropy_buffer(self):
        if self.store_entropy:
            self.stored_entropy = [[] for i in range(self.num_parallel_envs)]


    def clear_stored_entropy(self):
        self.init_entropy_buffer()


    def set_store_entropy(self, val : bool):
        self.store_entropy = val
        if self.store_entropy:
            self.init_entropy_buffer()


    @abstractmethod
    def init_models(self):
        raise NotImplementedError


    def add_to_obs_shape(self, data_dict):
        for k, v in data_dict.items():
            self.obs_shape[k] = v
        self.init_models()
        self.experience = ExperienceReplay(self.mem_size, self.obs_shape)

    @abstractmethod
    def save_agent(self,):
        raise NotImplementedError


    @abstractmethod
    def load_agent(self, f_name):
        raise NotImplementedError

    def set_train_mode(self):
        self.reset_rnn_hidden()
        self.eval_mode = self.TRAIN

    def set_eval_mode(self):
        self.reset_rnn_hidden()
        self.eval_mode = self.EVAL
        self.set_store_entropy(False)

    def train_episodial(self, env, n_episodes, max_episode_len=None, disable_tqdm=False, additional_const_features={}):
        old_parallel_envs = self.num_parallel_envs
        if n_episodes < self.num_parallel_envs:
            self.set_num_parallel_env(n_episodes)
        train_r = self._train_n_iters(env, n_episodes, True, max_episode_len=max_episode_len, disable_tqdm=disable_tqdm, additional_const_features=additional_const_features)
        return train_r

    def train_n_steps(self, env, n_steps, max_episode_len=None, disable_tqdm=False, additional_const_features={}):
        return self._train_n_iters(env, n_steps, episodes=False, max_episode_len=max_episode_len, disable_tqdm=disable_tqdm, additional_const_features=additional_const_features)

    def _train_n_iters(self, env, n_iters, episodes=False, max_episode_len=None, disable_tqdm=False, additional_const_features={}):
        """General train function, if episodes is true- each iter is episode, otherwise train steps"""
        self.set_train_mode()
        if self.env is None:
            env = ParallelEnv(env, self.num_parallel_envs)
            self.env = env
        else:
            self.env.change_env(env)

        env = self.env
        pbar = tqdm(total=n_iters, leave=False, disable=disable_tqdm)
        curr_training_steps = 0
        train_rewards = []
        if episodes:
            to_update_idx = 0
        else:
            to_update_idx = 1

        i = 0
        ep_number = 0
        while i < n_iters:
            rewards_vector = self.collect_episode_obs(env, max_episode_len, num_to_collect=self.num_parallel_envs, additional_const_features=additional_const_features)
            num_steps_collected = 0
            for r in rewards_vector:
                train_rewards.append(np.sum(r))
                num_steps_collected += len(r)

            collect_info = [self.num_parallel_envs, num_steps_collected]
            curr_training_steps += num_steps_collected

            desciption = f"episode {ep_number}, R:{np.round(np.mean(train_rewards[-self.batch_size:]), 2):08}, total_steps:{curr_training_steps}"
            pbar.set_description(desciption)

            pbar.update(collect_info[to_update_idx])
            i += collect_info[to_update_idx]
            ep_number += self.num_parallel_envs

            if len(self.experience) < self.batch_size:
                # collect more sampels if not enough..
                continue
            self.update_policy()
            if self.rnn:
                self.reset_rnn_hidden()

        pbar.close()
        return train_rewards

    @abstractmethod
    def update_policy_reg(self, *exp):
        """Get batched samples and update policy"""
        raise NotImplementedError

    @abstractmethod
    def update_policy_rnn(self, *exp):
        """Get batched samples and update policy"""
        raise NotImplementedError

    def set_num_parallel_env(self, num_parallel_envs):
        self.num_parallel_envs = num_parallel_envs
        self.init_entropy_buffer()

    @abstractmethod
    def act(self, observations, num_obs=1):
        raise NotImplementedError


    def get_seqs_indices_for_pack(self, done_indices):
        """returns seq_lens, sorted_data_sub_indices"""
        env_indices = np.zeros(self.num_parallel_envs, dtype=np.int32)
        env_indices[0] = -1
        env_indices[1:] = done_indices[:-1]
        all_lens = done_indices - env_indices
        data_sub_indices = np.array([list(range(env_indices[i]+1, done_indices[i]+1, 1)) for i in range(len(all_lens-1))], dtype=object)

        seq_indices = np.argsort(all_lens, kind='stable')[::-1]
        sorted_data_sub_indices = data_sub_indices[seq_indices]
        sorted_data_sub_indices = np.concatenate(sorted_data_sub_indices).astype(np.int32)
        seq_lens = all_lens[seq_indices]
        return seq_lens, sorted_data_sub_indices

    def pack_from_done_indices(self, data, sorted_seq_lens, done_indices):
        """returns pakced obs"""
        assert np.all(np.sort(sorted_seq_lens)[::-1] == sorted_seq_lens)
        seq_lens = sorted_seq_lens
        max_len = np.max(seq_lens)

        packed_obs = ObsWraper()
        for k in data:
            obs_shape = data[k][-1].shape
            temp = []
            curr_idx = 0
            for d_i in done_indices:
                temp.append(data[k][curr_idx:d_i+1])
                curr_idx = d_i+1
            temp.sort(reverse=True, key=lambda x: x.size())
            padded_seq_batch = torch.nn.utils.rnn.pad_sequence(temp, batch_first=True)
            padded_seq_batch = padded_seq_batch.reshape((self.num_parallel_envs, max_len, np.prod(obs_shape)))
            pakced_states = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True)
            packed_obs[k] = pakced_states
        return packed_obs

    def pack_sorted_data(self, sorted_data, sorted_seq_lens):
        states = ObsWraper()
        for k in sorted_data:
            tmp = [torch.from_numpy(x).float() for x in sorted_data[k]]
            states[k] = self.pack_sorted_data_h(tmp, sorted_seq_lens).to(self.device)
        return states

    def pack_sorted_data_h(self, data, seq_lens):
        batch_size = len(data)
        max_seq_len = np.max(seq_lens).astype(np.int32)
        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        padded_seq_batch = padded_seq_batch.reshape((batch_size, max_seq_len, np.prod(data[-1].shape)))
        p_data = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True)
        return p_data

    def pre_process_obs_for_act(self, observations, num_obs):
        if type(observations) != ObsWraper:
            observations = ObsWraper(observations)

        
        len_obs = len(observations)
        if num_obs != len_obs and num_obs == 1:
            # BATCH=1
            len_obs = 1
            observations = observations[np.newaxis, :]
            if self.rnn:
                # seq len = 1, single state eval
                observations = observations[np.newaxis, :]
        elif num_obs != len_obs and num_obs != 1:
            raise Exception(f"number of observations do not match real observation num obs: {num_obs}, vs real len: {len(observations)}")
        # return observations
        states = ObsWraper()
        if self.rnn:
            seq_lens = np.ones(len_obs)
            states = self.pack_sorted_data(observations, seq_lens)
        else:
            for k in observations:
                states[k] = torch.from_numpy(observations[k]).to(self.device)
        return states

    def return_correct_actions_dim(self, selected_actions, num_obs):
        if num_obs == 1 and self.eval_mode:
            """non batch case, probably eval mode action"""
            selected_actions = selected_actions[0]
        return selected_actions

    def close_env_procs(self):
        self.env = None

    def collect_episode_obs(self, env, max_episode_len=None, num_to_collect=None, env_funcs={"step": "step", "reset": "reset"}, additional_const_features={}):
        # supports run on different env api
        if type(env) != ParallelEnv:
            # env = ParallelEnv(env, num_to_collect)
            if self.env is None:
                env = ParallelEnv(env, self.num_parallel_envs)
                self.env = env
            else:
                self.env.change_env(env)
        else:
            self.env = env

        step_function = getattr(env, env_funcs["step"])
        reset_function = getattr(env, env_funcs["reset"])

        if max_episode_len:
            def episode_len_exceeded(x): return x > max_episode_len
        else:
            def episode_len_exceeded(x): return False

        observations = []

        for item in reset_function():
            curr_item = ObsWraper(item)
            for k in additional_const_features:
                curr_item[k] = np.array([additional_const_features[k]] * len(curr_item))
            observations.append([curr_item])

        env_dones = np.array([False for i in range(self.num_parallel_envs)])

        latest_observations = [ObsWraper(observations[i][-1]) for i in range(self.num_parallel_envs)]
        latest_observations = ObsWraper(latest_observations)
        

        rewards = [[] for i in range(self.num_parallel_envs)]
        actions = [[] for i in range(self.num_parallel_envs)]
        next_observations = [[] for i in range(self.num_parallel_envs)]
        dones = [[] for i in range(self.num_parallel_envs)]

        max_episode_steps = 0
        while not all(env_dones):
            relevant_indices = np.where(env_dones == False)[0].astype(np.int32)

            
            current_actions = self.act(latest_observations, self.num_parallel_envs)
            # TODO DEBUG
            # allways use all envs to step, even some envs are done already
            next_obs, reward, done, info = step_function(current_actions)
            next_obs = ObsWraper(next_obs, keep_dims=True)
            for k in additional_const_features:
                next_obs[k] = np.array([additional_const_features[k]] * len(next_obs))

            for i in relevant_indices:
                actions[i].append(current_actions[i])
                next_observations[i].append(next_obs[i])
                rewards[i].append(reward[i])
                dones[i].append(done[i])
                env_dones[i] = done[i]

                max_episode_steps += 1
                if done[i]:
                    continue

                if episode_len_exceeded(max_episode_steps):
                    done = True
                    dones[i][-1] = done
                    env_dones[i] = done
                    break
            
                observations[i].append(next_obs[i])

            latest_observations = [observations[i][-1] for i in range(self.num_parallel_envs)]
            latest_observations = ObsWraper(latest_observations)
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
    def get_last_collected_experiences(self, number_of_episodes):
        # Mainly for paired alg
        raise NotImplementedError

    @abstractmethod
    def clear_exp(self):
        raise NotImplementedError
