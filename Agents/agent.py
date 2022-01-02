from torch._C import Node
from tqdm import tqdm
from abc import ABC, abstractmethod
from .agent_utils import ExperienceReplay
import numpy as np
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


    @abstractmethod
    def save_agent(self,):
        raise NotImplementedError

    @abstractmethod
    def get_entropy(self, obs, batch_size):
        raise NotImplementedError

    @abstractmethod
    def load_agent(self,):
        raise NotImplementedError


    def set_train_mode(self):
        self.reset_rnn_hidden()
        self.eval_mode = self.TRAIN


    def set_eval_mode(self):
        self.reset_rnn_hidden()
        self.eval_mode = self.EVAL


    def train_episodial(self, env, n_episodes, max_episode_len=None, disable_tqdm=False):
        return self._train_n_iters(env, n_episodes, True, max_episode_len=max_episode_len, disable_tqdm=disable_tqdm)


    def train_n_steps(self, env, n_steps, max_episode_len=None, disable_tqdm=False):
        return self._train_n_iters(env, n_steps, episodes=False, max_episode_len=max_episode_len, disable_tqdm=disable_tqdm)


    def _train_n_iters(self, env, n_iters, episodes=False, max_episode_len=None, disable_tqdm=False):
        """General train function, if episodes is true- each iter is episode, otherwise train steps"""
        self.set_train_mode()
        env = ParallelEnv(env, self.num_parallel_envs)
        pbar = tqdm(total=n_iters, leave=False, disable=disable_tqdm)
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
            rewards_vector = self.collect_episode_obs(env, max_episode_len, num_to_collect=self.num_parallel_envs)
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
            if self.rnn:
                self.reset_rnn_hidden()

        env.close_procs()
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

    @abstractmethod
    def act(self, observations, num_obs=1):
        raise NotImplementedError


    def return_correct_actions_dim(self, selected_actions, num_obs):
        if num_obs == 1 and self.eval_mode:
            """non batch case, probably eval mode action"""
            selected_actions = selected_actions[0]
        return selected_actions


    def collect_episode_obs(self, env, max_episode_len = None, num_to_collect=None, env_funcs={"step": "step", "reset": "reset"}):
        # supports run on different env api
        if type(env) != ParallelEnv:
            env = ParallelEnv(env, num_to_collect)

        step_function = getattr(env, env_funcs["step"])
        reset_function = getattr(env, env_funcs["reset"])

        if max_episode_len:
            episode_len_exceeded = lambda x: x > max_episode_len
        else:
            episode_len_exceeded = lambda x: False

        # observations = [[item] for item in parallel_envs.reset()]
        observations = [[item] for item in reset_function()]
        env_dones = np.array([False for i in range(self.num_parallel_envs)])

        latest_observations = np.array(([observations[i][-1] for i in range(self.num_parallel_envs)]))

        rewards = [[] for i in range(self.num_parallel_envs)]
        actions = [[] for i in range(self.num_parallel_envs)]
        next_observations = [[] for i in range(self.num_parallel_envs)]
        dones = [[] for i in range(self.num_parallel_envs)]

        max_episode_steps = 0
        while not all(env_dones):
            relevant_indices = np.where(env_dones == False)[0].astype(np.int32)
            current_actions = self.act(latest_observations, self.num_parallel_envs)
            #TODO DEBUG 
            # allways use all envs to step, even some envs are done already
            next_obs, reward, done, info = step_function(current_actions)

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
                    break

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
    def get_last_collected_experiences(self, number_of_episodes):
        #Mainly for paired alg
        raise NotImplementedError

    @abstractmethod
    def clear_exp(self):
        raise NotImplementedError