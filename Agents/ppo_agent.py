import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Models.fc import FC
from .agent_utils import calc_returns, calc_gaes
from .agent import RL_Agent
from torch.distributions import Categorical



class PPO_Agent(RL_Agent):
    #TODO SUPPORT DIFFERNET OPTIMIZERS
    def __init__(self, obs_shape, n_actions, batch_size=32,
                 max_mem_size=10000, lr=0.0001, discount_factor=0.99, num_epochs_per_update=4, num_parallel_envs=1, model=FC, device = 'cpu'):
        """ppo recomeneded setting is 1 parallel env"""
        if num_parallel_envs > 1:
            print("Warning: PPO is online algorithm and do not benefit from multiple envirnoments, please set num_parallel_envs =1, convargence issue might preset if not")

        super().__init__(obs_shape, max_mem_size, batch_size, num_parallel_envs, model.is_rnn, device=device) # inits 
        # self.losses = []
        self.discount_factor = discount_factor
        self.action_space = [i for i in range(n_actions)]
        self.criterion = nn.MSELoss().to(device)
        self.model_constructor = model
        self.policy_nn = model(input_shape=obs_shape, out_shape=n_actions).to(device)
        self.actor_optimizer = optim.Adam(self.policy_nn.parameters(), lr)

        self.actor_model = lambda x : Categorical(logits=F.softmax(self.policy_nn(x), dim=1))

        self.critic_model = model(input_shape=obs_shape, out_shape=1).to(device) #output single value - V(s) and not Q(s,a) as before
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr)
        self.num_epochs_per_update = num_epochs_per_update
        self.clip_decay =  1
        self.clip_param =  0.1
        #FOR PPO UPDATE:
        self.init_ppo_buffers()


    def save_agent(self,f_name):
        torch.save({'actor_optimizer': self.actor_optimizer.state_dict(),
        'policy_nn':self.policy_nn.state_dict(),
        'critic_optimizer': self.critic_optimizer.state_dict(),
        'critic_model':self.critic_model.state_dict(),
        }, f_name)    


    def load_agent(self,f_name):
        checkpoint = torch.load(f_name)
        self.policy_nn.load_state_dict(checkpoint['policy_nn'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.actor_model = lambda x : Categorical(logits=F.softmax(self.policy_nn(x), dim=1))
        self.critic_model.load_state_dict(checkpoint['critic_model'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


    def reset_rnn_hidden(self,):
        """reset nn hidden_state"""
        self.policy_nn.reset()
        self.critic_model.reset()

    def set_num_parallel_env(self, num_parallel_envs):
        self.num_parallel_envs = num_parallel_envs
        self.init_ppo_buffers()

    # def update_rnn_indices(self, indices):
    #     """if agent uses rnn, this callback is called in many places so please impliment it"""
    #     self.policy_nn.update_hidden_state_indices(indices)
    #     self.critic_model.update_hidden_state_indices(indices)

    def get_last_collected_experiences(self, number_of_episodes):
        return self._get_ppo_experiences(number_of_episodes)

    def train(self, env, n_episodes):
        train_episode_rewards = super().train(env, n_episodes)
        self.experience.clear()
        self.init_ppo_buffers()
        return train_episode_rewards


    def act(self, observations, num_obs=1):
        """batched observation only!"""
        len_obs = len(observations)
        if num_obs != len_obs and num_obs == 1:
            observations = observations[np.newaxis, :]
            len_obs = 1
            if self.rnn:
                #seq len = 1, single state eval
                observations = observations[np.newaxis, :]
            
        elif num_obs != len_obs and num_obs != 1:
            raise Exception(f"number of observations do not match real observation len{num_obs}, vs {len_obs}")

        if self.rnn:
            observations = [torch.from_numpy(x) for x in observations]
            seq_lens = np.ones(len_obs)
            padded_seq_batch = torch.nn.utils.rnn.pad_sequence(observations, batch_first=True)
            padded_seq_batch= padded_seq_batch.reshape((len_obs, 1, np.prod(observations[-1].shape)))
            states = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True).to(self.device)
        else:
            states = torch.from_numpy(observations).to(self.device)

        with torch.no_grad():
            actions_dist = self.actor_model(states)
            if self.eval_mode:
                # all_actions = torch.max(actions_dist.probs, axis=1)[1]
                selected_actions = torch.argmax(actions_dist.probs, 1).detach().cpu().numpy().astype(np.int32)
            else:
                action = actions_dist.sample()
                log_probs = actions_dist.log_prob(action).detach().flatten().float()
                values = self.critic_model(states).detach().flatten().float()
                for i in range(num_obs):
                    self.logits[i].append(log_probs[i])
                    self.values[i].append(values[i])
                selected_actions = action.detach().cpu().numpy().astype(np.int32)

        return self.return_correct_actions_dim(selected_actions, num_obs)


    def init_ppo_buffers(self):
        self.logits = [[] for i in range(self.num_parallel_envs)]
        self.values = [[] for i in range(self.num_parallel_envs)]


    def _get_ppo_experiences(self, num_episodes= None):
        """Current PPO only suports random_Samples = False!!"""
        if num_episodes is None:
            num_episodes = self.num_parallel_envs

        assert num_episodes <= self.num_parallel_envs
        states, actions, rewards, dones, next_states = self.experience.get_last_episodes(self.num_parallel_envs)
        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)

        # ONLY POSSIBLE SINCE RANDOM SAMPLES ARE FALSE!!!
        done_indices = np.where(dones.cpu().numpy() == True)[0]
        values = torch.zeros_like(rewards, dtype=torch.float32, device=self.device)
        logits = torch.zeros_like(rewards, dtype=torch.float32, device=self.device)
        first_indice = 0

        for i in range(self.num_parallel_envs):
            current_done_indice = done_indices[i]
            curr_episode_len = current_done_indice - first_indice
            values[first_indice:current_done_indice] = torch.tensor(self.values[i][:curr_episode_len])
            logits[first_indice:current_done_indice] = torch.tensor(self.logits[i][:curr_episode_len])
            first_indice = current_done_indice

        return [states, actions, rewards, dones, next_states, values, logits]


    # def pack_data(self, data, seq_lens):
    #     padded_seq_batch = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    #     padded_seq_batch= padded_seq_batch.reshape((batch_size, 1, np.prod(observations[-1].shape)))
    #     states = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True)
    #     return states

    def get_entropy(self, obs, batch_size, seq_lens=None):
        if self.rnn:
            return self.get_entropy_rnn(obs, batch_size, seq_lens=None)
        return self.get_entropy_reg(obs, batch_size, seq_lens=None)


    def get_entropy_reg(self, obs, batch_size, seq_lens=None):
        assert seq_lens is None, "currently not supported"
        if batch_size == 1 and len(obs) != batch_size:
            obs = torch.unsqueeze(obs, 1)
        elif batch_size != len(obs):
            assert False, "Batch size doesnt match len of the obs"

        observations=  torch.from_numpy(obs)
        dist = self.actor_model(observations)

        entropy = dist.entropy().detach().cpu().numpy()
        self.reset_rnn_hidden()
        return entropy


    def get_entropy_rnn(self, obs, batch_size, seq_lens=None):
        self.reset_rnn_hidden()
        assert seq_lens is None, "currently not supported"
        observations = [torch.from_numpy(x) for x in obs]

        if batch_size == 1 and len(obs) != batch_size:
            obs = torch.unsqueeze(obs, 1)
        elif batch_size != len(obs):
            assert False, "Batch size doesnt match len of the obs"

        seq_lens = np.ones(batch_size)
        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(observations, batch_first=True)
        padded_seq_batch= padded_seq_batch.reshape((batch_size, 1, np.prod(observations[-1].shape)))
        states = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True).to(self.device)
        dist = self.actor_model(states)

        entropy = dist.entropy().detach().cpu().numpy()
        self.reset_rnn_hidden()
        return entropy


    def update_policy_reg(self):
        states, actions, rewards, dones, next_states, values, logits = self._get_ppo_experiences()
        returns = calc_returns(rewards, dones, self.discount_factor)
        advantages = calc_gaes(rewards, values, dones, self.discount_factor)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        self.init_ppo_buffers()

        all_samples_len = len(states)
        entropy_coeff = 0.001
        avg_c_loss = 0
        for e in range(self.num_epochs_per_update):
            indices_perm = torch.randperm(len(returns))
            states = states[indices_perm]
            actions = actions[indices_perm]
            returns = returns[indices_perm]
            advantages = advantages[indices_perm]
            logits = logits[indices_perm]
            kl_div_bool = False
            for b in range(0, all_samples_len, self.batch_size):
                batch_states = states[b:b+self.batch_size]
                batched_actions = actions[b:b+self.batch_size]
                batched_returns = returns[b:b+self.batch_size]
                batched_advantage = advantages[b:b+self.batch_size]
                batched_logits = logits[b:b+self.batch_size]

                dist = self.actor_model(batch_states)
                values = self.critic_model(batch_states)

                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(batched_actions)                

                old_log_probs = batched_logits # from acted policy

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                actor_loss  = - ((torch.min(surr1, surr2) * batched_advantage).mean()) - entropy_coeff * entropy
                critic_loss = self.criterion(values, torch.unsqueeze(batched_returns, 1))
                kl_div = (old_log_probs - new_log_probs).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                avg_c_loss  += critic_loss.item()
                self.critic_optimizer.step()

                if torch.abs(kl_div) > 0.01:
                    kl_div_bool = True
                    break
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # nn.utils.clip_grad_norm_(self.policy_nn.parameters(), 0.5)
                self.actor_optimizer.step()
            if kl_div_bool:
                break
        # self.losses.append(avg_c_loss / (self.batch_size*self.num_epochs_per_update))


    def update_policy_rnn(self, *exp):
        if len(exp) == 0:
            states, actions, rewards, dones, next_states, values, logits = self._get_ppo_experiences()
        else:
            states, actions, rewards, dones, next_states, values, logits = exp

        # if self.num_parallel_envs == 1:
        #     relevant_indices_list = np.arange(0, len(states), self.batch_size)        
        obs_shape = states[-1].shape
        returns = calc_returns(rewards, dones, self.discount_factor)
        advantages = calc_gaes(rewards, values, dones, self.discount_factor)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        self.init_ppo_buffers()

        env_indices = np.zeros(self.num_parallel_envs, dtype=np.int32)
        env_indices[0] = -1
        done_indices = torch.where(dones == True)[0].cpu().numpy().astype(np.int32)
        env_indices[1:] = done_indices[:-1]
        all_lens = done_indices - env_indices
        data_sub_indices = np.array([list(range(env_indices[i]+1,done_indices[i]+1, 1)) for i in range(len(all_lens-1))], dtype=object)
        
        seq_indices = np.argsort(all_lens)[::-1]
        sorted_data_sub_indices = data_sub_indices[seq_indices]
        sorted_data_sub_indices = np.concatenate(sorted_data_sub_indices).astype(np.int32)
        seq_lens = all_lens[seq_indices]

        unpadded_states = []
        unpadded_next_states = []
        curr_idx = 0
        for d_i in done_indices:
            unpadded_states.append(states[curr_idx:d_i+1])
            unpadded_next_states.append(next_states[curr_idx:d_i+1])
            curr_idx = d_i+1

        # can be better soretd in same loop!!!
        unpadded_states.sort(reverse=True,key=lambda x: x.size())
        sorted_actions = actions[sorted_data_sub_indices]
        sorted_returns = returns[sorted_data_sub_indices]
        sorted_advantage = advantages[sorted_data_sub_indices]
        sorted_logits = logits[sorted_data_sub_indices]

        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(unpadded_states, batch_first=True)
        max_len = np.max(seq_lens)
        padded_seq_batch= padded_seq_batch.reshape((self.num_parallel_envs, max_len, np.prod(obs_shape)))
        pakced_states = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True)        
        
        entropy_coeff = 0.001
        avg_c_loss = 0
        obs_shape = states[-1].shape

        for e in range(self.num_epochs_per_update):

            dist = self.actor_model(pakced_states)
            values = self.critic_model(pakced_states)

            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(sorted_actions)                

            old_log_probs = sorted_logits # from acted policy

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            actor_loss  = - ((torch.min(surr1, surr2) * sorted_advantage).mean()) - entropy_coeff * entropy
            critic_loss = self.criterion(values, torch.unsqueeze(sorted_returns, 1))
            kl_div = (old_log_probs - new_log_probs).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            avg_c_loss  += critic_loss.item()
            self.critic_optimizer.step()

            if torch.abs(kl_div) > 0.01:
                continue
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.policy_nn.parameters(), 0.5)
            self.actor_optimizer.step()

    def clear_exp(self):
        self.experience.clear()
        self.init_ppo_buffers()

