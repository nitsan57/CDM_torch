import functools
import operator
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Models.fc import FC
from .agent_utils import calc_returns, calc_gaes
from .drl_agent import RL_Agent
from torch.distributions import Categorical




class PPO_Agent(RL_Agent):
    #TODO SUPPORT DIFFERNET OPTIMIZERS
    def __init__(self, obs_shape, n_actions, batch_size=64,
                 max_mem_size=10000, lr=0.0001, discount_factor=0.99, exploration_epsilon=0.0, entropy_coeff=0.001, num_epochs_per_update=4, num_parallel_envs=1, model=FC, device = 'cpu'):
        """ppo recomeneded setting is 1 parallel env (with non RNN envs, bacuase of too many gradients updates)"""
        if num_parallel_envs > 1 and not model.is_rnn:
            print("Warning: PPO is online algorithm and do not benefit from multiple envirnoments, please set num_parallel_envs =1, convargence issue might preset if not")

        super().__init__(obs_shape, n_actions, max_mem_size, batch_size, num_parallel_envs, model=model, lr=lr, device=device) # inits 
        # self.losses = []
        self.entropy_coeff = entropy_coeff
        self.discount_factor = discount_factor
        self.action_space = [i for i in range(n_actions)]
        self.criterion = nn.MSELoss().to(device)
        self.init_models()
        self.num_epochs_per_update = num_epochs_per_update
        self.clip_decay =  1
        self.clip_param =  0.1
        #FOR PPO UPDATE:
        self.exploration_epsilon = exploration_epsilon
        self.init_ppo_buffers()


    def init_models(self):
        self.policy_nn = self.model(input_shape=self.obs_shape, out_shape=self.n_actions).to(self.device)
        self.actor_optimizer = optim.Adam(self.policy_nn.parameters(), self.lr)
        self.actor_model = lambda x : Categorical(logits=F.log_softmax(self.policy_nn(x), dim=1))
        self.critic_model = self.model(input_shape=self.obs_shape, out_shape=1).to(self.device) #output single value - V(s) and not Q(s,a) as before
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), self.lr)


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
        self.actor_model = lambda x : Categorical(logits=F.log_softmax(self.policy_nn(x), dim=1))
        self.critic_model.load_state_dict(checkpoint['critic_model'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


    def reset_rnn_hidden(self,):
        """reset nn hidden_state"""
        self.policy_nn.reset()
        self.critic_model.reset()

    def set_num_parallel_env(self, num_parallel_envs):
        super().set_num_parallel_env(num_parallel_envs)
        self.init_ppo_buffers()


    def get_last_collected_experiences(self, number_of_episodes):
        return self._get_ppo_experiences(number_of_episodes, safe_check=False)


    def train(self, env, n_episodes):
        train_episode_rewards = super().train(env, n_episodes)
        self.experience.clear()
        self.init_ppo_buffers()
        return train_episode_rewards

    def best_act(self, observations, num_obs=1):
        """batched observation only!"""
        if not self.eval_mode:
            print("warning using this function will not accumulate train data, if this is the intention use eval mode to avoid the message")

        states = self.pre_process_obs_for_act(observations, num_obs)

        with torch.no_grad():
            actions_dist = self.actor_model(states)

            selected_actions = torch.argmax(actions_dist.probs, 1).detach().cpu().numpy().astype(np.int32)
            return self.return_correct_actions_dim(selected_actions, num_obs)


    def act(self, observations, num_obs=1):
        """batched observation only!"""
        states = self.pre_process_obs_for_act(observations, num_obs)

        with torch.no_grad():
            actions_dist = self.actor_model(states)
            if self.eval_mode:
                # all_actions = torch.max(actions_dist.probs, axis=1)[1]
                # selected_actions = torch.argmax(actions_dist.probs, 1).detach().cpu().numpy().astype(np.int32)
                selected_actions = actions_dist.sample().detach().cpu().numpy().astype(np.int32)

            else:
                if not self.eval_mode and np.random.random() < self.exploration_epsilon:
                    action = torch.tensor(np.random.choice(self.action_space, num_obs).astype(np.int32), device=self.device)
                    log_probs = actions_dist.log_prob(action).detach().flatten().float()
                else:
                    action = actions_dist.sample()
                    log_probs = actions_dist.log_prob(action).detach().flatten().float()

                values = self.critic_model(states).detach().flatten().float()
                
                for i in range(num_obs):
                    self.logits[i].append(log_probs[i])
                    self.values[i].append(values[i])

                if self.store_entropy:
                    
                    all_ent = actions_dist.entropy().detach().cpu().numpy()
                    
                    for i in range(self.num_parallel_envs):
                        self.stored_entropy[i].append(all_ent[i])
                        
                if self.store_values:
                    for i in range(self.num_parallel_envs):
                        self.stored_values[i].append(values[i])
                selected_actions = action.detach().cpu().numpy().astype(np.int32)

        return self.return_correct_actions_dim(selected_actions, num_obs)


    def init_ppo_buffers(self):
        self.logits = [[] for i in range(self.num_parallel_envs)]
        self.values = [[] for i in range(self.num_parallel_envs)]


    def _get_ppo_experiences(self, num_episodes= None, safe_check=True):
        """Current PPO only suports random_Samples = False!!"""
        if num_episodes is None:
            num_episodes = self.num_parallel_envs

        if safe_check:
            assert num_episodes <= self.num_parallel_envs
        states, actions, rewards, dones, next_states = self.experience.get_last_episodes(num_episodes)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        states = states.get_as_tensors(self.device)
        next_states = next_states.get_as_tensors(self.device)

        # ONLY POSSIBLE SINCE RANDOM SAMPLES ARE FALSE!!!
        done_indices = np.where(dones.cpu().numpy() == True)[0]
        values = torch.zeros_like(rewards, dtype=torch.float32, device=self.device)
        logits = torch.zeros_like(rewards, dtype=torch.float32, device=self.device)

        first_indice = -1
        for i in range(self.num_parallel_envs):
            current_done_indice = done_indices[i]
            curr_episode_len = current_done_indice - first_indice
            values[first_indice+1:current_done_indice+1] = torch.tensor(self.values[i][:curr_episode_len])
            logits[first_indice+1:current_done_indice+1] = torch.tensor(self.logits[i][:curr_episode_len])
            first_indice = current_done_indice

        return [states, actions, rewards, dones, next_states, values, logits]


    # def get_entropy(self, obs, batch_size, seq_lens=None):
    #     if self.rnn:
    #         return self.get_entropy_rnn(obs, batch_size, seq_lens=None)
    #     return self.get_entropy_reg(obs, batch_size, seq_lens=None)


    # def get_entropy_reg(self, obs, batch_size, seq_lens=None):
    #     assert seq_lens is None, "currently not supported"
    #     if batch_size == 1 and len(obs) != batch_size:
    #         obs = torch.unsqueeze(obs, 1)
    #     elif batch_size != len(obs):
    #         assert False, "Batch size doesnt match len of the obs"

    #     observations=  torch.from_numpy(obs)
    #     dist = self.actor_model(observations)
    #     entropy = dist.entropy().detach().cpu().numpy()
    #     return entropy


    # def get_entropy_rnn(self, obs, batch_size, seq_lens=None):
    #     self.reset_rnn_hidden()
    #     assert seq_lens is None, "currently not supported"
    #     if batch_size == 1 and len(obs) != batch_size:
    #         observations = [torch.from_numpy(obs)]
    #     elif batch_size != len(obs):
    #         assert False, "Batch size doesnt match len of the obs"
    #     else:
    #         observations = [torch.from_numpy(x) for x in obs]
    #     seq_lens = np.ones(batch_size)
    #     states =  self.pack_data(observations, seq_lens).to(self.device)
    #     dist = self.actor_model(states)
    #     entropy = dist.entropy().detach().cpu().numpy()
    #     self.reset_rnn_hidden()
    #     return entropy


    def update_policy_reg(self):
        states, actions, rewards, dones, next_states, values, logits = self._get_ppo_experiences()
        # returns = calc_returns(rewards, dones, self.discount_factor)


        advantages = calc_gaes(rewards, values, dones, self.discount_factor)
        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        self.init_ppo_buffers()

        all_samples_len = len(states)
        entropy_coeff = self.entropy_coeff
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

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                avg_c_loss  += critic_loss.item()
                self.critic_optimizer.step()

                if torch.abs(kl_div) > 0.01:
                    kl_div_bool = True
                    break
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_nn.parameters(), 0.5)
                self.actor_optimizer.step()
            if kl_div_bool:
                break
        # self.losses.append(avg_c_loss / (self.batch_size*self.num_epochs_per_update))

    def update_policy_rnn(self, *exp):
        self.reset_rnn_hidden()
        if len(exp) == 0:
            states, actions, rewards, dones, next_states, values, logits = self._get_ppo_experiences()
        else:
            states, actions, rewards, dones, next_states, values, logits = exp

        advantages = calc_gaes(rewards, values, dones, self.discount_factor)

        returns = advantages + values


        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-10)

        self.init_ppo_buffers()

        done_indices = torch.where(dones == True)[0].cpu().numpy().astype(np.int32)
        assert len(done_indices) <= self.batch_size ,f"batch size < number of env to train, {self.batch_size}, {len(done_indices)}"


        seq_lens, sorted_data_sub_indices = self.get_seqs_indices_for_pack(done_indices)
        pakced_states = self.pack_from_done_indices(states, seq_lens, done_indices)
        # pakced_states = states[sorted_data_sub_indices]
        sorted_actions = actions[sorted_data_sub_indices]
        sorted_returns = returns[sorted_data_sub_indices]
        sorted_advantage = advantages[sorted_data_sub_indices]
        sorted_logits = logits[sorted_data_sub_indices]


        
        entropy_coeff = 0.001
        num_samples = len(states)
        num_grad_updates = num_samples // self.batch_size
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

            self.critic_optimizer.zero_grad(set_to_none=True)
            # self.losses.append(critic_loss.item())

            critic_loss.backward()
            self.critic_optimizer.step()

                
            if torch.abs(kl_div) > 0.01:  
                break
            self.actor_optimizer.zero_grad(set_to_none=True)
            
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_nn.parameters(), 0.5)
            self.actor_optimizer.step()


    def clear_exp(self):
        self.experience.clear()
        self.init_ppo_buffers()

