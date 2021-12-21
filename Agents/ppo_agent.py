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
                 max_mem_size=10000, lr=0.0001, discount_factor=0.99, num_epochs_per_update=4, num_parallel_envs=1, model=FC, rnn=False, device = 'cpu'):
        """ppo recoomeneded is 1 parallel env"""
        super().__init__(obs_shape, max_mem_size, batch_size, num_parallel_envs, rnn, device=device) # inits 
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

        torch.save({'optimizer': self.optimizer.state_dict(),
        'model':self.Q_network.state_dict()
        }, f_name)


    def load_agent(self,f_name):
        checkpoint = torch.load(f_name)
        self.Q_network.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


    def train(self, env, n_episodes):
        train_episode_rewards = super().train(env, n_episodes)
        self.experience.clear()
        return train_episode_rewards


    def act(self, observations, num_obs=1):
        """batched observation only!"""
        if num_obs != len(observations) and num_obs == 1:
            observations = observations[np.newaxis, :]
        elif num_obs != len(observations) and num_obs != 1:
            raise Exception(f"number of observations do not match real observation len{num_obs}, vs {len(observations)}")

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
        self.logits = [[] for i in range(self.batch_size)]
        self.values = [[] for i in range(self.batch_size)]


    def get_experiences(self, random_samples):
        """Current PPO only suports random_Samples = False!!"""
        assert random_samples == False, "Current PPO only suports random_Samples = False!!"
        states, actions, rewards, dones, next_states = super().get_experiences(random_samples) 

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

        return states, actions, rewards, dones, next_states, values, logits
    

    def update_policy(self):
        # assert self.num_parallel_envs == 1 and self.rnn or not self.rnn, "currently impl does not support multiple parralel envs"
        # states, actions, rewards, dones, next_states, values, logits = self.get_rnn_exp(random_samples=False)
        states, actions, rewards, dones, next_states, values, logits = self.get_experiences(random_samples=False)
        returns = calc_returns(rewards, dones, self.discount_factor)
        advantages = calc_gaes(rewards, values, dones, self.discount_factor)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        self.init_ppo_buffers()
        

        all_samples_len = len(states)
        entropy_coeff = 0.001
        avg_c_loss = 0
        for e in range(self.num_epochs_per_update):
            indices_perm = torch.arange(len(returns))
            if self.rand_perm:
                #do not when using RNN
                indices_perm = torch.randperm(len(returns))
            states = states[indices_perm]
            actions = actions[indices_perm]
            returns = returns[indices_perm]
            advantages = advantages[indices_perm]
            logits = logits[indices_perm]

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
                    continue
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # nn.utils.clip_grad_norm_(self.policy_nn.parameters(), 0.5)
                self.actor_optimizer.step()



        # self.losses.append(avg_c_loss / (self.batch_size*self.num_epochs_per_update))
