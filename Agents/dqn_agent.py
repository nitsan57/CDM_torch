import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Models.fc import FC
from .agent import RL_Agent


class DQN_Agent(RL_Agent):
    #TODO SUPPORT DIFFERNET OPTIMIZERS
    def __init__(self, obs_shape, n_actions, batch_size=32,
                 max_mem_size=10000, lr=0.0001, discount_factor=0.99, exploration_epsilon=1, eps_end=0.05, eps_dec=1e-4,num_parallel_envs=None, model=FC, rnn=False, device = 'cpu'):
        super().__init__(obs_shape, max_mem_size, batch_size, num_parallel_envs, rnn, device=device) # inits 

        self.discount_factor = discount_factor
        self.exploration_epsilon = exploration_epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.criterion = nn.MSELoss().to(device)
        self.Q_model = model(input_shape=obs_shape, out_shape=n_actions).to(device)
        self.target_Q_model = model(input_shape=obs_shape, out_shape=n_actions).to(device)
        self.update_target()
        self.target_update_time = 100 # update target every 100 learning steps
        self.target_update_counter = 0
        self.optimizer = optim.Adam(self.Q_model.parameters(), lr, amsgrad=True)


    def update_target(self):
        self.target_Q_model.load_state_dict(self.Q_model.state_dict())
        for p in self.target_Q_model.parameters():
            p.requires_grad = False
        self.target_Q_model.eval()


    def save_agent(self,f_name):

        torch.save({'optimizer': self.optimizer.state_dict(),
        'model':self.Q_model.state_dict()
        }, f_name)


    def load_agent(self,f_name):
        checkpoint = torch.load(f_name)
        self.Q_model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_target()


    def train(self, env, n_episodes):
        train_episode_rewards = super().train(env, n_episodes)
        self.experience.clear()
        return train_episode_rewards


    def act(self, observations, num_obs=1):
        """batched observation only!"""
        len_obs = len(observations)
        # if self.rnn:
        #     len_obs = len(observations.data)

        if num_obs != len_obs and num_obs == 1:
            observations = observations[np.newaxis, :]
        elif num_obs != len_obs and num_obs != 1:
            raise Exception(f"number of observations do not match real observation len{num_obs}, vs {len(observations)}")

        if self.eval_mode or np.random.random() > self.exploration_epsilon:
            # if not self.rnn:
            states = torch.from_numpy(observations).to(self.device)
            # else:
            #     states = observations.to(self.device)

            with torch.no_grad():
                all_actions = self.Q_model(states)
                selected_actions = torch.argmax(all_actions, 1).detach().cpu().numpy().astype(np.int32)
        else:
            selected_actions = np.random.choice(self.action_space, num_obs).astype(np.int32)

        return self.return_correct_actions_dim(selected_actions, num_obs)

    
    def reset_rnn_hidden(self,):
        """reset nn hidden_state"""
        self.Q_model.reset()
        self.target_Q_model.reset()


    def update_rnn_indices(self, indices):
        """if agent uses rnn, this callback is called in many places so please impliment it"""
        self.Q_model.update_hidden_state_indices(indices)
        self.target_Q_model.update_hidden_state_indices(indices)


    def update_policy_rnn(self):
        # self.get_rnn_exp(random_samples=self.rand_perm)
        # states, actions, rewards,dones, next_states = self.get_experiences(random_samples=self.rand_perm)
        # all_samples_len = len(states)

        # for b in range(0,all_samples_len, self.batch_size):
            # batch_states = states[b:b+self.batch_size]
            # batched_actions = actions[b:b+self.batch_size]
            # batched_next_states = next_states[b:b+self.batch_size]
            # batched_rewards = rewards[b:b+self.batch_size]
            # batched_dones = dones[b:b+self.batch_size]

        for batch_states, batched_actions, batched_rewards, batched_dones, batched_next_states in self.get_rnn_exp(random_samples=self.rand_perm):
            v_table = self.Q_model(batch_states)
            # only because last batch is smaller
            real_batch_size = len(batch_states) 
            q_values = v_table[np.arange(real_batch_size), batched_actions]
            with torch.no_grad():
                q_next = self.target_Q_model(batched_next_states).detach().max(1)[0]

            expected_next_values = batched_rewards + (1-batched_dones) * self.discount_factor*q_next

            loss = self.criterion(q_values, expected_next_values)

            # Optimize the model
            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()
            self.target_update_counter += 1
            if self.target_update_counter > self.target_update_time:
                self.update_target()
                self.target_update_counter = 0

        self.exploration_epsilon = self.exploration_epsilon * (1-self.eps_dec) \
            if self.exploration_epsilon > self.eps_min else self.eps_min


    def update_policy_reg(self):
        states, actions, rewards,dones, next_states = self.get_experiences(random_samples=self.rand_perm)
        all_samples_len = len(states)

        for b in range(0,all_samples_len, self.batch_size):
        # for batch_states, batched_actions, batched_rewards, batched_dones, batched_next_states in self.get_rnn_exp(random_samples=self.rand_perm):
            batch_states = states[b:b+self.batch_size]
            batched_actions = actions[b:b+self.batch_size]
            batched_next_states = next_states[b:b+self.batch_size]
            batched_rewards = rewards[b:b+self.batch_size]
            batched_dones = dones[b:b+self.batch_size]
            v_table = self.Q_model(batch_states)
            # only because last batch is smaller
            real_batch_size = len(batch_states) 
            q_values = v_table[np.arange(real_batch_size), batched_actions]
            with torch.no_grad():
                q_next = self.target_Q_model(batched_next_states).detach().max(1)[0]

            expected_next_values = batched_rewards + (1-batched_dones) * self.discount_factor*q_next

            loss = self.criterion(q_values, expected_next_values)

            # Optimize the model
            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()
            self.target_update_counter += 1
            if self.target_update_counter > self.target_update_time:
                self.update_target()
                self.target_update_counter = 0

        self.exploration_epsilon = self.exploration_epsilon * (1-self.eps_dec) \
            if self.exploration_epsilon > self.eps_min else self.eps_min