import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Models.fc import FC
from .agent_utils import ExperienceReplay
from .agent import RL_Agent


class DQN_Agent(RL_Agent):
    #TODO SUPPORT DIFFERNET OPTIMIZERS
    def __init__(self, obs_shape, n_actions, batch_size=32,
                 max_mem_size=10000, lr=0.0001, discount_factor=0.99, exploration_epsilon=1, eps_end=0.05, eps_dec=1e-4, model=FC, device = 'cpu'):
        super().__init__(obs_shape,max_mem_size, batch_size, device=device) # inits 

        self.discount_factor = discount_factor
        self.exploration_epsilon = exploration_epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.criterion = nn.SmoothL1Loss().to(device)



        self.Q_network = model(n_actions=n_actions, obs_shape=obs_shape).to(device)
        self.target_Q_network = model(n_actions=n_actions, obs_shape=obs_shape).to(device)
        self.update_target()
        self.target_update_time = self.collect_before_update * 100 # update target every 100 learning steps
        self.target_update_counter = 0

        self.optimizer = optim.Adam(self.Q_network.parameters(),lr)

    def update_target(self):
        self.target_Q_network.load_state_dict(self.Q_network.state_dict())
        for p in self.target_Q_network.parameters():
            p.requires_grad = False
        self.target_Q_network.eval()

    def save_agent(self,f_name):

        torch.save({'optimizer': self.optimizer.state_dict(),
        'model':self.Q_network.state_dict()
        }, f_name)

    def load_agent(self,f_name):
        checkpoint = torch.load(f_name)
        self.Q_network.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_target()

    def train(self, env, n_episodes):
        train_episode_rewards = super().train(env, n_episodes)
        self.experience.clear()
        return train_episode_rewards


    def act(self, observation):
        if self.act_mode or np.random.random() > self.exploration_epsilon:
            state = torch.tensor([observation]).to(self.device)
            actions = self.Q_network(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space).astype(np.int32)
        # TO FIX

        return action

    def update_policy(self, states, actions, rewards, dones, next_states):
        v_table = self.Q_network(states)
        q_values = v_table[np.arange(self.batch_size), actions]
        with torch.no_grad():
            q_next = self.target_Q_network(next_states).max(1)[0]

        expected_next_values = rewards + self.discount_factor*q_next

        loss = self.criterion(q_values, expected_next_values) #.unsqueeze(1)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in policy_net.parameters():
            # param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.target_update_counter += 1
        if self.target_update_counter > self.target_update_time:
            self.update_target()
            self.target_update_counter = 0


        self.exploration_epsilon = self.exploration_epsilon * (1-self.eps_dec) \
            if self.exploration_epsilon > self.eps_min else self.eps_min
