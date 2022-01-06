import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Agents.agent_utils import ObsWraper
from Models.fc import FC
from .agent import RL_Agent
from torch.distributions import Categorical

class DQN_Agent(RL_Agent):
    #TODO SUPPORT DIFFERNET OPTIMIZERS
    def __init__(self, obs_shape, n_actions, batch_size=32,
                 max_mem_size=10000, lr=0.0001, discount_factor=0.99, exploration_epsilon=1, eps_end=0.05, eps_dec=1e-4, num_parallel_envs=None, model=FC, device = 'cpu'):
        super().__init__(obs_shape, n_actions, max_mem_size, batch_size, num_parallel_envs, model=model, lr=lr, device=device) # inits
        
        self.discount_factor = discount_factor
        self.exploration_epsilon = exploration_epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.criterion = nn.MSELoss().to(device)
        self.init_models()

        self.target_update_time = 100 # update target every 100 learning steps
        self.target_update_counter = 0
        
        # self.losses =[]

    def init_models(self):
        self.Q_model = self.model(input_shape=self.obs_shape, out_shape=self.n_actions).to(self.device)
        self.target_Q_model = self.model(input_shape=self.obs_shape, out_shape=self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.Q_model.parameters(), self.lr, amsgrad=True)
        self.update_target()
        
        
    def update_target(self):
        self.target_Q_model.load_state_dict(self.Q_model.state_dict())
        for p in self.target_Q_model.parameters():
            p.requires_grad = False
        self.target_Q_model.eval()


    def get_entropy(self, obs, batch_size, seq_lens=None):
        if self.rnn:
            return self.get_entropy_rnn(obs, batch_size, seq_lens=None)
        return self.get_entropy0_reg(obs, batch_size, seq_lens=None)



    # def get_entropy_reg(self, obs, batch_size, seq_lens=None):
    #     assert seq_lens is None, "currently not supported"
    #     if batch_size == 1 and len(obs) != batch_size:
    #         obs = torch.unsqueeze(obs, 1)
    #     elif batch_size != len(obs):
    #         assert False, "Batch size doesnt match len of the obs"

    #     observations =  torch.from_numpy(obs)
    #     res = self.Q_model(observations)
    #     logits = F.log_softmax(res,1)
    #     entropy = Categorical(logits= logits).entropy().detach().cpu().numpy()
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
    #     states =  self.pack_sorted_data_h(observations, seq_lens).to(self.device)
    #     res = self.Q_model(states)
    #     logits = F.log_softmax(res,1)
    #     self.reset_rnn_hidden()
    #     return Categorical(logits= logits).entropy().detach().cpu().numpy()


    # def get_entropy(self, obs, batch_size):
    #     if batch_size == len(obs):
    #         res = self.Q_model(obs)
    #     elif batch_size == 1 and len(obs) != batch_size:
    #         obs = torch.unsqueeze(obs, 1)
    #         res = self.Q_model(obs)
    #     else:
    #         assert False, "Batch size doesnt match len of the obs"
    #     logits = F.log_softmax(res,1)
    #     return Categorical(logits= logits).entropy().detach().cpu().numpy()
        

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
        if not self.eval_mode and np.random.random() > self.exploration_epsilon:
            selected_actions = np.random.choice(self.action_space, num_obs).astype(np.int32)
        else:
            states = self.pre_process_obs_for_act(observations, num_obs)
            with torch.no_grad():
                all_actions = self.Q_model(states)
                all_actions = torch.squeeze(all_actions,1)

                if self.store_entropy:
                    all_ent = self.calc_entropy_from_vec(all_actions)
                    for i in range(self.num_parallel_envs):
                        self.stored_entropy[i].append(all_ent[i])
                selected_actions = torch.argmax(all_actions, -1).detach().cpu().numpy().astype(np.int32)

        return self.return_correct_actions_dim(selected_actions, num_obs)

    
    def reset_rnn_hidden(self,):
        """reset nn hidden_state"""
        self.Q_model.reset()
        self.target_Q_model.reset()


    def _get_dqn_experiences(self, random_samples):
        """Get a mix of samples, including all last episode- makes sure we dont miss any seen states"""
        if random_samples:
            latest_experience_batch = self.experience.get_last_episodes(self.num_parallel_envs)
            last_episode_len = len(latest_experience_batch[0])
            #50% last episodes, 50% random
            random_experience_batch = self.experience.sample_random_batch(last_episode_len)
            observations, actions, rewards, dones, next_observations = random_experience_batch
            latest_observations, latest_actions, latest_rewards, latest_dones, latest_next_observations = latest_experience_batch
            rand_perm = torch.randperm(2*len(observations))
            observations = observations.np_cat(latest_observations)[rand_perm]
            actions = np.concatenate([actions, latest_actions])[rand_perm]
            rewards = np.concatenate([rewards, latest_rewards])[rand_perm]
            dones = np.concatenate([dones, latest_dones])[rand_perm]
            next_observations = next_observations.np_cat(latest_next_observations)[rand_perm]

        else:
            observations, actions, rewards, dones, next_observations = self.experience.get_last_episodes(self.num_parallel_envs)
        
        # observations = torch.tensor(observations).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        # next_observations = torch.tensor(next_observations).to(self.device)
        observations = observations.get_as_tensors(self.device)
        next_observations = next_observations.get_as_tensors(self.device)

        return observations, actions, rewards, dones, next_observations


    def update_policy_reg(self, *exp):
        if len(exp) == 0:
            states, actions, rewards, dones, next_states = self._get_dqn_experiences(random_samples=(not self.rnn))
        else:
            states, actions, rewards, dones, next_states = exp

        all_samples_len = len(states)

        for b in range(0,all_samples_len, self.batch_size):
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

            expected_next_values = batched_rewards + (1-batched_dones) * self.discount_factor * q_next

            loss = self.criterion(q_values, expected_next_values)

            # Optimize the model
            self.optimizer.zero_grad()
            # self.losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            self.target_update_counter += 1
            if self.target_update_counter > self.target_update_time:
                self.update_target()
                self.target_update_counter = 0

        self.exploration_epsilon = self.exploration_epsilon * (1-self.eps_dec) \
            if self.exploration_epsilon > self.eps_min else self.eps_min


    def update_policy_rnn(self, *exp):
        if len(exp) == 0:
            states, actions, rewards, dones, next_states = self._get_dqn_experiences(random_samples=(not self.rnn))
        else:
            states, actions, rewards, dones, next_states = exp
        done_indices = torch.where(dones == True)[0].cpu().numpy().astype(np.int32)
        seq_lens, sorted_data_sub_indices = self.get_seqs_indices_for_pack(done_indices)
        sorted_actions = actions[sorted_data_sub_indices]
        sorted_rewards = rewards[sorted_data_sub_indices]
        sorted_dones = dones[sorted_data_sub_indices]
        pakced_states = self.pack_from_done_indices(states, seq_lens, done_indices)
        pakced_next_states = self.pack_from_done_indices(next_states, seq_lens, done_indices)

        v_table = self.Q_model(pakced_states)
        v_table = v_table #.reshape(normal_b_size, len(self.action_space))
        q_values = v_table[np.arange(len(v_table)), sorted_actions]
        with torch.no_grad():
            q_next = self.target_Q_model(pakced_next_states).detach().max(1)[0] #.reshape(normal_b_size, len(self.action_space))
        expected_next_values = sorted_rewards + (1-sorted_dones) * self.discount_factor * q_next
        loss = self.criterion(q_values, expected_next_values)
        # self.losses.append(loss.item())
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


    def get_last_collected_experiences(self, num_episodes):
        """Mainly for Paired Algorithm support"""
        # return [torch.tensor(x).to(self.device) for x in self.experience.get_last_episodes(num_episodes)]

        exp  = self.experience.get_last_episodes(num_episodes)
        res = []
        for i,x in enumerate(exp):
            if i == self.experience.states_index or i == self.experience.next_states_index:
                res.append(x.get_as_tensors(self.device))
            else:
                res.append(torch.tensor(x).to(self.device))
        return res

    
    def clear_exp(self):
        self.experience.clear()