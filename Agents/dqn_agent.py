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
                 max_mem_size=10000, lr=0.0001, discount_factor=0.99, exploration_epsilon=1, eps_end=0.05, eps_dec=1e-4, num_parallel_envs=None, model=FC, rnn=False, device = 'cpu'):
        # rnn = model.is_rnn()
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
        # self.losses =[]


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

        if num_obs != len_obs and num_obs == 1:
            #BATCH=1
            len_obs = 1
            observations = observations[np.newaxis, :]
            if self.rnn:
                #seq len = 1, single state eval
                observations = observations[np.newaxis, :]

        elif num_obs != len_obs and num_obs != 1:
            raise Exception(f"number of observations do not match real observation len{num_obs}, vs {len(observations)}")

        if not self.eval_mode and np.random.random() > self.exploration_epsilon:
            selected_actions = np.random.choice(self.action_space, num_obs).astype(np.int32)
        else:
            if self.rnn:
                observations = [torch.from_numpy(x) for x in observations]
                # if not self.eval_mode:
                #     seq_lens = np.ones(self.num_parallel_envs)
                #     padded_seq_batch = torch.nn.utils.rnn.pad_sequence(observations, batch_first=True)
                # elif self.eval_mode:
                #     # X obs of len 1
                #     seq_lens = np.ones(len_obs)
                #     padded_seq_batch = torch.nn.utils.rnn.pad_sequence(observations, batch_first=True)
                seq_lens = np.ones(len_obs)
                padded_seq_batch = torch.nn.utils.rnn.pad_sequence(observations, batch_first=True)
                padded_seq_batch= padded_seq_batch.reshape((len_obs, 1, np.prod(observations[-1].shape)))
                states = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True).to(self.device)
            else:
                states = torch.from_numpy(observations).to(self.device)

            with torch.no_grad():
                all_actions = self.Q_model(states)
                all_actions = torch.squeeze(all_actions,1)
                selected_actions = torch.argmax(all_actions, -1).detach().cpu().numpy().astype(np.int32)

        return self.return_correct_actions_dim(selected_actions, num_obs)

    
    def reset_rnn_hidden(self,):
        """reset nn hidden_state"""
        self.Q_model.reset()
        self.target_Q_model.reset()


    # def update_rnn_indices(self, indices):
    #     """if agent uses rnn, this callback is called in many places so please impliment it"""
    #     self.Q_model.update_hidden_state_indices(indices)
    #     self.target_Q_model.update_hidden_state_indices(indices)


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
            observations = np.concatenate([observations, latest_observations])[rand_perm]
            actions = np.concatenate([actions, latest_actions])[rand_perm]
            rewards = np.concatenate([rewards, latest_rewards])[rand_perm]
            dones = np.concatenate([dones, latest_dones])[rand_perm]
            next_observations = np.concatenate([next_observations, latest_next_observations])[rand_perm]
        else:
            observations, actions, rewards, dones, next_observations = self.experience.get_last_episodes(self.num_parallel_envs)
        
        observations = torch.tensor(observations).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        next_observations = torch.tensor(next_observations).to(self.device)
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

        obs_shape = states[-1].shape

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
        unpadded_next_states.sort(reverse=True,key=lambda x: x.size())
        sorted_actions = actions[sorted_data_sub_indices]
        sorted_rewards = rewards[sorted_data_sub_indices]
        sorted_dones = dones[sorted_data_sub_indices]
        

        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(unpadded_states, batch_first=True)
        padded_next_seq_batch = torch.nn.utils.rnn.pad_sequence(unpadded_next_states, batch_first=True)
        max_len = np.max(seq_lens)
        padded_seq_batch= padded_seq_batch.reshape((self.num_parallel_envs, max_len, np.prod(obs_shape)))
        padded_next_seq_batch= padded_next_seq_batch.reshape((self.num_parallel_envs, max_len, np.prod(obs_shape)))
        pakced_states = torch.nn.utils.rnn.pack_padded_sequence(padded_seq_batch, lengths=seq_lens, batch_first=True)        
        pakced_next_states = torch.nn.utils.rnn.pack_padded_sequence(padded_next_seq_batch, lengths=seq_lens, batch_first=True)      
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


    def get_last_collected_experiences(self):
        """Mainly for Paired Algorithm support"""
        return (torch.tensor(x).to(self.device) for x in self.experience.get_last_episodes(self.num_parallel_envs))