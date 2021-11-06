import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Models.fc import FC
from .agent_utils import ExperienceReplay, calc_reward_to_go
from .agent import RL_Agent
import tqdm

class PPO_Agent(RL_Agent):
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

        self.policy_model = model(n_actions=n_actions, obs_shape=obs_shape, softmax=True).to(device)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(),lr)

        self.value_model = model(n_actions=n_actions, obs_shape=obs_shape).to(device)
        self.value_optimizer = optim.Adam(self.value_model.parameters(),lr)
        super().__init__()


    def ppo_loss(advantages, prediction_picks, actions, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 5e-3

        prob = y_pred * actions
        old_prob = actions * prediction_picks
        r = prob/(old_prob + 1e-10)
        p1 = r * advantages
        p2 = torch.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
        loss =  -torch.mean(torch.minimum(p1, p2) + ENTROPY_LOSS * -(prob * torch.log(prob + 1e-10)))
        return loss


    # def train_episodial(self, env, n_episodes):
    #     self.set_train_mode()
    #     pbar = tqdm(range(n_episodes))
    #     curr_training_steps = 0
    #     for i in pbar:
    #         self.curr_collect_step = len(self.experience)
    #         reward_vector = self.collect_episode_obs(env)
    #         num_steps_collected = len(reward_vector)
    #         curr_training_steps +=num_steps_collected
            
    #         desciption = f"episode {i}, R:{np.sum(reward_vector)}, total_steps:{curr_training_steps}"
    #         pbar.set_description(desciption)

    #         self.curr_collect_step += len(reward_vector) # how many steps where collected before update policy
    #         if len(self.experience) < self.batch_size or self.curr_collect_step < self.collect_before_update:
    #             #collect more sampels if not enough..
    #             continue
    #         self.curr_collect_step  = 0
    #         states, actions, rewards, next_states = self.get_experiences(random_samples=True)
    #         self.update_policy(states, actions, rewards, next_states)


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


    def act(self, observation):
        if self.act_mode or np.random.random() > self.exploration_epsilon:
            state = torch.tensor([observation]).to(self.device)
            actions = self.Q_network(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space).astype(np.int32)
        # TO FIX

        return action


    def get_experiences(self, random_samples):
        """override random_samples and use only orderd samples"""
        observations, actions, rewards, dones, next_observations = super().get_experiences(random_samples=False)
        return observations, actions, rewards, dones, next_observations
        


    def get_advantages(values, rewards):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            gae = delta + gamma * lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    def update_policy(self, observations, actions, rewards, dones, next_observations):
        done_indices = np.where(dones == True)[0]
        rewards_per_episodes = rewards.tensor_split(done_indices, 1)
        rewards_to_go_per_episode = []
        for episode_rewards in rewards_per_episodes:
            rewards_to_go_per_episode.append(calc_reward_to_go(episode_rewards))
        rewards_to_go_per_episode = torch.cat(rewards_to_go_per_episode, 1)

        # num_epochs = 4
        # for j in range(num_epochs):
        start_idx = 0
        for i, done_idx in enumerate(done_indices):
            self.value_optimizer.zero_grad()
            values = self.value_model(observations[start_idx:done_idx])
            rewards_to_go = rewards_to_go_per_episode[i]

            loss = self.criterion(values, rewards_to_go)

            loss.backward()

            self.value_optimizer.step()

            #  observations = observations.float().to(device)
            #     actions = actions.long().to(device)
            #     advantages = advantages.float().to(device)
            #     old_log_probabilities = log_probabilities.float().to(device)

            self.policy_optimizer.zero_grad()

            # new_log_probabilities, entropy = policy_model.evaluate_actions(
            #     observations, actions
            # )

                # loss = (
                #     ac_loss(
                #         new_log_probabilities,
                #         old_log_probabilities,
                #         advantages,
                #         epsilon_clip=clip,
                #     ).mean()
                #     - c1 * entropy.mean()
                # )

            loss.backward()

            self.policy_optimizer.step()
            start_idx = done_idx
