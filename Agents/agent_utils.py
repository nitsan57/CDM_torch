import gym
from time import sleep
from collections import namedtuple
from multiprocessing import Process, Pipe
import numpy as np
import torch
import copy

from Environments.base_curriculum_env import Base_Env


def calc_returns(rewards, dones, discount_factor):
    """works with rewards vector which consitst of many epidsodes"""
    curr = 0
    device = rewards.device
    returns = torch.zeros_like(rewards, device=device)
    num_rewards = len(rewards)
    for i, r in enumerate(torch.flip(rewards, dims=(0,))):
        reveres_idx = num_rewards - i - 1
        if dones[reveres_idx]:
            curr = 0
        curr = r + discount_factor * curr
        returns[reveres_idx] = curr  # append in reverse
    return returns


def calc_gaes(rewards, values, dones, discount_factor=0.99, decay=0.95):
    """
    works with rewards vector which consitst of many epidsodes
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    gamma = discount_factor
    device = rewards.device

    next_values = torch.cat([values[1:], torch.zeros(1, device=device)])
    # deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]
    deltas = rewards + gamma * next_values * (1 - dones) - values

    gaes = torch.zeros_like(deltas, device=device)
    gaes[-1] = deltas[-1]

    for i in reversed(range(len(deltas)-1)):

        gaes[i] = deltas[i] + gamma * decay * gaes[i+1] * (1 - dones[i])

    return gaes


class ObsShapeWraper(dict):
    def __init__(self, *args, **kwargs):
        try:
            super(ObsShapeWraper, self).__init__(*args, **kwargs)
        except TypeError:
            super(ObsShapeWraper, self).__init__({'data': tuple(*args)})          


class ObsWraper:
    def __init__(self, data=None, keep_dims=False, tensors=False):
        if data is None:
            return self._init_from_none_()
        if type(data) == list or type(data) == tuple:
            if type(data[0]) == ObsWraper:
                return self.init_from_list_np_obs(data)
            else:
                return self.init_from_list_generic_data(data)
        else:
            try:
                if type(data) == dict:
                    to_add = data
                elif type(data) == ObsWraper:
                    to_add = data.data
                else:
                    raise TypeError
                self.data = {}
                self.len = 0
                for k, v in to_add.items():
                    if tensors:
                        self.data[k] = v
                    else:
                        self.data[k] = np.array(v)
                    len_v = len(v)
                    if self.len == 0:
                        self.len = len_v
                    assert self.len == len_v, "cant init a dict with a value with different len"
            except TypeError:
                if keep_dims:
                    to_add = np.array(data)
                else:
                    to_add = np.array(data)[np.newaxis,:]
                self.data = {'data': to_add}
                self.len = len(to_add)


    def init_from_list_np_obs(self, obs_list):
        self.data = {}
        self.len = 0
        keys = list(obs_list[0].keys())
        for k in keys:
            res = []
            for obs in obs_list:
                res.append(obs[k])
            res = np.array(res)
            if res.shape[0] != 1:
                self.data[k] = np.squeeze(res, axis=1)
            else:
                self.data[k] = res
                
        self.len = len(obs_list)

    
    def init_from_list_generic_data(self, obs_list):
        self.data = {}
        self.len = 0
        res = np.array(obs_list)
        self.data['data'] = res
        self.len = len(res)

    def _init_from_none_(self):
        self.data = {}
        self.__init__(self.data)


    def __setitem__(self, key, value):
        if type(key) is str:
            self.data[key] = value
            len_v = len(value)
            if self.len == 0:
                self.len = len_v
            assert self.len == len_v, "cant set a value with differnet len"
        elif np.issubdtype(type(value), np.integer):
            for k in self.data.keys():
                self.data[k][key] = value
        else:
            assert self.data.keys() == value.keys(), "has to set item with same keys"
            for k in self.data.keys():
                self.data[k][key] = value[k]


    def __iter__(self):
        return iter(self.data)


    def __getitem__(self, key):
        if type(key) is str:
            return self.data[key]
        temp_dict = {}
        for k, v in self.data.items():

            if np.issubdtype(type(key), np.integer):
                temp_dict[k] = np.array([v.__getitem__(key)])
            else:
                temp_dict[k] = np.array(v.__getitem__(key))
        return ObsWraper(temp_dict)


    def keys(self):
        return self.data.keys()


    def items(self):
        return self.data.items()


    def values(self):
        return self.data.values()


    def __len__(self):
        return self.len


    def __str__(self) -> str:
        return self.data.__str__()


    def __repr__(self) -> str:
        return self.data.__repr__()


    def get_as_tensors(self, device='cpu'):
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = torch.tensor(v).float().to(device)
        return ObsWraper(temp_dict, keep_dims=True, tensors=True)


    def np_cat(self, other, axis=0):
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = np.concatenate([self.data[k], other[k]], axis)
        return ObsWraper(temp_dict)


    def np_append(self, other, axis=0):
        if self.len != 0:
            for k, v in self.data.items():
                self.data[k] = np.concatenate([self.data[k], other[k]], axis)
            self.len = self.len + len(other)
        else:
            self.data = copy.deepcopy(other.data)
            self.len = other.len


    # def torch_cat(self, other, axis=0):
    #     temp_dict = {}
    #     for k, v in self.data.items():
    #         temp_dict[k] = torch.cat([self.data[k], other[k]], axis)
    #     return ObsWraper(temp_dict)


    # def torch_append(self, other, axis=0):
    #     if self.len != 0:
    #         for k, v in self.data.items():
    #             self.data[k] = torch.cat([self.data[k], other[k]], axis)
    #         self.len = self.len + len(other)
    #     else:
    #         self.data = copy.deepcopy(other.data)
    #         self.len = other.len

    def np_roll(self, indx, axis=0, inplace=False):
        temp_dict = {}
        for k, v in self.data.items():
            temp_dict[k] = np.roll(self.data[k], indx, axis=axis)
        if inplace:
            self.data = temp_dict
        else:
            return ObsWraper(temp_dict)
        


class ExperienceReplay:
    # TODO: support continous infinte env by removeing long seeing states without dones
    def __init__(self, capacity, obs_shape, continous_mem=False):
        
        self.obs_shape = ObsShapeWraper(obs_shape)
        self.capacity = capacity
        self.init_buffers()
        self.continous_mem = continous_mem

    def __len__(self):
        return self.curr_size

    def append(self, curr_obs, action, reward, done, next_obs):
        # extra_exps
        
        curr_obs = ObsWraper(curr_obs)
        next_obs = ObsWraper(next_obs)
        
        num_samples = len(curr_obs)

        free_space = self.capacity - self.curr_size
        if num_samples > self.capacity:
            self.curr_size = 0
            curr_obs = curr_obs[:self.capacity]
            action = action[:self.capacity]
            reward = reward[:self.capacity]
            done = done[:self.capacity]
            next_obs = next_obs[:self.capacity]

            done[-1] = True
            num_samples = self.capacity

        elif self.__len__() + num_samples > self.capacity and not self.continous_mem:
            dones = np.where(self.all_buffers[self.dones_index] == True)[0]
            relevant_index = np.where(dones > num_samples - free_space)[0][0]
            done_index = dones[relevant_index]

            for i in range(len(self.all_buffers)):
                self.all_buffers[i][:done_index+1] = 0
                if i == self.states_index or i == self.next_states_index:
                    self.all_buffers[i] = self.all_buffers[i].np_roll(-done_index - 1, axis=0, inplace=False)
                else:
                    self.all_buffers[i] = np.roll(
                        self.all_buffers[i], -done_index - 1, axis=0)

            self.curr_size -= (done_index+1)

        self.all_buffers[self.states_index][self.curr_size:self.curr_size + num_samples] = curr_obs
        self.all_buffers[self.actions_index][self.curr_size:self.curr_size + num_samples] = action
        self.all_buffers[self.reward_index][self.curr_size:self.curr_size + num_samples] = reward
        self.all_buffers[self.dones_index][self.curr_size:self.curr_size + num_samples] = done
        self.all_buffers[self.next_states_index][self.curr_size: self.curr_size+num_samples] = next_obs

        self.curr_size += num_samples

    def init_buffers(self):

        self.curr_size = 0
        actions_buffer = np.zeros((self.capacity), dtype=np.int64)
        reward_buffer = np.zeros((self.capacity), dtype=np.float32)
        dones_buffer = np.zeros((self.capacity), dtype=np.uint8)

        states_buffer = ObsWraper()
        next_states_buffer = ObsWraper()
        for k in self.obs_shape:
            shape = (self.capacity, *self.obs_shape[k])
            states_buffer[k] = np.zeros(shape, dtype=np.float32)
            next_states_buffer[k] = np.zeros(shape, dtype=np.float32)

        self.all_buffers = [states_buffer, actions_buffer,
                            reward_buffer, dones_buffer, next_states_buffer]

        self.states_index = 0
        self.actions_index = 1
        self.reward_index = 2
        self.dones_index = 3
        self.next_states_index = 4

    def clear(self):
        self.init_buffers()

    def get_last_episodes(self, num_episodes):
        """return all last episode samples, or specified num samples"""
        episode_indices = [0]
        episode_indices.extend(np.where(self.all_buffers[self.dones_index] == True)[
                               0])  # last episode indx is done =1

        assert len(
            episode_indices) >= num_episodes, "requested more episodes then actual stored in mem"
        # it is a "False" done just for episode begin idx
        if episode_indices[-num_episodes - 1] == 0:
            num_samples = self.curr_size - episode_indices[-num_episodes - 1]
        else:  # we dont want the last done in our batch
            num_samples = self.curr_size - \
                episode_indices[-num_episodes - 1] - \
                1  # exclude first done indice
        return self.get_last_samples(num_samples)


    def get_last_samples(self, num_samples=None):
        """return all last episode samples, or specified num samples"""
        if num_samples is None:
            "return last episode"

            dones = np.where(self.all_buffers[self.dones_index] == True)[0]
            if len(dones) > 1:
                # exclude the latest done sample
                first_sample_idx = dones[-2] + 1
            else:
                # from 0 index to last done(which is also the first..)
                first_sample_idx = 0
                last_samples = [buff[first_sample_idx:self.curr_size]
                                for buff in self.all_buffers]

        else:
            last_samples = [buff[self.curr_size-num_samples:self.curr_size]
                            for buff in self.all_buffers]

        return last_samples

    def get_buffers_at(self, indices):
        buffers_at = (buff[indices] for buff in self.all_buffers)
        return buffers_at

    def sample_random_batch(self, sample_size):
        indices = np.random.choice(self.curr_size, sample_size, replace=False)
        return self.get_buffers_at(indices)


def worker(env, conn, idx):
    proc_running = True
    env.reset()

    while proc_running:
        cmd, msg = conn.recv()

        if (cmd == "step"):
            next_state, reward, done, _ = env.step(msg)
            if done:
                next_state = env.reset()
            conn.send((next_state, reward, done, _))

        elif (cmd == "reset"):
            next_state = env.reset()
            conn.send(next_state)

        elif (cmd == "clear_env"):
            next_state = env.clear_env()
            conn.send(next_state)

        elif (cmd == "step_generator"):
            next_state, reward, done, _ = env.step_generator(msg)
            conn.send((next_state, reward, done, _))

        elif(cmd == "get_env"):
            conn.send(env)

        elif (cmd == "close"):
            proc_running = False
            conn.close()

        elif (cmd == "sample_random_state"):
            state = env.sample_random_state()
            conn.send(state)

        elif (cmd == "change_env"):
            env = msg

        else:
            raise Exception("Command not implemented")


class ParallelEnv():
    def __init__(self, env, num_envs):
        self.num_envs = num_envs
        if num_envs > 1:
            self.p_env = ParallelEnv_m(env, num_envs)
        else:
            self.p_env = SingleEnv_m(env)

    def __del__(self):
        self.p_env.close_procs()
    
    def change_env(self, env):
        self.p_env.change_env(env)

    def get_envs(self):
        if self.num_envs == 1:
            return [self.p_env.env]
        else:
            return self.p_env.get_envs()

    def reset(self):
        return self.p_env.reset()

    def step(self, actions):
        return self.p_env.step(actions)

    def step_generator(self, actions):
        return self.p_env.step_generator(actions)

    def clear_env(self):
        return self.p_env.clear_env()

    def close_procs(self):
        self.p_env.close_procs()


class ParallelEnv_m():
    def __init__(self, env, num_envs):

        self.num_envs = num_envs
        self.process = namedtuple("Process", field_names=[
                                  "proc", "connection"])

        self.comm = []
        for idx in range(self.num_envs):
            parent_conn, worker_conn = Pipe()
            proc = Process(target=worker, args=(env, worker_conn, idx))
            proc.start()
            self.comm.append(self.process(proc, parent_conn))

    def change_env(self, env):
        [p.connection.send(("change_env", env)) for p in self.comm]

    def get_envs(self):
        [p.connection.send(("get_env", "")) for p in self.comm]
        res = [p.connection.recv() for p in self.comm]
        return res

    def reset(self):
        [p.connection.send(("reset", "")) for p in self.comm]
        res = [p.connection.recv() for p in self.comm]
        return res

    def step(self, actions):
        # send actions to envs
        [p.connection.send(("step", action)) for i, p, action in zip(
            range(self.num_envs), self.comm, actions)]

        # Receive response from envs.
        res = [p.connection.recv() for p in self.comm]
        next_states, rewards, dones, _ = zip(*res)
        rewards = np.array(rewards)
        dones = np.array(dones)

        return next_states, rewards, dones, np.array(_)

    def step_generator(self, actions):
        # send actions to envs
        [p.connection.send(("step_generator", action)) for i, p, action in zip(
            range(self.num_envs), self.comm, actions)]

        # Receive response from envs.
        res = [p.connection.recv() for p in self.comm]
        next_states, rewards, dones, _ = zip(*res)
        rewards = np.array(rewards)
        dones = np.array(dones)
        return next_states, rewards, dones, np.array(_)

    def clear_env(self):
        [p.connection.send(("clear_env", "")) for p in self.comm]
        res = [p.connection.recv() for p in self.comm]
        return res

    def sample_random_state(self):
        [p.connection.send(("sample_random_state", "")) for p in self.comm]
        res = [p.connection.recv() for p in self.comm]
        return

    def close_procs(self):
        [p.connection.send(("close", "")) for p in self.comm]


class SingleEnv_m():
    def __init__(self, env):
        self.env = copy.deepcopy(env)

    def change_env(self, env):
        self.env = env

    def reset(self):
        return [self.env.reset()]

    def step(self, actions):
        next_states, rewards, dones, _ = self.env.step(actions)
        next_states = next_states[np.newaxis, :]
        rewards = np.array(rewards).reshape(1, 1)
        dones = np.array(dones).reshape(1, 1)
        return next_states, rewards, dones, _

    def step_generator(self, actions):
        next_states, rewards, dones, _ = self.env.step_generator(actions)
        next_states = next_states[np.newaxis, :]
        rewards = np.array(rewards).reshape(1, 1)
        dones = np.array(dones).reshape(1, 1)
        return next_states, rewards, dones, _

    def clear_env(self):
        return [self.env.clear_env()]

    def sample_random_state(self):
        return [self.env.sample_random_state()]

    def close_procs(self):
        pass
