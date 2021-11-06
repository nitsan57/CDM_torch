# import torch
# import torch.nn as nn
# from abc import ABC, abstractmethod

# class FC(torch.nn.Module, ABC):
#     def __init__(self, obs_shape, n_actions, embed_dim = 50, repeat=7, softmax=False):
#         super().__init__()
#         self.obs_shape = obs_shape
#         self.n_actions = n_actions
#         self.softmax=softmax

#     @abstractmethod
#     def forward(self, x):
#         raise NotImplementedError