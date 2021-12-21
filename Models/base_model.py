import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class AbstractModel(torch.nn.Module, ABC):
    def __init__(self, input_shape, out_shape):
        super().__init__()
        self.input_shape = input_shape
        self.out_shape = out_shape

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """if not rnn - impl can be just: pass, otherwise resets hidden state"""
        raise NotImplementedError

    def update_hidden_state_indices(self, indices):
        """if not rnn - impl can be just: pass, otherwise updates hidden state for finished environments"""
        pass