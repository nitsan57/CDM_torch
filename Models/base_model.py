import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np
from Agents.agent_utils import ObsShapeWraper


class AbstractModel(torch.nn.Module, ABC):
    is_rnn=False

    def __init__(self, input_shape, out_shape):
        super().__init__()
    
        self.input_shape = ObsShapeWraper(input_shape)


        self.out_shape = np.array(out_shape)
        
        if len(self.out_shape.shape) == 0:
            self.out_shape = self.out_shape.reshape((1,))


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