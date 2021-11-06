import torch
import torch.nn as nn
import numpy as np

class RNN(torch.nn.Module):
    def __init__(self, obs_size, n_actions, hidden_dim = 50, num_grus=2, softmax=False):
        super().__init__()
        input_size = np.prod(obs_size)

        self.l1 = nn.Sequential(nn.GRU(input_size, hidden_dim, num_layers=num_grus, batch_first=True), nn.GELU())
        if not softmax:
            self.l2 = nn.Sequential(nn.Linear(hidden_dim, n_actions))
        else:
            self.l2 = nn.Sequential(nn.Linear(hidden_dim, n_actions), nn.Softmax())


    def forward(self, x):
        b,obs,data = x.shape
        x = torch.flatten(x, start_dim=2)
        x,h = self.l1(x)
        x = self.l2(x)
        return x