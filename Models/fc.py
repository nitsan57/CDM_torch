import torch
import numpy as np
import torch.nn as nn

class FC(torch.nn.Module):
    def __init__(self, obs_shape, n_actions, embed_dim = 50, repeat=1, softmax=False):
        super().__init__()
        input_size = np.prod(obs_shape)

        self.l1 = nn.Sequential(nn.Linear(input_size, embed_dim), nn.GELU())
        embed_layer = [nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.GELU()) for i in range(repeat)]
        self.l2 = nn.Sequential(*embed_layer)
        if not softmax:
            self.l3 = nn.Sequential(nn.Linear(embed_dim, n_actions))
        else:
            self.l3 = nn.Sequential(nn.Linear(embed_dim, n_actions), nn.Softmax())


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x