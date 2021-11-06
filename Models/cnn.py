import torch
import numpy as np
import torch.nn as nn

class CNN(torch.nn.Module):
    def __init__(self, obs_shape, n_actions, embed_dim = 50, repeat=7, softmax=False):
        super().__init__()
        
        c,h,w = obs_shape
        input_c = c
        out_c = 6
        num_convs = 5
        self.conv = []
        for i in range(num_convs):
            self.conv.append(nn.Sequential(nn.Conv2d(input_c, out_c, 3, 1, 1, padding_mode='reflect'), nn.GELU()))
            input_c = out_c
            out_c +=10
        self.conv = nn.Sequential(*self.conv)

        fc_input_size = np.prod(out_c*h*w)
        self.l1 = nn.Sequential(nn.Linear(fc_input_size, embed_dim), nn.GELU())
        embed_layer = [nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.GELU()) for i in range(repeat)]
        self.l2 = nn.Sequential(*embed_layer)
        if not softmax:
            self.l3 = nn.Sequential(nn.Linear(embed_dim, n_actions))
        else:
            self.l3 = nn.Sequential(nn.Linear(embed_dim, n_actions), nn.Softmax())


    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x