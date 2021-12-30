import torch
import numpy as np
import torch.nn as nn
from .base_model import AbstractModel

class FC(AbstractModel):
    def __init__(self, input_shape, out_shape, embed_dim = 64, repeat=3):
        super().__init__(input_shape, out_shape)
        input_size = 0
        if type(input_shape) == dict:
            for k,val in input_shape.items():
                input_size += np.prod(val)
        else:
            input_size = np.prod(input_shape)

        self.l1 = nn.Sequential(nn.Linear(input_size, embed_dim), nn.ReLU())
        embed_layer = [nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.ReLU()) for i in range(repeat)]
        self.l2 = nn.Sequential(*embed_layer)

        self.l3 = nn.Linear(embed_dim, out_shape)
    

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x
    
    def reset(self):
        pass