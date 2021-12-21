import torch
import numpy as np
import torch.nn as nn

class CNN(AbstractModel):
    def __init__(self, input_shape, out_shape, embed_dim = 64, repeat=1):
        super().__init__(input_shape, out_shape)
        
        c,h,w = input_shape
        input_c = c
        out_c = 6
        num_convs = 5
        self.conv = []
        for i in range(num_convs):
            self.conv.append(nn.Sequential(nn.Conv2d(input_c, out_c, 3, 1, 1, padding_mode='reflect'), nn.ReLU()))
            input_c = out_c
            out_c +=10
        self.conv = nn.Sequential(*self.conv)

        fc_input_size = np.prod(out_c*h*w)
        self.l1 = nn.Sequential(nn.Linear(fc_input_size, embed_dim), nn.ReLU())
        embed_layer = [nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.ReLU()) for i in range(repeat)]
        self.l2 = nn.Sequential(*embed_layer)
        self.l3 = nn.Sequential(nn.Linear(embed_dim, out_shape))


    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

    def reset(self):
        pass

    def update_hidden_state(self, indices):
    """if not rnn - impl can be just: pass, otherwise updates hidden state for finished environments"""
        raise NotImplementedError