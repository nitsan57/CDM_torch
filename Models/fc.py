import torch
import numpy as np
import torch.nn as nn
from .base_model import AbstractModel
import gym 

class FC(AbstractModel):
    
    def __init__(self, input_shape, out_shape, embed_dim = 128, repeat=3):
        super().__init__(input_shape, out_shape)
        input_size = 0
        
        # if type(input_shape) == dict or type(input_shape) == gym.spaces.dict.Dict:
        #     for k,val in input_shape.items():
        #         input_size += np.prod(val.shape)
        # else:
        self.input_size_dict = {k: np.prod(self.input_shape[k]) for k in self.input_shape}
        self.num_inputs = len(self.input_size_dict)

        self.l1 =torch.nn.ModuleDict({k:nn.Sequential(nn.Linear(input_size, embed_dim), nn.ReLU()) for k,input_size in self.input_size_dict.items()})
        self.embed_layer = torch.nn.ModuleDict({k:nn.Sequential(*[nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.ReLU()) for i in range(repeat)])  for k in self.input_size_dict})
        self.concat_layer = nn.Sequential(nn.Linear(embed_dim * self.num_inputs, embed_dim), torch.nn.ReLU())
        self.l2 = nn.Linear(embed_dim, out_shape)
    

    def forward(self, x):
        # temp_k = list(x.keys())[0]
        # device = x[temp_k].data.get_device()

        res_dict = dict()
        for k in x:
            layer = self.l1[k]
            layer_in = torch.flatten(x[k], start_dim=1)
            out = layer(layer_in)
            res_dict[k] = out

        for k in x:
            layer = self.embed_layer[k]
            out = layer(res_dict[k])
            res_dict[k] = out

        res = torch.cat(list(res_dict.values()))
        res = self.concat_layer(res)
        out = self.l2(res)
        return out


    def reset(self):
        pass
