import torch
import torch.nn as nn
import numpy as np
from .base_model import AbstractModel

class RNN(AbstractModel):
    def __init__(self, input_shape, out_shape, hidden_dim = 64, num_grus=2):
        super().__init__(input_shape, out_shape)
        input_size = np.prod(input_shape)
        self.hidden_state = None
        self.num_grus = num_grus
        self.l1 = nn.GRU(input_size, hidden_dim, num_layers=num_grus, batch_first=True)
        
        self.l2 = nn.Sequential(nn.Linear(hidden_dim, out_shape))


    def forward(self, x):
        """x is padded pack"""

        # lengths = x.batch_sizes
        # if self.hidden_state is not None:
        #     print("!",x.shape, self.hidden_state.shape, hex(id(self.hidden_state)))
        # else:
        #     print("!",x.shape, None)
        out,h = self.l1(x, self.hidden_state)
        self.hidden_state = h.detach()
        # print("after net",out.shape, h.shape, h.detach().shape , hex(id(self.hidden_state)))

        out = out[:,-1]

        # padded_output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # last_seq_items = padded_output[torch.arange(padded_output.shape[0]),output_lens-1,:]

        out = torch.flatten(out, start_dim=1)
        out = self.l2(out)
        return out


    def reset(self):
        self.hidden_state = None

    def update_hidden_state_indices(self, indices):
        # print("befirre",self.hidden_state.shape, hex(id(self.hidden_state)))
        self.hidden_state = self.hidden_state[:,indices,:]
        # print("after",self.hidden_state.shape, hex(id(self.hidden_state)))