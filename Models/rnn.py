import torch
from torch._C import device
import torch.nn as nn
import numpy as np
from .base_model import AbstractModel

class RNN(AbstractModel):
    is_rnn = True
    def __init__(self, input_shape, out_shape, hidden_dim = 128, num_grus=3):
        super().__init__(input_shape, out_shape)
        input_size = np.prod(input_shape)
        self.hidden_dim = hidden_dim
        self.hidden_state = None
        self.num_grus = num_grus
        self.l1 = nn.GRU(input_size, hidden_dim, num_layers=num_grus, batch_first=True)
        self.l1.flatten_parameters()
        self.l2 = nn.Linear(hidden_dim, out_shape)


    def forward(self, x):
        """x is padded pack"""

        device = x.data.get_device()
        self.l1.flatten_parameters()
        out,h = self.l1(x, self.hidden_state)
        self.hidden_state = h.detach()
        padded_output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        all_outs_len= torch.sum(output_lens)
        relevant_flatten_out = torch.zeros((all_outs_len, *padded_output.shape[2:]), device=device)
        last_idx = 0
        for i, out_len in enumerate(output_lens):
            curr_idx = last_idx+out_len
            relevant_flatten_out[last_idx:curr_idx] = padded_output[i][:out_len]
            last_idx = curr_idx
        
        out = self.l2(relevant_flatten_out)
        return out


    def reset(self):
        self.hidden_state = None