import torch
from torch._C import device
import torch.nn as nn
import numpy as np
from .base_model import AbstractModel

class RNN(AbstractModel):
    def __init__(self, input_shape, out_shape, hidden_dim = 64, num_grus=2):
        super().__init__(input_shape, out_shape)
        input_size = np.prod(input_shape)
        self.hidden_dim = hidden_dim
        self.hidden_state = None
        self.num_grus = num_grus
        self.l1 = nn.GRU(input_size, hidden_dim, num_layers=num_grus, batch_first=True)
        
        self.l2 = nn.Linear(hidden_dim, out_shape)


    def forward(self, x):
        """x is padded pack"""
        # batch_Size, seq_len = x.shape[:2]
        # flatten_dim_obs = np.prod(x.shape[2:])


        # seq_len = x.data.shape[1]
        # x = torch.flatten(x, 2)
        device = x.data.get_device()

        # print(x.data.shape)
        out,h = self.l1(x, self.hidden_state)
        self.hidden_state = h.detach()
        # self.hidden_state = hs
        #REG:
        # out = out[:,-1]
        # PACK PADD RNN:
        padded_output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # last_seq_items = padded_output[torch.arange(padded_output.shape[0]),output_lens-1,:]
        # if output_lens[0] != 1:
        #     import pdb
        #     pdb.set_trace()
        batch_Size = padded_output.shape[0]
        all_outs_len= torch.sum(output_lens)
        relevant_flatten_out = torch.zeros((all_outs_len, *padded_output.shape[2:]), device=device)
        last_idx = 0
        for i, out_len in enumerate(output_lens):
            curr_idx = last_idx+out_len
            relevant_flatten_out[last_idx:curr_idx] = padded_output[i][:out_len]
            last_idx = curr_idx
                

        
        out = self.l2(relevant_flatten_out)
        # out = out.reshape(batch_Size*all_outs_shape, self.hidden_dim)
        # out = self.l2(out)
        return out


    def reset(self):
        self.hidden_state = None

    # def update_hidden_state_indices(self, indices):
    #     if self.hidden_state is not None:
    #         self.hidden_state = self.hidden_state[:,indices,:]
