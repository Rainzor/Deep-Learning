import torch
import torch.nn as nn
import numpy as np
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def trunc_normal(tensor, mean=0., std=1., a=-2., b=2.):
    """In-place truncated normal initialization."""
    with torch.no_grad():
        # Draw random numbers from normal distribution
        tensor.normal_(mean, std)
        
        # Truncate values
        while True:
            mask = (tensor < a) | (tensor > b)
            if not mask.any():
                break
            tensor[mask] = torch.normal(mean, std, size=mask.sum().item())