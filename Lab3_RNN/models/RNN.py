import torch
import torch.nn as nn
from torch.nn import Parameter
import math

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weights for input-to-hidden and hidden-to-hidden connections
        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))  # Weight for input to hidden
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # Weight for hidden to hidden
        
        # Bias terms for input-to-hidden and hidden-to-hidden connections
        self.b_ih = nn.Parameter(torch.Tensor(hidden_size))  # Bias for input to hidden
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))  # Bias for hidden to hidden
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights using the same method as PyTorch's RNN.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, h_prev):
        """
        Args:
        - x: Current input at time t (batch_size, input_size)
        - h_prev: Previous hidden state (batch_size, hidden_size)
        
        Returns:
        - h_next: Current hidden state (batch_size, hidden_size)
        """
        h_next = torch.tanh(
            torch.mm(x, self.W_ih.t()) + self.b_ih +
            torch.mm(h_prev, self.W_hh.t()) + self.b_hh
        )
        return h_next


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        
        # Initialize multiple layers of RNNCell
        self.cells = nn.ModuleList([
            RNNCell(input_size if i == 0 else hidden_size, hidden_size) 
            for i in range(num_layers)
        ])
        
        # Dropout layer to be applied between layers (except after the last layer)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # Initialize parameters using the reset method
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights for the entire RNN using the same method as PyTorch's RNN.
        """
        for cell in self.cells:
            cell.reset_parameters()

    def forward(self, x, h0=None):
        """
        Args:
        - x: Input tensor 
             (batch_size, seq_len, input_size) if batch_first=True, 
             or (seq_len, batch_size, input_size) if batch_first=False
        - h0: Initial hidden state (num_layers, batch_size, hidden_size), or None (initialized to zero)
        
        Returns:
        - output: Output for each time step 
                  (batch_size, seq_len, hidden_size) if batch_first=True,
                  or (seq_len, batch_size, hidden_size) if batch_first=False
        - hn: Final hidden state for each layer (num_layers, batch_size, hidden_size)
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)  # Convert to (batch_size, seq_len, input_size)
        
        # If no initial hidden state is provided, initialize to zeros
        if h0 is None:
            h0 = x.new_zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            h0 = h0.contiguous()
        
        # Initialize a list to hold the hidden states for each layer
        h_t = [h0[layer] for layer in range(self.num_layers)]
        
        # Store all outputs across the sequence
        outputs = []
        
        # Iterate over each time step
        for t in range(seq_len):
            input_t = x[:, t, :]  # Input at time step t
            
            # Iterate over each layer
            for layer in range(self.num_layers):
                # Current layer's input is the output from the previous layer
                # For the first layer, it's the input at the current time step
                current_input = input_t if layer == 0 else h_t[layer - 1]
                
                # Compute the next hidden state
                h_t[layer] = self.cells[layer](current_input, h_t[layer])
                
                # Apply dropout except for the last layer
                if self.dropout > 0 and layer < self.num_layers - 1:
                    h_t[layer] = self.dropout_layer(h_t[layer])
            
            # The output at time t is the hidden state of the last layer
            outputs.append(h_t[-1].unsqueeze(1))
        
        # Concatenate all time steps
        output = torch.cat(outputs, dim=1)  # Shape: (batch_size, seq_len, hidden_size)
        
        # If batch_first=False, transpose back to (seq_len, batch_size, hidden_size)
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        # Stack the final hidden states from all layers
        hn = torch.stack(h_t, dim=0)  # Shape: (num_layers, batch_size, hidden_size)
        
        return output, hn
