import torch
import torch.nn as nn
from torch.nn import Parameter
import math

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined weight for input and hidden states
        self.W = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W, -stdv, stdv)
        nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, h_prev):
        # Concatenate input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)
        # Linear transformation followed by tanh activation
        h_next = torch.tanh(F.linear(combined, self.W, self.bias))
        return h_next

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 batch_first=False, dropout=0.0, bidirectional=False):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Initialize multiple layers of RNNCell for each direction
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                self.cells.append(RNNCell(layer_input_size, hidden_size))
        
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
        - h0: Initial hidden state 
             (num_layers * num_directions, batch_size, hidden_size), or None (initialized to zero)
        
        Returns:
        - output: Output for each time step 
                  (batch_size, seq_len, hidden_size * num_directions) if batch_first=True,
                  or (seq_len, batch_size, hidden_size * num_directions) if batch_first=False
        - hn: Final hidden state for each layer and direction 
              (num_layers * num_directions, batch_size, hidden_size)
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)  # Convert to (batch_size, seq_len, input_size)
        
        # If no initial hidden state is provided, initialize to zeros
        if h0 is None:
            h0 = x.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        else:
            h0 = h0.contiguous()
        
        # Initialize a list to hold the hidden states for each layer and direction
        # h_t[layer][direction] = hidden state tensor  (batch_size, hidden_size)
        h_t = []
        for layer in range(self.num_layers):
            layer_hidden = []
            for direction in range(self.num_directions):
                idx = layer * self.num_directions + direction
                layer_hidden.append(h0[idx])
            h_t.append(layer_hidden)
        
        # Store all outputs across the sequence
        outputs = []
        
        # Iterate over each layer
        for layer in range(self.num_layers):
            # Extract forward and backward cells for the current layer
            forward_cell = self.cells[layer * self.num_directions + 0]
            backward_cell = self.cells[layer * self.num_directions + 1] if self.bidirectional else None
            
            # Initialize outputs for forward and backward directions
            forward_output = []
            backward_output = []
            
            # Forward direction
            h_forward = h_t[layer][0]
            for t in range(seq_len):
                input_t = x[:, t, :] if layer == 0 else outputs[:, t, :]
                h_forward = forward_cell(input_t, h_forward)
                forward_output.append(h_forward.unsqueeze(1))  # (batch_size, 1, hidden_size)
            
            # Backward direction
            if self.bidirectional:
                h_backward = h_t[layer][1]
                for t in reversed(range(seq_len)):
                    input_t = x[:, t, :] if layer == 0 else outputs[:, t, :]
                    h_backward = backward_cell(input_t, h_backward)
                    backward_output.insert(0, h_backward.unsqueeze(1))  # Prepend to maintain order
            
            # Concatenate forward and backward outputs
            if self.bidirectional:
                layer_output = torch.cat([torch.cat(forward_output, dim=1), 
                                          torch.cat(backward_output, dim=1)], dim=2)  # (batch_size, seq_len, hidden_size * 2)
            else:
                layer_output = torch.cat(forward_output, dim=1)  #  (batch_size, seq_len, hidden_size)
            
            # Apply dropout except for the last layer
            if self.dropout > 0 and layer < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            
            # Update outputs for the next layer
            outputs = layer_output
        
        # Final output
        output = outputs  # Shape: (batch_size, seq_len, hidden_size * num_directions)
        
        # If batch_first=False, transpose back to (seq_len, batch_size, hidden_size * num_directions)
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        # Prepare final hidden states
        hn = []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                hn.append(h_t[layer][direction])
        hn = torch.stack(hn, dim=0)  # Shape: (num_layers * num_directions, batch_size, hidden_size)
        
        return output, hn
