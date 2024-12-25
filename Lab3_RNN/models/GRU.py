import torch
import torch.nn as nn
from torch.nn import Parameter
import math
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Combine weights for update, reset, and new gates
        self.W_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(3 * hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W_ih, -stdv, stdv)
        nn.init.uniform_(self.W_hh, -stdv, stdv)
        nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x, h_prev):
        # Linear transformations
        gates = F.linear(x, self.W_ih, self.bias) + F.linear(h_prev, self.W_hh)
        
        # Split into update, reset, and new gates
        z_t, r_t, n_t = gates.chunk(3, dim=1)
        
        # Apply activations
        z_t = torch.sigmoid(z_t)
        r_t = torch.sigmoid(r_t)
        n_t = torch.tanh(n_t + r_t * F.linear(h_prev, self.W_hh, self.bias))  # Adjusted for the new gate
        
        # Compute next hidden state
        h_next = (1 - z_t) * n_t + z_t * h_prev
        
        return h_next
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 batch_first=False, dropout=0.0, bidirectional=False):
        """
        Initialize the GRU module.

        Args:
            input_size (int): The number of expected features in the input.
            hidden_size (int): The number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            batch_first (bool): If True, input and output tensors are provided as (batch, seq, feature).
            dropout (float): If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer.
            bidirectional (bool): If True, becomes a bidirectional GRU.
        """
        super(GRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Initialize multiple layers of GRUCell for each direction
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                self.cells.append(GRUCell(layer_input_size, hidden_size))
        
        # Dropout layer to be applied between layers (except after the last layer)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # Initialize parameters using the reset method
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights for the entire GRU using the same method as PyTorch's GRU.
        """
        for cell in self.cells:
            cell.reset_parameters()

    def forward(self, x, h0=None):
        """
        Forward pass for the GRU.

        Args:
            x (Tensor): Input tensor 
                 (batch_size, seq_len, input_size) if batch_first=True, 
                 or (seq_len, batch_size, input_size) if batch_first=False
            h0 (Tensor, optional): Initial hidden state 
                 (num_layers * num_directions, batch_size, hidden_size), or None (initialized to zero)
        
        Returns:
            output (Tensor): Output for each time step 
                      (batch_size, seq_len, hidden_size * num_directions) if batch_first=True,
                      or (seq_len, batch_size, hidden_size * num_directions) if batch_first=False
            hn (Tensor): Final hidden state for each layer and direction 
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
            
            # Determine the input for the current layer
            if layer == 0:
                layer_input = x
            else:
                layer_input = outputs  # Output from the previous layer
            
            # Forward direction
            h_forward = h_t[layer][0]
            for t in range(seq_len):
                input_t = layer_input[:, t, :]  # (batch_size, input_size)
                h_forward = forward_cell(input_t, h_forward)
                forward_output.append(h_forward.unsqueeze(1))  # (batch_size, 1, hidden_size)
            
            # Backward direction
            if self.bidirectional:
                h_backward = h_t[layer][1]
                for t in reversed(range(seq_len)):
                    input_t = layer_input[:, t, :]  # (batch_size, input_size)
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
