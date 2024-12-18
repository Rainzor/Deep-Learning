import torch
import torch.nn as nn
from torch.nn import Parameter
import math

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Initialize the LSTM cell.

        Args:
            input_size (int): The number of expected features in the input.
            hidden_size (int): The number of features in the hidden state.
        """
        super(LSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input-to-hidden weights for input, forget, cell, and output gates
        self.W_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        
        # Hidden-to-hidden weights for input, forget, cell, and output gates
        self.W_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        
        # Bias terms for input, forget, cell, and output gates
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights and biases using a uniform distribution.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, hidden):
        """
        Forward pass for the LSTM cell.

        Args:
            x (Tensor): Input tensor at the current time step (batch_size, input_size).
            hidden (tuple): Tuple of (h_prev, c_prev), each of shape (batch_size, hidden_size).

        Returns:
            h_next (Tensor): Next hidden state (batch_size, hidden_size).
            c_next (Tensor): Next cell state (batch_size, hidden_size).
        """
        h_prev, c_prev = hidden

        # Compute all gate activations in a single matrix multiplication for efficiency
        gates = (torch.mm(x, self.W_ih.t()) + self.b_ih +
                 torch.mm(h_prev, self.W_hh.t()) + self.b_hh)
        
        # Split the gates into their respective components
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
        
        # Apply activations
        i = torch.sigmoid(i_gate)  # Input gate
        f = torch.sigmoid(f_gate)  # Forget gate
        g = torch.tanh(g_gate)     # Cell gate
        o = torch.sigmoid(o_gate)  # Output gate
        
        # Compute the next cell state
        c_next = f * c_prev + i * g
        
        # Compute the next hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 batch_first=False, dropout=0.0, bidirectional=False):
        """
        Initialize the LSTM module.

        Args:
            input_size (int): The number of expected features in the input.
            hidden_size (int): The number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            batch_first (bool): If True, input and output tensors are provided as (batch, seq, feature).
            dropout (float): If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer.
            bidirectional (bool): If True, becomes a bidirectional LSTM.
        """
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Initialize multiple layers of LSTMCell for each direction
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                self.cells.append(LSTMCell(layer_input_size, hidden_size))
        
        # Dropout layer to be applied between layers (except after the last layer)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize all LSTM cells.
        """
        for cell in self.cells:
            cell.reset_parameters()

    def forward(self, x, h0=None):
        """
        Forward pass for the LSTM.

        Args:
            x (Tensor): Input tensor
                      (batch_size, seq_len, input_size) if batch_first=True,
                      or (seq_len, batch_size, input_size) if batch_first=False.
            h0 (tuple, optional): Tuple of (h_0, c_0) containing the initial hidden and cell states.
                                  Each has shape (num_layers * num_directions, batch_size, hidden_size).
                                  If not provided, defaults to zeros.

        Returns:
            output (Tensor): Output features from the last layer of the LSTM for each time step.
                             Shape: (batch_size, seq_len, hidden_size * num_directions) if batch_first=True,
                                    or (seq_len, batch_size, hidden_size * num_directions) otherwise.
            (hn, cn) (tuple of Tensors): Final hidden and cell states for each layer and direction.
                                         Each has shape (num_layers * num_directions, batch_size, hidden_size).
        """
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)  # Convert to (batch_size, seq_len, input_size)
        
        # If no initial hidden state is provided, initialize to zeros
        if h0 is None:
            h0 = (x.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size),
                  x.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
        else:
            h0 = (h0[0].contiguous(), h0[1].contiguous())
        
        h0, c0 = h0
        
        # Initialize a list to hold the hidden and cell states for each layer and direction
        # h_t[layer][direction] = (hidden, cell) tuple (batch_size, hidden_size)
        h_t = []
        for layer in range(self.num_layers):
            layer_hidden = []
            for direction in range(self.num_directions):
                idx = layer * self.num_directions + direction
                layer_hidden.append((h0[idx], c0[idx]))
            h_t.append(layer_hidden)
        
        # Store all outputs across the sequence
        outputs = None  # Will hold the output of the current layer
        
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
            h_forward, c_forward = h_t[layer][0]
                # Iterate over each time step
            for t in range(seq_len):
                input_t = layer_input[:, t, :]  # (batch_size, input_size)
                h_forward, c_forward = forward_cell(input_t, (h_forward, c_forward))
                forward_output.append(h_forward.unsqueeze(1))  # (batch_size, 1, hidden_size)
            
            # Backward direction
            if self.bidirectional:
                h_backward, c_backward = h_t[layer][1]
                for t in reversed(range(seq_len)):
                    input_t = layer_input[:, t, :]  # (batch_size, input_size)
                    h_backward, c_backward = backward_cell(input_t, (h_backward, c_backward))
                    backward_output.insert(0, h_backward.unsqueeze(1))  # Prepend to maintain order
            
            # Concatenate forward and backward outputs
            if self.bidirectional:
                forward_cat = torch.cat(forward_output, dim=1)  # (batch_size, seq_len, hidden_size)
                backward_cat = torch.cat(backward_output, dim=1)  # (batch_size, seq_len, hidden_size)
                layer_output = torch.cat([forward_cat, backward_cat], dim=2)  # (batch_size, seq_len, hidden_size * 2)
            else:
                layer_output = torch.cat(forward_output, dim=1)  # (batch_size, seq_len, hidden_size)
            
            # Apply dropout except for the last layer
            if self.dropout > 0 and layer < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            
            # Update outputs for the next layer
            outputs = layer_output
            
            # Update hidden states
            if self.bidirectional:
                h_t[layer][0] = (h_forward, c_forward)
                h_t[layer][1] = (h_backward, c_backward)
            else:
                h_t[layer][0] = (h_forward, c_forward)
        
        # Final output
        output = outputs  # Shape: (batch_size, seq_len, hidden_size * num_directions)
        
        # If batch_first=False, transpose back to (seq_len, batch_size, hidden_size * num_directions)
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        # Prepare final hidden states
        hn = []
        cn = []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                h, c = h_t[layer][direction]
                hn.append(h)
                cn.append(c)
        hn = torch.stack(hn, dim=0)  # Shape: (num_layers * num_directions, batch_size, hidden_size)
        cn = torch.stack(cn, dim=0)  # Shape: (num_layers * num_directions, batch_size, hidden_size)
        
        return output, (hn, cn)
