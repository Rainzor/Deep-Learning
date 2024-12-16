import torch
import torch.nn as nn
from torch.nn import Parameter
import math

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False):
        super(CustomLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create LSTM layers
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            lstm_layer = LSTMCell(layer_input_size, hidden_size, bias, self.num_directions)
            self.layers.append(lstm_layer)
        
        # Dropout layer (applied between LSTM layers, except the last layer)
        if self.dropout > 0 and self.num_layers > 1:
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            self.dropout_layer = None

    def forward(self, input, hx=None):
        """
        input: (seq_len, batch, input_size) if batch_first=False
               (batch, seq_len, input_size) if batch_first=True
        hx: Tuple of (h_0, c_0)
            h_0: (num_layers * num_directions, batch, hidden_size)
            c_0: (num_layers * num_directions, batch, hidden_size)
        """
        if self.batch_first:
            input = input.transpose(0, 1)  # (seq_len, batch, input_size)
        
        seq_len, batch, _ = input.size()
        
        if hx is None:
            h_0 = input.new_zeros(self.num_layers * self.num_directions, batch, self.hidden_size, requires_grad=False)
            c_0 = input.new_zeros(self.num_layers * self.num_directions, batch, self.hidden_size, requires_grad=False)
        else:
            h_0, c_0 = hx
            if h_0.size(0) != self.num_layers * self.num_directions or \
               c_0.size(0) != self.num_layers * self.num_directions:
                raise ValueError("Initial hidden states have incorrect dimensions")
        
        h_n = []
        c_n = []
        outputs = input
        
        for layer in range(self.num_layers):
            layer_outputs = []
            layer_h = []
            layer_c = []
            for direction in range(self.num_directions):
                layer_idx = layer * self.num_directions + direction
                h_prev = h_0[layer_idx]
                c_prev = c_0[layer_idx]
                
                # If bidirectional and direction == 1, reverse the input sequence
                if self.bidirectional and direction == 1:
                    inputs_to_process = torch.flip(outputs, [0])
                else:
                    inputs_to_process = outputs
                
                # Pass through LSTMCell
                lstm_cell = self.layers[layer]
                output, (h_new, c_new) = lstm_cell(inputs_to_process, (h_prev, c_prev), direction)
                
                # If bidirectional and direction == 1, reverse back the output
                if self.bidirectional and direction == 1:
                    output = torch.flip(output, [0])
                
                layer_outputs.append(output)
                layer_h.append(h_new)
                layer_c.append(c_new)
            
            # Concatenate outputs from directions
            if self.bidirectional:
                outputs = torch.cat(layer_outputs, dim=2)  # (seq_len, batch, hidden_size * 2)
            else:
                outputs = layer_outputs[0]
            
            # Collect hidden states
            h_n.extend(layer_h)
            c_n.extend(layer_c)
            
            # Apply dropout if not the last layer
            if self.dropout_layer and layer < self.num_layers - 1:
                outputs = self.dropout_layer(outputs)
        
        h_n = torch.stack(h_n, dim=0)  # (num_layers * num_directions, batch, hidden_size)
        c_n = torch.stack(c_n, dim=0)  # (num_layers * num_directions, batch, hidden_size)
        
        if self.batch_first:
            outputs = outputs.transpose(0, 1)  # (batch, seq_len, hidden_size * num_directions)
        
        return outputs, (h_n, c_n)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, num_directions=1):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_directions = num_directions

        # Input weights
        self.weight_ih = Parameter(torch.Tensor(num_directions, 4 * hidden_size, input_size))
        # Hidden weights
        self.weight_hh = Parameter(torch.Tensor(num_directions, 4 * hidden_size, hidden_size))
        
        if self.bias:
            self.bias_ih = Parameter(torch.Tensor(num_directions, 4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(num_directions, 4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights and biases following PyTorch's default initialization
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, input, hx, direction):
        """
        input: (seq_len, batch, input_size)
        hx: Tuple of (h_0, c_0) for this layer and direction
            h_0: (batch, hidden_size)
            c_0: (batch, hidden_size)
        direction: 0 for forward, 1 for backward
        """
        h_0, c_0 = hx
        seq_len, batch, _ = input.size()
        
        outputs = []
        h = h_0
        c = c_0
        
        for t in range(seq_len):
            x = input[t]
            gates = (torch.matmul(x, self.weight_ih[direction].t()) +
                     torch.matmul(h, self.weight_hh[direction].t()))
            if self.bias:
                gates += self.bias_ih[direction] + self.bias_hh[direction]
            
            # Split gates into 4 parts
            i, f, g, o = gates.chunk(4, 1)
            
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            
            c = f * c + i * g
            h = o * torch.tanh(c)
            
            outputs.append(h.unsqueeze(0))
        
        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_size)
        return outputs, (h, c)