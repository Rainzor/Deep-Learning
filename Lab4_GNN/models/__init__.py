import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from .GCN import GCNConv
from .GAT import GATConv

GNN = {
    'gcn': GCNConv,
    'gat': GATConv
}

class PairNorm(nn.Module):
    def __init__(self, mode='PN'):
        """
        PairNorm 

        Args:
            mode (str): normlization mode
        """
        super(PairNorm, self).__init__()
        mode = mode.upper()
        assert mode in ['PN', 'PN-SI', 'PN-SCS'], "Invalid PairNorm mode."
        self.mode = mode

    def forward(self, x):
        if self.mode == 'PN':
            # L2 Norm
            norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8
            x = x / norm
        elif self.mode == 'PN-SI':
            # Scale-Invariant PairNorm
            mean = x.mean(dim=0, keepdim=True)
            x = x - mean
            norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8
            x = x / norm
        elif self.mode == 'PN-SCS':
            # Scale and Center Scale PairNorm
            mean = x.mean(dim=0, keepdim=True)
            x = x - mean
            std = x.std(dim=0, keepdim=True) + 1e-8
            x = x / std
        return x

class GNNEnocder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=None, gnn_type='gcn', 
                    num_layers=2, 
                    dropout=0.5, 
                    edge_dropout = 0.0,
                    self_loop = True,
                    pairnorm_mode = None,
                    residual = False,
                    activation = 'gelu',
                    **kwargs):
        """
        Initialize a simple GCN model.

        Args:
            in_channels (int): Number of input features.
            hidden_channels (int): Number of hidden features.
            out_channels (int): Number of output features.
        """
        super(GNNEnocder, self).__init__()
        if out_channels is None:
            out_channels = hidden_channels
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.pairnorm_mode = pairnorm_mode
        self.residual = residual
        gnn_type = gnn_type.lower()
        assert num_layers >= 2, "Number of layers must be at least 2."
        assert gnn_type in ['gcn', 'gat'], "GNN type must be 'gcn' or 'gat'."
        assert activation in ['gelu', 'relu'], "Activation must be 'gelu' or 'relu'."
        self.conv_layers = nn.ModuleList()

        self.conv_layers.append(GNN[gnn_type](in_channels, hidden_channels, self_loops=self_loop))
        if in_channels != hidden_channels and residual:
            self.skip_connections = nn.Linear(in_channels, hidden_channels) 
        else: 
            self.skip_connections = None

        for _ in range(num_layers - 2):
            self.conv_layers.append(GNN[gnn_type](hidden_channels, hidden_channels, self_loops=self_loop))
        
        self.conv_layers.append(GNN[gnn_type](hidden_channels, out_channels, self_loops=self_loop))

        self.pairnorms = None
        if pairnorm_mode:
            self.pairnorms = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.pairnorms.append(PairNorm(mode=self.pairnorm_mode))
        self.dropout = nn.Dropout(p=dropout)
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize model parameters.
        """
        if self.skip_connections is not None:
            nn.init.xavier_uniform_(self.skip_connections.weight)
            nn.init.zeros_(self.skip_connections.bias)

        for layer in self.conv_layers:
            layer.reset_parameters()
        


    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.

        Args:
            x (Tensor): Node features of shape [N, in_channels].
            edge_index (LongTensor): Edge indices in COO format of shape [2, E].

        Returns:
            Tensor: Output features of shape [N, out_channels].
        """
        if self.training and self.edge_dropout > 0:
            edge_index, _ = torch_geometric.utils.dropout_adj(edge_index, p=self.edge_dropout, training=self.training)

        prev_x = self.skip_connections(x) if self.skip_connections is not None else x

        for i, conv in enumerate(self.conv_layers[:-1]):
            x = conv(x, edge_index)
            if self.pairnorms:
                x = self.pairnorms[i](x)
            if self.residual:
                x = x + prev_x
                prev_x = x
            x = self.activation(x)
            x = self.dropout(x)

        x = self.conv_layers[-1](x, edge_index)
        return x