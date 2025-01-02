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


class GNNEnocder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=None, gnn_type='gcn', num_layers=2, dropout=0.5, residual=False,  **kwargs):
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
        self.residual = residual
        gnn_type = gnn_type.lower()
        assert num_layers >= 2, "Number of layers must be at least 2."
        assert gnn_type in ['gcn', 'gat'], "GNN type must be 'gcn' or 'gat'."

        self.conv_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        self.conv_layers.append(GNN[gnn_type](in_channels, hidden_channels))
        self.skip_connections = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

        for _ in range(num_layers - 2):
            self.conv_layers.append(GNN[gnn_type](hidden_channels, hidden_channels))
        
        self.conv_layers.append(GNN[gnn_type](hidden_channels, out_channels))

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
        prev_x = self.skip_connections(x) if self.skip_connections is not None else x
        for i, conv in enumerate(self.conv_layers):
            if i > 0:
                if self.residual:
                    x = x + prev_x
                    prev_x = x
                x = F.dropout(F.gelu(x), p=self.dropout, training=self.training)
            x = conv(x, edge_index)
        return x