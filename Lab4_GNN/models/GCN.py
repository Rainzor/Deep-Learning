import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import torch_geometric.utils as utils

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, self_loop=True):
        """
        Initialize the GCN convolution layer.

        Args:
            in_channels (int): Number of input features per node.
            out_channels (int): Number of output features per node.
        """
        # Initialize the MessagePassing class with 'add' aggregation
        super(GCNConv, self).__init__(aggr='add')

        self.self_loop = self_loop

        # Define a linear transformation (weight matrix)
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN layer.

        x_i  = W * \sum_{j \in N(i)} (1/sqrt(d_i * d_j)) * x_j

        Args:
            x (Tensor): Node feature matrix of shape [N, in_channels].
            edge_index (LongTensor): Edge indices in COO format of shape [2, E].

        Returns:
            Tensor: Output node features of shape [N, out_channels].
        """
        # Add self-loops to the adjacency matrix
        if self.self_loop:
            edge_index, _ = utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization coefficients
        row, col = edge_index
        deg = utils.degree(row, x.size(0), dtype=x.dtype)  # Degree of each node
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle division by zero

        # Apply linear transformation
        x = self.linear(x)  # [N, out_channels]

        # Initiate message passing
        return self.propagate(edge_index, x=x, norm=(deg_inv_sqrt, row, col))

    def message(self, x_j, norm):
        """
        Define the message computation.

        Args:
            x_j (Tensor): Neighbor node features of shape [E, out_channels].
            norm (tuple): Tuple containing normalization factors.

        Returns:
            Tensor: Messages to be aggregated.
        """
        deg_inv_sqrt, row, col = norm

        # Source node degree (j)
        D_j = deg_inv_sqrt[col]
        # Target node degree (i)
        D_i = deg_inv_sqrt[row]

        # Compute normalization factor for each edge
        alpha = D_i * D_j

        # Scale the messages
        return alpha.view(-1, 1) * x_j  # [E, out_channels]

    def update(self, aggr_out):
        """
        Update node embeddings after aggregation.

        Args:
            aggr_out (Tensor): Aggregated messages of shape [N, out_channels].

        Returns:
            Tensor: Updated node features.
        """
        return aggr_out