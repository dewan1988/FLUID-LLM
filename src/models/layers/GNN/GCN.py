import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv


class GCN_layers(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout, heads):
        """
        Initializes the Graph Convolutional Network with variable input dimensions,
        output dimensions, and hidden layers.

        Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dims (list of int): List containing the sizes of each hidden layer.
        output_dim (int): Dimensionality of the output features.
        """
        super().__init__()

        # layer_fn = GATv2Conv
        # Create a list of all GCN convolutional layers
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.out_conv = GATv2Conv(in_dim, out_dim, dropout=dropout, add_self_loops=True, bias=False)
        else:
            self.convs.append(GATv2Conv(in_dim, hid_dim//heads, heads=heads, dropout=dropout, add_self_loops=True))
            for l in range(num_layers - 2):
                self.convs.append(GATv2Conv(hid_dim, hid_dim//heads, heads=heads, dropout=dropout, add_self_loops=True))

            # Output layer
            self.out_conv = GATv2Conv(hid_dim, out_dim, dropout=dropout, add_self_loops=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN.

        Args:
        x (Tensor): Node feature matrix (shape [num_nodes, input_dim])
        edge_index (Tensor): Graph connectivity in COO format (shape [2, num_edges])
        batch (Tensor): Batch vector, which assigns each node to a specific example

        Returns:
        Tensor: Output from the GNN after applying log softmax.
        """
        # Pass through each convolutional layer with ReLU activation
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.softplus(x)
            # x = F.dropout(x, p=0.5, training=self.training)

        # Pass through the output convolutional layer
        x = self.out_conv(x, edge_index)
        return x
