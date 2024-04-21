import torch
from torch_geometric.nn import GATConv
from models.layers.MLP import MLP
import torch.nn.functional as F


class GATNet(torch.nn.Module):
    def __init__(self, vertex_dim, edge_dim, out_dim, cfg: dict):
        """
        Encode the edges and vertices with MLPs and then pass them through GAT layers.

        """
        super(GATNet, self).__init__()
        self.cfg = cfg

        mlp_layers = cfg["mlp_layers"]
        mlp_hidden_dim = cfg["mlp_hid_dim"]
        N_gnn_layers = cfg["gnn_layers"]
        gnn_dim = cfg["gnn_dim"]
        heads = cfg["gnn_heads"]
        dropout = cfg["gnn_dropout"]

        # MLP embedder
        self.vertx_mlp = MLP(vertex_dim, gnn_dim, mlp_hidden_dim, num_layers=mlp_layers, act="relu", zero_last=False)
        self.edge_mlp = MLP(edge_dim, gnn_dim, mlp_hidden_dim, num_layers=mlp_layers, act="relu", zero_last=False)

        # GNN layers
        self.gnn_layers = torch.nn.ModuleList()
        # First layer
        self.gnn_layers.append(GATConv(gnn_dim, gnn_dim, heads=heads, dropout=dropout, edge_dim=gnn_dim))
        # Hidden layers (if any)
        for _ in range(N_gnn_layers-2):
            self.gnn_layers.append(GATConv(gnn_dim, gnn_dim, heads=heads, dropout=dropout, edge_dim=gnn_dim))
        # Output layer
        self.gnn_layers.append(GATConv(gnn_dim * heads, out_dim, heads=1, concat=True, dropout=dropout, edge_dim=gnn_dim))

    def forward(self, data):
        vert_in, edge_index, edge_in = data.x, data.edge_index, data.edge_attr

        V = self.vertx_mlp(vert_in)
        E = self.edge_mlp(edge_in)

        for i, layer in enumerate(self.gnn_layers):
            V = layer(V, edge_index, edge_attr=E)
            if i < len(self.gnn_layers) - 1:
                V = F.softplus(V)

        return V

