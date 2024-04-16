import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data, Batch
from models.layers.MLP import MLP
from models.layers.GNN.GCN import GCN_layers


class GNNDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, patch_shape, params):
        super().__init__()
        self.channel, self.N_px, self.N_py = patch_shape
        self.Nx_patch, self.Ny_patch = 15, 4
        self.Nx_mesh, self.Ny_mesh = 60, 16
        self.patch_size = 4
        self.hidden_dim = 128

        # Trainable MLP input layer
        self.input_mlp = MLP(3, 128, hid_dim=self.hidden_dim, num_layers=2, act='relu', zero_last=False).cuda()
        # GNN
        self.GNN = GCN_layers(self.hidden_dim+4, self.hidden_dim, 3, num_layers=2).cuda()

        # Indices for patch number
        tot_patches = self.Nx_patch * self.Ny_patch
        idx = torch.arange(tot_patches)
        x_idx = idx // self.Ny_patch
        y_idx = idx % self.Ny_patch
        patch_idx = torch.stack((x_idx, y_idx), dim=1)
        patch_idx = patch_idx.view(1, 1, 60, 2)
        self.patch_idx = self.patch_to_px(patch_idx).cuda()

        # Indices for pixel number
        x_px_grid = torch.arange(self.Nx_mesh)
        y_px_grid = torch.arange(self.Ny_mesh)
        x_px_grid, y_px_grid = torch.meshgrid(y_px_grid, x_px_grid, indexing='ij')
        self.pixel_idx = torch.stack((y_px_grid, x_px_grid), dim=-1).unsqueeze(0).unsqueeze(0).cuda()

        # Indices for edges
        self.mesh_edges = make_edge_idx(60, 16).cuda()

        bs, seq_len = 8, 3
        patch_vectors = torch.arange(bs * seq_len * self.Nx_patch * self.Ny_patch).view(bs, -1, 1).cuda()
        patch_vectors = torch.cat([patch_vectors, patch_vectors, patch_vectors], dim=-1)
        node_features = self.forward(patch_vectors.float()).cpu().detach()

        # from matplotlib import pyplot as plt
        # plt.imshow(node_features[0, 0, :, :, 0], vmin=0)
        # plt.show()
        #
        # print(f'{node_features.shape = }')
        # print(f'{self.mesh_edges.shape = }')
        exit(9)

    def forward(self, patch_vectors):
        N_patch = 60
        bs, tot_patchs, llm_dim = patch_vectors.shape
        seq_len = tot_patchs // N_patch

        # 0) Preprocess input MLP
        patch_vectors = self.input_mlp(patch_vectors)

        # 1) Split GNN
        patch_vectors = patch_vectors.view(bs, seq_len, N_patch, self.hidden_dim)
        node_features = self.patch_to_px(patch_vectors)

        patch_idx = self.patch_idx.repeat(bs, seq_len, 1, 1, 1)
        pixel_idx = self.pixel_idx.repeat(bs, seq_len, 1, 1, 1)

        node_features = torch.cat((node_features, patch_idx, pixel_idx), dim=-1)  # shape = [bs, seq_len, Ny_mesh, Nx_mesh, hid_dim+4]

        # Convert to graph for GNN
        node_features = node_features.view(bs * seq_len, self.Nx_mesh * self.Ny_mesh, -1)
        graphs = []
        for single_graph in node_features:
            graph = Data(x=single_graph, edge_index=self.mesh_edges)
            graphs.append(graph)
        graphs = Batch.from_data_list(graphs)

        preds = self.GNN(graphs.x, graphs.edge_index)       # shape = [bs*seq_len*Nx_mesh*Ny_mesh, 3]
        preds = preds.view(bs, seq_len, self.Nx_mesh, self.Ny_mesh, 3)
        print(preds.shape)

        return node_features

    def patch_to_px(self, patch_vectors):
        """ return.shape = (bs, seq_len, Nx_mesh, Ny_mesh, hid_dim)"""
        bs, seq_len, N_patch, hid_dim = patch_vectors.shape

        patch_vectors_reshaped = patch_vectors.view(bs, seq_len, self.Nx_patch, self.Ny_patch, hid_dim)
        # Repeat each patch vector to cover its corresponding 4x4 subgrid
        patch_vectors_repeated = patch_vectors_reshaped.repeat_interleave(self.patch_size, dim=-3).repeat_interleave(self.patch_size, dim=-2)
        # Flatten the result to match the node feature matrix expected by PyTorch Geometric
        node_features = patch_vectors_repeated.view(bs, seq_len, self.Nx_mesh, self.Ny_mesh, hid_dim).transpose(2, 3)
        return node_features


def make_edge_idx(rows, cols):
    num_nodes = rows * cols
    nodes = torch.arange(num_nodes)

    # Horizontal edges
    mask = nodes % cols != (cols - 1)  # Mask to avoid wrapping around rows
    horizontal_edges = torch.stack([nodes[mask], nodes[mask] + 1], dim=0)

    # Vertical edges
    mask = nodes < (num_nodes - cols)  # Mask to avoid wrapping around columns
    vertical_edges = torch.stack([nodes[mask], nodes[mask] + cols], dim=0)

    # Combine horizontal and vertical edges
    edge_index = torch.cat([horizontal_edges, vertical_edges], dim=1)

    return edge_index


def test_plot(node_values, edge_index):
    from matplotlib import pyplot as plt
    import networkx as nx

    rows, cols = 16, 60
    num_nodes = rows * cols

    # Create an empty graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(num_nodes))

    # Add edges based on edge_index
    edges = edge_index.t().tolist()  # Convert edge_index to a list of edges
    G.add_edges_from(edges)

    # Plot the graph
    pos = {i: ((i // rows), -(i % rows)) for i in range(num_nodes)}  # Position nodes based on their grid coordinates
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=50, font_size=8, edge_color='k')
    plt.title("Grid Graph Connectivity")
    plt.show()
