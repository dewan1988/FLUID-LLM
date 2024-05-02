import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

from models.layers.MLP import MLP
from models.layers.GNN.GCN import GCN_layers
from dataloader.ds_props import DSProps
from utils_model import patch_to_img

from matplotlib import pyplot as plt


class Decoder(nn.Module):
    def __init__(self, ds_props: DSProps):
        super().__init__()
        self.channel = ds_props.channel
        self.Nx_patch, self.Ny_patch = ds_props.Nx_patch, ds_props.Ny_patch
        self.N_patch = ds_props.N_patch
        self.out_patch_size = ds_props.out_patch_size
        self.Npx_mesh, self.Npy_mesh = ds_props.out_tot_size

        self.kern = (self.out_patch_size[0], self.out_patch_size[1])

    def patch_to_px(self, patch_vectors):
        """ return.shape = (bs, seq_len, Ny_mesh, Nx_mesh, hid_dim)"""
        bs, seq_len, N_patch, hid_dim = patch_vectors.shape

        patch_vectors_reshaped = patch_vectors.view(bs, seq_len, self.Nx_patch, self.Ny_patch, hid_dim)

        # Repeat each patch vector to cover its corresponding 4x4 subgrid
        patch_vectors_repeated = patch_vectors_reshaped.repeat_interleave(self.out_patch_size[0], dim=-3).repeat_interleave(self.out_patch_size[1], dim=-2)
        # Flatten the result to match the node feature matrix expected by PyTorch Geometric
        node_features = patch_vectors_repeated.view(bs, seq_len, self.Npx_mesh, self.Npy_mesh, hid_dim)
        return node_features


class MLPDecoder(Decoder):
    def __init__(self, in_dim, out_dim, ds_props: DSProps, params):
        super().__init__(ds_props)
        self.mlp_out_dim = (self.out_patch_size[0] * self.out_patch_size[1]) * self.channel

        self.hidden_dim = params['mlp_hid_dim']
        self.mlp_layers = params['mlp_layers']

        # Trainable MLP input layer
        self.input_mlp = MLP(in_dim, self.mlp_out_dim, hid_dim=self.hidden_dim, num_layers=self.mlp_layers, act='softplus', zero_last=False)

    def forward(self, patch_vectors):
        """ Return shape = [bs*seq_len, Nx_mesh, Ny_mesh, 3] """
        N_patch = 60
        bs, tot_patchs, llm_dim = patch_vectors.shape
        seq_len = tot_patchs // N_patch

        # 0) Preprocess input MLP
        patch_vectors = self.input_mlp(patch_vectors)  # shape = [bs, seq_len*N_patch, patch_size^2 * 3]
        patch_vectors = patch_vectors.view(bs * seq_len, N_patch, self.mlp_out_dim)
        patch_vectors = patch_vectors.permute(0, 2, 1)

        preds = F.fold(patch_vectors,
                       output_size=(self.Npx_mesh, self.Npy_mesh),
                       kernel_size=self.kern, stride=self.kern)
        preds = preds.permute(0, 2, 3, 1)

        # plt.imshow(patch_vectors[0, 1].cpu().detach().float().numpy())
        # plt.show()
        # print(patch_vectors.shape)
        return preds


class GNNDecoder(Decoder):
    def __init__(self, in_dim, out_dim, ds_props: DSProps, params):
        super().__init__()
        self.channel = ds_props.channel
        self.Nx_patch, self.Ny_patch = ds_props.Nx_patch, ds_props.Ny_patch
        self.patch_px = ds_props.patch_size

        self.hidden_dim = 128

        # Trainable MLP input layer
        self.input_mlp = MLP(in_dim, self.hidden_dim, hid_dim=512, num_layers=2, act='softplus', zero_last=False).cuda()
        # GNN
        self.GNN = GCN_layers(self.hidden_dim + 4, self.hidden_dim, 3, num_layers=1).cuda()

        # Indices for patch number
        x_pa_grid = torch.arange(self.Nx_patch)
        y_pa_grid = torch.arange(self.Ny_patch)
        x_pa_grid, y_pa_grid = torch.meshgrid(x_pa_grid, y_pa_grid, indexing='ij')
        patch_idx = torch.stack((x_pa_grid, y_pa_grid), dim=-1).view(1, 1, 60, 2)
        # patch_idx = patch_idx / patch_idx.max()
        self.patch_idx = self.patch_to_px(patch_idx).cuda()

        # Indices for pixel number
        # x_px_grid = torch.arange(self.Nx_mesh)
        # y_px_grid = torch.arange(self.Ny_mesh)
        # x_px_grid, y_px_grid = torch.meshgrid(x_px_grid, y_px_grid, indexing='ij')
        # pixel_idx = torch.stack((x_px_grid, y_px_grid), dim=-1)
        # pixel_idx = pixel_idx / pixel_idx.max()
        # self.pixel_idx = pixel_idx.unsqueeze(0).unsqueeze(0).cuda()

        x_px_grid = torch.arange(self.patch_size)
        y_px_grid = torch.arange(self.patch_size)
        x_px_grid, y_px_grid = torch.meshgrid(x_px_grid, y_px_grid, indexing='ij')
        pixel_idx = torch.stack((x_px_grid, y_px_grid), dim=-1).repeat(self.Nx_patch, self.Ny_patch, 1)
        pixel_idx = pixel_idx / pixel_idx.max()
        self.pixel_idx = pixel_idx.unsqueeze(0).unsqueeze(0).cuda()

        # plt.imshow(self.pixel_idx[0, 0, :, :, 0].cpu())
        # plt.show()
        # print(f'{self.patch_idx.shape = }, {self.pixel_idx.shape = }')
        # exit(7)

        # Indices for edges
        self.mesh_edges = make_edge_idx(self.Ny_mesh, self.Nx_mesh).cuda()

        # bs, seq_len = 8, 3
        # patch_vectors = torch.randn(8, 540, 768).cuda()
        # node_features = self.forward(patch_vectors)

    def forward(self, patch_vectors):
        """ Return shape = [bs*seq_len, Nx_mesh, Ny_mesh, 3] """
        N_patch = 60
        bs, tot_patchs, llm_dim = patch_vectors.shape
        seq_len = tot_patchs // N_patch

        # 0) Preprocess input MLP
        patch_vectors = self.input_mlp(patch_vectors)  # shape = [bs, seq_len*N_patch, hid_dim]

        # 1) Map patches to higher resolution grid
        # patch_vectors = torch.zeros_like(patch_vectors)
        # patch_vectors[0, 4] = torch.ones_like(patch_vectors[0, 0])
        patch_vectors = patch_vectors.view(bs, seq_len, N_patch, self.hidden_dim)  # shape = [bs, seq_len, N_patch, hid_dim]
        node_features = self.patch_to_px(patch_vectors)  # shape = [bs, seq_len, Nx_patch, Ny_patch, hid_dim]

        # node_features = node_features.reshape(bs, seq_len, self.Nx_mesh, self.Ny_mesh, self.hidden_dim)
        # print(node_features.shape)
        # # node_features = node_features.reshape(bs, seq_len, self.Nx_mesh * self.Ny_mesh, self.hidden_dim)
        # # node_features = node_features.reshape(bs, seq_len, self.Nx_mesh, self.Ny_mesh, self.hidden_dim)
        # gnn_input = node_features[0, 0, :, :, 0]
        # # gnn_input = gnn_input.view(self.Ny_mesh, self.Nx_mesh, -1)[:, :, 0]
        # plt.imshow(gnn_input.cpu().detach().numpy())
        # plt.show()
        # print(gnn_input.shape)
        # exit(9)

        # 3) Concatenate on positional features
        patch_idx = self.patch_idx.repeat(bs, seq_len, 1, 1, 1)
        pixel_idx = self.pixel_idx.repeat(bs, seq_len, 1, 1, 1)
        # patch_idx, pixel_idx = torch.zeros_like(patch_idx), torch.zeros_like(pixel_idx)
        # print(f'{node_features.std() = }, {pixel_idx.std() = }, {patch_idx.std() = }')
        node_features = torch.cat((node_features, patch_idx, pixel_idx), dim=-1)  # shape = [bs, seq_len, Nx_mesh, Ny_mesh, hid_dim+4]

        # print(f'{node_features.shape = }')
        # Convert to graph for GNN
        node_features = node_features.view(bs * seq_len, self.Nx_mesh * self.Ny_mesh, self.hidden_dim + 4)
        #
        # gnn_input = node_features[0] # , 0, : ,:, 0]
        # gnn_input = gnn_input.view(self.Nx_mesh, self.Ny_mesh, -1)[:, :, -2]
        # plt.imshow(gnn_input.cpu().detach().float().numpy())
        # plt.show()
        # print(gnn_input.shape)
        # exit(9)

        # 4) Pass through GNN
        graphs = []
        for single_graph in node_features:
            graph = Data(x=single_graph, edge_index=self.mesh_edges)
            graphs.append(graph)
        graphs = Batch.from_data_list(graphs)

        preds = self.GNN(graphs.x, graphs.edge_index)  # shape = [bs*seq_len*Nx_mesh*Ny_mesh, 3]

        # print(preds.shape)
        # analyse_num = 50
        # neigbours = analyse_edge_idx(graphs.edge_index, analyse_num)
        # preds[analyse_num] = 2.
        # for n in neigbours:
        #     preds[n] = 1.
        #

        # 5) Reshape to image format
        preds = preds.view(-1, self.Nx_mesh, self.Ny_mesh, 3)  # shape = [bs*seq_len, Nx_mesh, Ny_mesh, 3]

        # preds = preds.detach().cpu()
        # plt.imshow(preds[0, :, :, 0].T)
        # plt.show()
        # exit(9)
        # print(f'{preds.shape = }')"""
        # patch_vectors = patch_vectors.view(bs*seq_len, 120, 32, 3)
        #

        return preds


class MLPGNNDecoder(Decoder):
    def __init__(self, in_dim, out_dim, ds_props: DSProps, params):
        super().__init__(ds_props)
        self.dropout = params['dropout']
        self.gnn_dim = params['gnn_dim']
        self.gnn_layers = params['gnn_layers']
        self.gnn_heads = params['gnn_heads']
        self.gnn_hid_dim = params['gnn_hid_dim']
        self.mlp_hid_dim = params['mlp_hid_dim']
        self.mlp_out_dim = (self.out_patch_size[0] * self.out_patch_size[1]) * self.gnn_dim

        # Trainable MLP input layer
        self.input_mlp = MLP(in_dim, self.mlp_out_dim, hid_dim=self.mlp_hid_dim, num_layers=2, act='softplus', zero_last=False)
        # GNN
        self.GNN = GCN_layers(in_dim=self.gnn_dim, hid_dim=self.gnn_hid_dim, out_dim=3, num_layers=self.gnn_layers, heads=self.gnn_heads, dropout=self.dropout)
        self.GNN = torch.compile(self.GNN)

        # Indices for edges
        mesh_edges = make_edge_idx(self.Npy_mesh, self.Npx_mesh)
        self.mesh_edges = torch.nn.Parameter(mesh_edges, requires_grad=False)

    def forward(self, patch_vectors):
        """ Return shape = [bs, seq_len, Npx_mesh, Ny_mesh, 3] """
        bs, tot_patchs, _ = patch_vectors.shape
        seq_len = tot_patchs // self.N_patch

        # 0) Preprocess input MLP
        patch_vectors = self.input_mlp(patch_vectors)  # shape = [bs, seq_len*N_patch, hid_dim]

        # 1) Flatten out patches to grid
        patch_vectors = patch_vectors.view(bs * seq_len, self.N_patch, self.mlp_out_dim)  # shape = [bs*seq_len, N_patch, gnn_out_dim]
        patch_vectors = patch_vectors.permute(0, 2, 1)  # shape = [bs*seq_len, gnn_out_dim, N_patch]

        node_ft = F.fold(patch_vectors,
                         output_size=(self.Npx_mesh, self.Npy_mesh),
                         kernel_size=self.kern, stride=self.kern)

        # 4) Flatten nodes and pass through GNN
        node_ft = node_ft.view(bs * seq_len, self.gnn_dim, self.Npx_mesh, self.Npy_mesh)
        node_ft = node_ft.permute(0, 2, 3, 1)  # shape = [bs*seq_len, Npx_mesh, Ny_mesh, gnn_dim]
        node_ft = node_ft.view(bs * seq_len, self.Npx_mesh * self.Npy_mesh, self.gnn_dim)
        graphs = []
        for single_graph in node_ft:
            graph = Data(x=single_graph, edge_index=self.mesh_edges)
            graphs.append(graph)
        graphs = Batch.from_data_list(graphs)

        preds = self.GNN(graphs.x, graphs.edge_index)  # shape = [bs*seq_len*Npx_mesh*Ny_mesh, 3]

        # 5) Reshape to image format
        preds = preds.view(bs, seq_len, self.Npx_mesh, self.Npy_mesh, 3)  # shape = [bs, seq_len, Nx_mesh, Ny_mesh, 3]

        return preds


def analyse_edge_idx(edge_indices, node):
    num_nodes = edge_indices.shape[1]
    # Print edges for each node
    # Find indices where 'node' is a source
    source_mask = edge_indices[0] == node
    # Find indices where 'node' is a destination
    dest_mask = edge_indices[1] == node

    # Get edges where 'node' is a source
    connected_as_source = edge_indices[:, source_mask]
    # Get edges where 'node' is a destination
    connected_as_dest = edge_indices[:, dest_mask]
    #
    # Print the information
    print(f"\nNode {node} is connected to:")
    if connected_as_source.nelement() > 0:
        print(f"  As source to: {connected_as_source[1].tolist()}")
    if connected_as_dest.nelement() > 0:
        print(f"  As destination from: {connected_as_dest[0].tolist()}")

    # Return all connected nodes
    return connected_as_source[1].tolist() + connected_as_dest[0].tolist()


def make_edge_idx(n, m):
    """ n=rows, m=cols"""
    # Number of nodes
    num_nodes = n * m

    # Edge lists
    edge_indices = []

    # Vertical edges (within columns)
    for j in range(m):
        for i in range(n - 1):
            idx = j * n + i
            edge_indices.append([idx, idx + 1])
            edge_indices.append([idx + 1, idx])  # Add reverse direction if undirected graph

    # Horizontal edges (across columns)
    for i in range(n):
        for j in range(m - 1):
            idx = j * n + i
            edge_indices.append([idx, idx + n])
            edge_indices.append([idx + n, idx])  # Add reverse direction if undirected graph

    # Convert to tensor
    edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return edge_index_tensor


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
