"""Plots a CFD trajectory rollout."""

import numpy as np
import os
import pickle
import random
from cprint import c_print
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

from dataloader.gnn.gnn_utils import faces_to_edges, plot_mesh, unflatten_states


def num_patches(dim_size, kern_size, stride, padding=0):
    """
    Returns the number of patches that can be extracted from an image
    """
    return (dim_size + 2 * padding - kern_size) // stride + 1


class MGNDataset(Dataset):
    """ Load a single timestep from the dataset."""

    def __init__(self, load_dir, seq_len: int, seq_interval=1,
                 mode="train", normalize=True):
        super(MGNDataset, self).__init__()

        assert mode in ["train", "valid", "test"]

        self.mode = mode
        self.load_dir = load_dir
        self.seq_len = seq_len
        self.seq_interval = seq_interval
        self.max_step_num = 600 - self.seq_len * self.seq_interval
        self.normalize = normalize

        self.save_files = sorted([f for f in os.listdir(f"{self.load_dir}/") if f.endswith('.pkl')])
        # Load a random file to get min and max values and patch size
        # triang, tri_index, grid_x, grid_y, save_data = self._load_step(self.save_files[1])
        V_states, _, node_types, _, _ = self._load_steps(save_file=self.save_files[0], step_num=20)

        # Get min and max values for each channel
        mins = torch.amin(V_states, dim=(0, 1))
        maxs = torch.amax(V_states, dim=(0, 1))
        self.ds_min_max = torch.stack([mins, maxs]).T
        self.vertex_dim = 3 + len(torch.unique(node_types))
        self.edge_dim = 3

    def __getitem__(self, idx):
        # Time sampling is random during training, but set to a fix value during test and valid, to ensure repeatability.
        step_num = random.randint(0, self.max_step_num)
        step_num = 100 if self.mode in ["test", "valid"] else step_num

        return self.ds_get(save_file=self.save_files[idx], step_num=step_num)

    def ds_get(self, save_file=None, step_num=None):
        """ Returns graph and positional embeddings
            Return:
                edge_embeddings.shape = [seq_len-1, N_edges, 3]
                vertex_embeddings.shape = [seq_len-1, N_nodes, 3 + N_node_types]
                vertex_next.shape = [seq_len-1, N_nodes, 3]
                mask.shape = [seq_len, N_nodes]

                And some other mesh stuff used for plotting
        """
        save_file, step_num = self._select_step(save_file, step_num)

        V_states, mesh_pos, node_type, edges, faces = self._load_steps(save_file, step_num)

        # Get coordinates of mesh
        senders = torch.gather(mesh_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 2))
        receivers = torch.gather(mesh_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 2))

        # Make embeddings from distance and coordinates
        distance = senders - receivers
        norm = torch.sqrt((distance ** 2).sum(-1, keepdims=True))
        edge_embeddings = torch.cat([distance, norm], dim=-1)

        # Onehot node embeddings
        unique_idx = torch.unique(node_type)
        mapped_idx = torch.searchsorted(unique_idx, node_type)
        onehot_idx = F.one_hot(mapped_idx, num_classes=len(unique_idx))

        # Repeat edge values for entire sequence
        E_embeddings = edge_embeddings.repeat(self.seq_len - 1, 1, 1)
        onehot_idx = onehot_idx.repeat(self.seq_len - 1, 1, 1)
        edges = edges.T.repeat(self.seq_len - 1, 1, 1)

        # Make vertex_embeddings current state, vertex_next next state
        V_state, V_next = V_states[:-1], V_states[1:]
        V_embeddings = torch.cat([V_state, onehot_idx], dim=-1)

        # Mask of nodes that are not free
        mask = node_type != 0
        mask = mask.repeat(self.seq_len - 1, 1)

        return E_embeddings, V_embeddings, V_next, edges, mask, mesh_pos, faces

    def _load_steps(self, save_file, step_num):
        """ Load save file from disk and calculate mesh interpolation triangles"""

        with open(f"{self.load_dir}/{save_file}", 'rb') as f:
            save_data = pickle.load(f)  # ['velocity', 'pressure', 'cells', 'mesh_pos', 'node_type']

        V = torch.from_numpy(save_data['velocity'])
        P = torch.from_numpy(save_data['pressure'])[:, :, 0]
        Vx, Vy = V[:, :, 0], V[:, :, 1]
        V_states = torch.stack([Vx, Vy, P], dim=-1)

        # These are the same for each timestep
        mesh_pos = torch.from_numpy(save_data['mesh_pos'])
        node_type = torch.from_numpy(save_data['node_type']).long().squeeze()
        faces = torch.from_numpy(save_data['cells']).long()

        V_states, mesh_pos, node_type, faces = self._filter_nodes(V_states, mesh_pos, node_type, faces)
        edges = faces_to_edges(faces).to(torch.int64)

        # Select wanted states
        get_idx = torch.arange(step_num, step_num + self.seq_len * self.seq_interval, self.seq_interval)
        V_states = V_states[get_idx]

        return V_states, mesh_pos, node_type, edges, faces

    def _filter_nodes(self, V_states, mesh_pos, node_type, faces):
        """ Extract values from save_data as torch tensors and filter out nodes that are not in the mask."""
        # ['velocity', 'pressure', 'cells', 'mesh_pos', 'node_type']

        # Mask off unwanted region
        x_mask = (mesh_pos[:, 0] > 0.) & (mesh_pos[:, 0] < 2)
        y_mask = (mesh_pos[:, 1] > 0.) & (mesh_pos[:, 1] < 0.5)
        mask = x_mask & y_mask

        V_states = V_states[:, mask]
        mesh_pos = mesh_pos[mask]
        node_type = node_type[mask]

        # Filter out faces that are not in the mask
        wanted_nodes = torch.nonzero(mask).squeeze()
        # Make a mapping from all nodes to wanted nodes, unwanted nodes are set to 0
        all_nodes = torch.zeros(len(mask)).long()
        all_nodes[mask] = torch.arange(len(wanted_nodes)).long()

        face_mask = torch.isin(faces, wanted_nodes).all(dim=1)
        faces = faces[face_mask]
        faces = all_nodes[faces]

        return V_states, mesh_pos, node_type, faces

    def _normalize(self, states):

        # Coordinate
        # State 0:  0.823, 0.3315
        # Diff 0: 1.614e-05, 0.000512
        # 0.195, 0.000515
        # Coordinate
        # State 1:  0.0005865, 0.01351
        # Diff 1: 3.7e-06, 0.0005696
        # 0.0135, 0.000572
        # Coordinate
        # State 2:  0.04763, 0.07536
        # Diff 2: -0.002683, 0.00208
        # 0.0739, 0.00208
        raise NotImplementedError("Normalisation is tricky when x and y velocities have different scale, even though the system is x_y invariant")

        s0_mean, s0_var = 0.823, 0.3315
        s1_mean, s1_var = 0.0005865, 0.01351
        s2_mean, s2_var = 0.04763, 0.07536

        means = torch.tensor([s0_mean, s1_mean, s2_mean]).reshape(1, 3, 1, 1)
        stds = torch.sqrt(torch.tensor([s0_var, s1_var, s2_var]).reshape(1, 3, 1, 1))

        # Normalise states
        states = states - means
        states = states / stds

        return states

    def _select_step(self, save_file, step_num):
        """ Choose which file and step to load from"""
        if save_file is None:
            save_file = random.choice(self.save_files)
        elif isinstance(save_file, int):
            save_file = self.save_files[save_file]

        if step_num is None:
            step_num = np.random.randint(0, self.max_step_num)
        if step_num > self.max_step_num:
            c_print(f"Step number {step_num} too high, setting to max step number {self.max_step_num}", 'red')
            step_num = self.max_step_num
        return save_file, step_num

    def __len__(self):
        return len(self.save_files)


def collate_graph_sequences(batch):
    """`batch` is a list of samples from the dataloader. V_next/target doesn't need to be graphed.
        Concatenate all values into a single tensor, while keeping track of vertex structure, to allow recovery
        """
    bs = len(batch)
    split_idx = []
    state_graphs, targets, masks = [], [], []
    mesh_properties = []
    for e_embs, v_embs, v_nexts, e_idxs, m, mesh_pos, faces in batch:
        seq_len, n_nodes, _ = v_embs.shape
        for e_emb, v_emb, e_idx in zip(e_embs, v_embs, e_idxs):
            state_graphs.append(Data(edge_index=e_idx, edge_attr=e_emb, x=v_emb))

        # Keep track of mesh properties
        split_idx.append(n_nodes)
        targets.append(v_nexts.view(-1, 3))
        mesh_properties.append((mesh_pos, faces))
        masks.append(m.flatten())

    state_graphs = Batch.from_data_list(state_graphs)
    split_idx = torch.tensor(split_idx)
    targets = torch.cat(targets, dim=0)  # Concatenate along vertex dim
    masks = torch.cat(masks)
    return state_graphs, targets, masks, (bs, seq_len, split_idx, mesh_properties)


def plot_all_patches():
    seq_dl = MGNDataset(load_dir="./ds/MGN/cylinder_dataset/train",
                        seq_len=10, seq_interval=2, normalize=False)
    ds = DataLoader(seq_dl, batch_size=8, num_workers=0, collate_fn=collate_graph_sequences, shuffle=True)

    plot_batch_num = 0
    plot_step = 1
    plot_coord = 0

    for batch in ds:
        data_graphs, targets, masks, (bs, seq_len, split_idx, mesh_properties) = batch
        break

    mesh_pos, faces = mesh_properties[plot_batch_num]

    V_emb = data_graphs.x
    V_emb = unflatten_states(V_emb, split_idx, seq_len)
    targets = unflatten_states(targets, split_idx, seq_len)

    Vs = V_emb[plot_batch_num]
    Vs = Vs[plot_step, :, plot_coord]

    targets = targets[plot_batch_num]
    targets = targets[plot_step, :, plot_coord]

    plot_mesh(mesh_pos, faces, Vs)
    plot_mesh(mesh_pos, faces, targets)
    plot_mesh(mesh_pos, faces, Vs - targets)


if __name__ == '__main__':
    from utils import set_seed

    set_seed(0)

    # plot_patches(None, 10, 20)
    plot_all_patches()
