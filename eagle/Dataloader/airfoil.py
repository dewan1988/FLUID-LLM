import os.path
import random
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import torch
import numpy as np
import pickle
import natsort

NODE_NORMAL = 0
NODE_INPUT = 4
NODE_OUTPUT = 5
NODE_WALL = 6
NODE_DISABLE = 2


class AirfoilDataset(Dataset):
    def __init__(self, data_path, mode="test", window_length=990, apply_onehot=True, with_cells=False,
                 with_cluster=False, n_cluster=20, normalize=False):
        """ Eagle dataset
        :param data_path: path to the dataset
        :param window_length: length of the temporal window to sample the simulation
        :param apply_onehot: Encode node type as onehot vector, see global variables to see what is what
        :param with_cells: Default is to return edges as pairs of indices, if True, return also the triangles (cells),
        useful for visualization
        :param with_cluster: Load the clustered indices of the nodes
        :param n_cluster: Number of cluster to use, 0 means no clustering / one node per cluster
        :param normalize: center mean and std of velocity/pressure field
        """
        super().__init__()
        assert mode in ["train", "test", "valid"]

        self.window_length = window_length
        assert window_length <= 990, "window length must be smaller than 990"

        self.fn = os.path.join(data_path, mode)
        assert os.path.exists(self.fn), f"Path {self.fn} does not exist"

        self.apply_onehot = apply_onehot
        self.dataloc = []

        for root, directories, files in os.walk(self.fn):
            for filename in files:
                filepath = os.path.join(root, filename)

                if filepath.endswith(".pkl"):
                    self.dataloc.append(filepath)

        self.dataloc = natsort.natsorted(self.dataloc)

        self.with_cells = with_cells
        self.with_cluster = with_cluster
        self.n_cluster = n_cluster
        self.mode = mode
        self.do_normalization = normalize
        self.length = 990

        if self.with_cluster:
            assert self.n_cluster in [0, 10, 20, 40, 30], f'Please, check if clustering has been computed offline ' \
                                                          f'for {self.n_cluster} nodes per cluster'

    def __len__(self):
        return len(self.dataloc)

    def __getitem__(self, item):
        mesh_pos, faces, node_type, t, velocity, pressure = get_data(self.dataloc[item], self.window_length, self.mode)
        faces = torch.from_numpy(faces).long()
        mesh_pos = torch.from_numpy(mesh_pos).float()
        velocity = torch.from_numpy(velocity).float()
        pressure = torch.from_numpy(pressure).float()
        edges = faces_to_edges(faces)  # Convert triangles to edges (pairs of indices)
        node_type = torch.from_numpy(node_type).long()

        if self.apply_onehot:
            node_type = one_hot(node_type, num_classes=9).squeeze(-2)

        if self.do_normalization:
            s0_mean, s0_std = 170.1, 71.06
            s1_mean, s1_std = -1.183, 46.73
            s2_mean, s2_std = 9.935e+04, 8964

            v_means = torch.tensor([s0_mean, s1_mean])
            v_stds = torch.tensor([50, 50])

            velocity = (velocity - v_means) / v_stds

            p_means = torch.tensor([s2_mean, s2_mean])
            p_stds = torch.tensor([s2_std, s2_std])

            pressure = (pressure - p_means) / p_stds
            # # Normalise states
            # masks = masks.unsqueeze(2).repeat(1, 1, 3, 1, 1).bool()
            # masks = masks.expand_as(states)
            # expanded_means = means.expand_as(states)
            # expanded_stds = stds.expand_as(states)
            #
            # states[~masks] = states[~masks] - expanded_means[~masks]
            # states[~masks] = states[~masks] / expanded_stds[~masks]
            #velocity, pressure = self.normalize(velocity, pressure)

        output = {'mesh_pos': mesh_pos,
                  'edges': edges,
                  'velocity': velocity,
                  'pressure': pressure,
                  'node_type': node_type}

        if self.with_cells:
            output['cells'] = faces

        if self.with_cluster:
            # If you want to use our clusters, you need to download them, and update this path
            cluster_path = self.dataloc[item]
            assert os.path.exists(cluster_path), f'Pre-computed cluster are not found, check path: {cluster_path}.\n' \
                                                 f'or update the path in the Dataloader.'
            if self.n_cluster == 0:
                clusters = torch.arange(mesh_pos.shape[1] + 1).view(1, -1, 1).repeat(velocity.shape[0], 1, 1)
            else:
                save_name = cluster_path.split("/")[-1][:-4]
                clusters = np.load(os.path.join(self.fn, f"constrained_kmeans_{self.n_cluster}_{save_name}.npy"),
                                   mmap_mode='r')[t:t + self.window_length].copy()
                clusters = torch.from_numpy(clusters).long()

            output['cluster'] = clusters

        return output

    # def normalize(self, velocity=None, pressure=None):
    #     if pressure is not None:
    #         pressure_shape = pressure.shape
    #         mean = torch.tensor([0.8845, -0.0002054]).to(pressure.device)
    #         std = torch.tensor([0.5875, 0.1286]).to(pressure.device)
    #         pressure = pressure.reshape(-1, 2)
    #         pressure = (pressure - mean) / std
    #         pressure = pressure.reshape(pressure_shape)
    #     if velocity is not None:
    #         velocity_shape = velocity.shape
    #         mean = torch.tensor([0.04064, 0.04064]).to(velocity.device).view(-1, 2)
    #         std = torch.tensor([0.2924, 0.2924]).to(velocity.device).view(-1, 2)
    #         velocity = velocity.reshape(-1, 2)
    #         velocity = (velocity - mean) / std
    #         velocity = velocity.reshape(velocity_shape)
    #
    #     return velocity, pressure
    #
    # def denormalize(self, velocity=None, pressure=None):
    #     if pressure is not None:
    #         pressure_shape = pressure.shape
    #         mean = torch.tensor([0.8845, -0.0002054]).to(pressure.device)
    #         std = torch.tensor([0.5875, 0.1286]).to(pressure.device)
    #         pressure = pressure.reshape(-1, 2)
    #         pressure = (pressure * std) + mean
    #         pressure = pressure.reshape(pressure_shape)
    #     if velocity is not None:
    #         velocity_shape = velocity.shape
    #         mean = torch.tensor([0.04064, 0.04064]).to(velocity.device).view(-1, 2)
    #         std = torch.tensor([0.2924, 0.2924]).to(velocity.device).view(-1, 2)
    #         velocity = velocity.reshape(-1, 2)
    #         velocity = velocity * std + mean
    #         velocity = velocity.reshape(velocity_shape)
    #
    #     return velocity, pressure


def get_data(path, window_length, mode):
    # Time sampling is random during training, but set to a fix value during test and valid, to ensure repeatability.
    t = 0 if window_length == 600 else random.randint(0, 600 - window_length)
    t = 100 if mode != "train" and window_length != 600 else t

    with open(f"{path}", 'rb') as f:
        save_data = pickle.load(f)  # ['faces', 'mesh_pos', 'velocity', 'pressure']
    pos = save_data['mesh_pos']
    faces = save_data['cells']
    node_type = save_data['node_type'].squeeze()

    V = save_data['velocity'][t:t + window_length]
    P = save_data['pressure'][t:t + window_length]

    P = np.repeat(P, 2, axis=-1)
    pos = np.repeat(pos[np.newaxis], window_length, axis=0)
    node_type = np.repeat(node_type[np.newaxis], window_length, axis=0)

    # Remove outer region
    x_mask = (pos[:, :, 0] > -.42) & (pos[:, :, 0] < 1.71)
    y_mask = (pos[:, :, 1] > -.64) & (pos[:, :, 1] < 0.85)
    mask = x_mask & y_mask
    mask = mask[0]

    pos = pos[:, mask]
    V = V[:,mask]
    P = P[:,mask]
    node_type = node_type[:, mask]

    # Filter faces
    wanted_nodes = np.squeeze(np.nonzero(mask))
    # Make a mapping from all nodes to wanted nodes, unwanted nodes are set to 0
    all_nodes = np.zeros(len(mask), dtype=np.int64)
    all_nodes[mask] = np.arange(len(wanted_nodes), dtype=np.int64)

    # Filter out faces that are not in the mask
    face_mask = np.isin(faces, wanted_nodes).all(axis=1)
    faces = faces[face_mask]
    faces = all_nodes[faces]

    faces = np.repeat(faces[np.newaxis], window_length, axis=0)

    return pos, faces, node_type, t, V, P


def faces_to_edges(faces):
    edges = torch.cat([faces[:, :, :2], faces[:, :, 1:], faces[:, :, ::2]], dim=1)

    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)

    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    unique_edges = torch.unique(packed_edges, dim=1)
    unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=1)

    return unique_edges


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.tri as mtri
    from matplotlib import pyplot as plt

    def collate(X):
        """ Convoluted function to stack simulations together in a batch. Basically, we add ghost nodes
        and ghost edges so that each sim has the same dim. This is useless when batchsize=1 though..."""
        N_max = max([x["mesh_pos"].shape[-2] for x in X])
        E_max = max([x["edges"].shape[-2] for x in X])
        C_max = max([x["cluster"].shape[-2] for x in X])

        for batch, x in enumerate(X):
            # This step add fantom nodes to reach N_max + 1 nodes
            for key in ['mesh_pos', 'velocity', 'pressure']:
                tensor = x[key]
                T, N, S = tensor.shape
                x[key] = torch.cat([tensor, torch.zeros(T, N_max - N + 1, S)], dim=1)

            tensor = x["node_type"]
            T, N, S = tensor.shape
            x["node_type"] = torch.cat([tensor, 2 * torch.ones(T, N_max - N + 1, S)], dim=1)

            x["cluster_mask"] = torch.ones_like(x["cluster"])
            x["cluster_mask"][x["cluster"] == -1] = 0
            x["cluster"][x["cluster"] == -1] = N_max

            if x["cluster"].shape[1] < C_max:
                c = x["cluster"].shape[1]
                x["cluster"] = torch.cat(
                    [x["cluster"], N_max * torch.ones(x["cluster"].shape[0], C_max - c, x['cluster'].shape[-1])], dim=1)
                x["cluster_mask"] = torch.cat(
                    [x["cluster_mask"], torch.zeros(x["cluster_mask"].shape[0], C_max - c, x['cluster'].shape[-1])], dim=1)

            edges = x['edges']
            T, E, S = edges.shape
            x['edges'] = torch.cat([edges, N_max * torch.ones(T, E_max - E + 1, S)], dim=1)

            x['mask'] = torch.cat([torch.ones(T, N), torch.zeros(T, N_max - N + 1)], dim=1)

        output = {key: None for key in X[0].keys()}
        for key in output.keys():
            if key != "example":
                output[key] = torch.stack([x[key] for x in X], dim=0)
            else:
                output[key] = [x[key] for x in X]

        return output


    train_dataset = AirfoilDataset("./ds/MGN/airfoil_dataset", mode="train", window_length=6, with_cluster=True,
                                    n_cluster=0, normalize=True)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                                  pin_memory=True, collate_fn=collate)

    for i, sample in enumerate(train_dataloader):
        # mesh_pos, edges, V, P, node_type, clust, clust_mask, mask = sample
        mesh_pos = sample['mesh_pos']
        edges = sample['edges']
        V = sample['velocity']
        node_type = sample['node_type'].bool()

        print(f'{node_type.shape = }')

        mask = node_type[0, 0, :, 2] == 1

        plot_V = V[0, 0, :, 0]
        plot_V[mask] = 0.

        print(f'{plot_V.max() = }, {plot_V.min() = }')

        # Create a Delaunay triangulation
        triang = mtri.Triangulation(mesh_pos[0, 0, :, 0], mesh_pos[0, 0,:, 1])
        plt.tripcolor(triang, plot_V)
        plt.show()

        exit(4)

