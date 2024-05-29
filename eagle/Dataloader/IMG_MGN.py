import os.path
import random
from torch.utils.data import Dataset
import numpy as np
import torch
import pickle

from src.dataloader.mesh_utils import get_mesh_interpolation, to_grid


class EagleDataset(Dataset):
    def __init__(self, data_path, mode="test", window_length=990, with_mesh=False):
        """ Eagle dataloader for images
        :param data_path: path to dataset (grid)
        :param mode: train, test or valid set
        :param window_length: length of the temporal window to sample the simulation
        :param with_mesh: load the irregular mesh, useful for evaluation purposes
        """

        super(EagleDataset, self).__init__()
        assert mode in ["train", "test", "valid"]

        self.window_length = window_length
        assert window_length <= 990, "window length must be smaller than 990"

        self.fn = os.path.join(data_path, mode)
        assert os.path.exists(self.fn), f"Path {self.fn} does not exist"

        self.dataloc = []
        for root, directories, files in os.walk(self.fn):
            for filename in files:
                filepath = os.path.join(root, filename)

                if filepath.endswith(".pkl"):
                    self.dataloc.append(filepath)

        self.mode = mode
        self.length = 990
        self.with_mesh = with_mesh

    def __len__(self):
        return len(self.dataloc)

    def _load_step(self, save_file):
        """ Load save file from disk and calculate mesh interpolation triangles"""
        with open(f"{save_file}", 'rb') as f:
            save_data = pickle.load(f)  # ['cells', 'mesh_pos', 'velocity', 'pressure', 'density' ]
        pos = save_data['mesh_pos']
        faces = save_data['cells']

        # # Mask of unwatned nodes
        # x_mask = (pos[:, 0] > -.5) & (pos[:, 0] < 2)
        # y_mask = (pos[:, 1] > -.75) & (pos[:, 1] < 0.75)
        # mask = x_mask & y_mask
        #
        # # Remove values
        # pos = pos[mask]
        # save_data['velocity'] = save_data['velocity'][:, mask]
        # save_data['pressure'] = save_data['pressure'][:, mask]
        # save_data['density'] = save_data['density'][:, mask]
        #
        # # Filter out faces that are not in the mask
        # wanted_nodes = np.squeeze(np.nonzero(mask))
        # # Make a mapping from all nodes to wanted nodes, unwanted nodes are set to 0
        # all_nodes = np.zeros(len(mask), dtype=np.int64)
        # all_nodes[mask] = np.arange(len(wanted_nodes), dtype=np.int64)
        # face_mask = np.isin(faces, wanted_nodes).all(axis=1)
        # faces = faces[face_mask]
        # faces = all_nodes[faces]

        triang, tri_index, grid_x, grid_y = get_mesh_interpolation(pos, faces, 238)

        return triang, tri_index, grid_x, grid_y, save_data

    def _get_step(self, triang, tri_index, grid_x, grid_y, save_data, step_num):
        """
        Returns all interpolated measurements for a given step, including padding.
        """
        Vx = save_data['velocity'][step_num][:, 0]
        Vy = save_data['velocity'][step_num][:, 1]
        P = save_data['pressure'][step_num][:, 0]

        Vx_interp, Vx_mask = to_grid(Vx, grid_x, grid_y, triang, tri_index)
        Vy_interp, Vy_mask = to_grid(Vy, grid_x, grid_y, triang, tri_index)
        P_interp, P_mask = to_grid(P, grid_x, grid_y, triang, tri_index)
        step_state = np.stack([Vx_interp, Vy_interp, P_interp], axis=0)

        return step_state, P_mask

    def __getitem__(self, item):
        # Time sampling is random during training, but set to a fix value during test and valid, to ensure repeatability.
        t = 0 if self.window_length == 600 else random.randint(0, 600 - self.window_length)
        t = 100 if self.mode != "train" and self.window_length != 600 else t

        triang, tri_index, grid_x, grid_y, save_data = self._load_step(self.dataloc[item])

        states, masks = [], []
        for i in range(t, t + self.window_length):
            state, mask = self._get_step(triang, tri_index, grid_x, grid_y, save_data, step_num=i)

            states.append(state)
            masks.append(mask)

        states = np.stack(states)
        masks = np.stack(masks)
        states = torch.from_numpy(states).float()

        output = {'states': self.normalize(states).permute(0, 2, 3, 1),
                  'mask': masks.copy(), # [:, np.newaxis].repeat(3, axis=1),
                  }
        #
        # if self.with_mesh:
        #     path = self.dataloc[item].replace("_img", "")
        #     assert os.path.exists(path), f"Can not find mesh files in {path}, please check the path in the dataloader"
        #     data = np.load(os.path.join(path, 'sim.npz'), mmap_mode='r')
        #     mesh_pos = data["pointcloud"][t:t + self.window_length].copy()
        #     Vx = data['VX'][t:t + self.window_length].copy()
        #     Vy = data['VY'][t:t + self.window_length].copy()
        #     Ps = data['PS'][t:t + self.window_length].copy()
        #     Pg = data['PG'][t:t + self.window_length].copy()
        #     velocity = np.stack([Vx, Vy], axis=-1)
        #     pressure = np.stack([Ps, Pg], axis=-1)
        #     node_type = data['mask'][t:t + self.window_length].copy()
        #
        #     output['mesh_pos'] = mesh_pos
        #     output['mesh_velocity'] = velocity
        #     output['mesh_pressure'] = pressure
        #     output['mesh_node_type'] = node_type

        return output

    def normalize(self, state):
        shape = state.shape

        mean = torch.tensor([0.823, 0.0005865, 0.04763]).to(state.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.275, 0.275, 0.275]).to(state.device).reshape(1, 3, 1, 1)

        state = (state - mean) / std
        return state.reshape(shape)

    def denormalize(self, state):
        shape = state.shape
        # print(f'{state.shape = }')
        # exit(7)
        mean = torch.tensor([0.823, 0.0005865, 0.04763]).to(state.device)#.reshape(1, 3, 1, 1)
        std = torch.tensor([0.275, 0.275, 0.275]).to(state.device)#.reshape(1, 3, 1, 1)

        state = state * std + mean
        return state.reshape(shape)

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # Test the dataloader
    dataset = EagleDataset(data_path="./ds/MGN/cylinder_dataset/", mode="train", window_length=10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for i, batch in enumerate(dataloader):
        state, mask = batch['states'], batch['mask']
        print(f'{state.shape = }, {mask.shape = }')
        plt.imshow(state[0, 0, :, :, 0].T)
        plt.show()
        break
