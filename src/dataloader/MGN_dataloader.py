"""Plots a CFD trajectory rollout."""

import numpy as np
import torch
import os
import pickle
import random
from cprint import c_print

from visualisation.mesh_utils import to_grid


def num_patches(dim_size, kern_size, stride, padding=0):
    """
    Returns the number of patches that can be extracted from an image
    """
    return (dim_size + 2 * padding - kern_size) // stride + 1


class MGNDataloader:
    """ Load a single timestep from the dataset."""

    def __init__(self, load_dir, resolution=512, patch_size=(32, 32), stride=(32, 32)):
        self.load_dir = load_dir
        self.resolution = resolution
        self.patch_size = patch_size
        self.stride = stride

        self.save_files = sorted([f for f in os.listdir(f"{self.load_dir}/") if f.endswith('.pkl')])

        # Get typical min/max values
        state, _ = self._get_step(save_file=self.save_files[1], step_num=20)
        self.ds_min_max = [(state[0].min(), state[0].max()), (state[1].min(), state[1].max()), (state[2].min(), state[2].max())]

        # Calculate number of patches, assuming stride = patch_size
        x_px, y_px = state.shape[1:]
        self.N_x_patch, self.N_y_patch = num_patches(x_px, patch_size[0], stride[0]), num_patches(y_px, patch_size[1], stride[1])
        self.N_patch = self.N_x_patch * self.N_y_patch

    def ds_get(self, save_file=None, step_num=None):
        """ Returns image from a given save and step patched."""
        if type(save_file) == int:
            save_file = f'save_{save_file}.pkl'
        else:
            save_file = random.choice(self.save_files)
        if step_num is None:
            step_num = np.random.randint(0, 100)

        state, mask = self._get_step(save_file, step_num=step_num)

        # Patch mask with state
        to_patch = np.concatenate([state, mask[None, :, :]], axis=0)
        to_patch = torch.from_numpy(to_patch).float()
        patches = self._patch(to_patch)
        state, mask = patches[:-1], patches[-1]

        return state, mask.bool()

    def _get_step(self, save_file, step_num, interp_type='linear'):
        """
        Returns all interpolated measurements for a given step
        """

        with open(f"{self.load_dir}/{save_file}", 'rb') as f:
            save_data = pickle.load(f)  # ['faces', 'mesh_pos', 'velocity', 'pressure']

        pos = save_data['mesh_pos']
        faces = save_data['cells']

        Vx = save_data['velocity'][step_num][:, 0]
        Vy = save_data['velocity'][step_num][:, 1]
        P = save_data['pressure'][step_num][:, 0]

        Vx_interp, Vx_mask = to_grid(pos, Vx, faces, grid_res=self.resolution, type=interp_type)
        Vy_interp, Vy_mask = to_grid(pos, Vy, faces, grid_res=self.resolution, type=interp_type)
        P_interp, P_mask = to_grid(pos, P, faces, grid_res=self.resolution, mask_interp=True, type=interp_type)

        Vx_interp, Vy_interp, P_interp = Vx_interp.astype(np.float32), Vy_interp.astype(np.float32), P_interp.astype(np.float32)
        step_state = np.stack([Vx_interp, Vy_interp, P_interp], axis=0)

        return step_state, P_mask

    def _patch(self, states: torch.Tensor):
        """
        Returns a list of patches of size patch_size from the states
        """
        C = states.shape[0]

        states = states.unsqueeze(0)
        patches = torch.nn.functional.unfold(states, kernel_size=self.patch_size, stride=self.stride)

        # Reshape patches to (N, C, H, W, num_patches)
        h, w = self.patch_size
        patches_reshaped = patches.view(C, h, w, patches.size(2))

        return patches_reshaped


class MGNSeqDataloader(MGNDataloader):
    """ Load a sequence of steps from the dataset. """

    def __init__(self, load_dir, resolution=512, patch_size=(32, 32), stride=(32, 32), seq_len=10, seq_interval=1):
        super().__init__(load_dir, resolution, patch_size, stride)
        self.seq_len = seq_len
        self.seq_interval = seq_interval

    def ds_get(self, save_file=None, step_num=0):
        """ Returns image from a given save and step patched.
            Last state has no corresponding diff
            Return:
                 state.shape = (seq_len, num_patches, 3, H, W)
                 diff.shape = (seq_len - 1, num_patches, 3, H, W)
                 mask.shape(num_patches, H, W)
        """
        if type(save_file) == int:
            save_file = f'save_{save_file}.pkl'
        else:
            save_file = random.choice(self.save_files)

        max_step_num = 600 - self.seq_len * self.seq_interval
        if step_num is None:
            step_num = np.random.randint(0, max_step_num)
        if step_num > max_step_num:
            c_print(f"Step number {step_num} too high, setting to max step number {max_step_num}", 'red')
            step_num = max_step_num

        # Load multiple states
        states = []
        for i in range(step_num, step_num + self.seq_len * self.seq_interval, self.seq_interval):
            state, mask = self._get_step(save_file, step_num=i)

            # Patch mask with state
            to_patch = np.concatenate([state, mask[None, :, :]], axis=0)
            to_patch = torch.from_numpy(to_patch).float()
            patches = self._patch(to_patch)
            state, mask = patches[:-1], patches[-1]

            state = torch.permute(state, [3, 0, 1, 2])
            mask = torch.permute(mask, [2, 0, 1])

            states.append(state)

        states = torch.stack(states, dim=0)
        diffs = states[1:] - states[:-1]  # shape = (seq_len, num_patches, C, H, W)

        return states, diffs, mask.bool()

    def get_sequence(self, save_file=None, step_num=0):
        """
        Returns as all patches as a single sequence, ready to be encoded by the LLM as a single element of batch.
        Return:
             state.shape = (1, (seq_len - 1) * num_patches, 3, H, W)
             diff.shape = (1, (seq_len - 1)  * num_patches, 3, H, W)
             patch_idx: [x_idx, y_idx, t_idx] for each patch

        """
        states, diffs, mask = self.ds_get(save_file, step_num)
        states = states[:-1]

        seq_dim = (self.seq_len - 1) * self.N_patch
        states = states.view(1, seq_dim, 3, self.patch_size[0], self.patch_size[1])
        diffs = diffs.view(1, seq_dim, 3, self.patch_size[0], self.patch_size[1])

        # Get positions / times for each patch
        arange = np.arange(seq_dim)
        x_idx = arange % self.N_x_patch
        y_idx = (arange // self.N_x_patch) % self.N_y_patch
        t_idx = arange // self.N_patch

        position_ids = np.stack([x_idx, y_idx, t_idx], axis=1)

        return states, diffs, mask, torch.from_numpy(position_ids)


def plot_patches(save_num, step_num, patch_no):
    dl = MGNDataloader(load_dir="../ds/MGN/cylinder_dataset", resolution=512, patch_size=(32, 32), stride=(32, 32))

    state, mask = dl.ds_get(save_num, step_num)

    p_show = state[:, :, :, patch_no].numpy()
    p_show = np.transpose(p_show, (2, 1, 0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        axes[i].imshow(p_show[:, :, i], vmin=-0.5, vmax=0.75)
    plt.show()


def plot_all_patches():
    load_no = 0
    step_num = 50
    patch_size, stride = (32, 32), (32, 32)

    seq_dl = MGNSeqDataloader(load_dir="../ds/MGN/cylinder_dataset", resolution=512, patch_size=patch_size, stride=stride, seq_len=5, seq_interval=2)
    state, diff, mask = seq_dl.ds_get(save_file=load_no, step_num=step_num)

    x_count, y_count = seq_dl.N_x_patch, seq_dl.N_y_patch

    p_shows = state[0]
    fig, axes = plt.subplots(y_count, x_count, figsize=(16, 4))
    for i in range(y_count):
        for j in range(x_count):
            p_show = p_shows[i + j * y_count].numpy()
            p_show = np.transpose(p_show, (2, 1, 0))
            for k in range(5):
                min, max = seq_dl.ds_min_max[0]

                axes[i, j].imshow(p_show[:, :, 0], vmin=min, vmax=max)
                axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

    p_shows = diff[0]

    fig, axes = plt.subplots(y_count, x_count, figsize=(16, 4))
    for i in range(y_count):
        for j in range(x_count):
            p_show = p_shows[i + j * y_count].numpy()
            p_show = np.transpose(p_show, (2, 1, 0))
            for k in range(3):
                min, max = -0.005, 0.005  # seq_dl.ds_min_max[0]

                axes[i, j].imshow(p_show[:, :, 0], vmin=min, vmax=max)
                axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # plot_patches(None, 10, 20)
    plot_all_patches()
