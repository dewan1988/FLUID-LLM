"""Plots a CFD trajectory rollout."""

import numpy as np
import os
import pickle
import random
from cprint import c_print
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from dataloader.mesh_utils import to_grid, get_mesh_interpolation


def num_patches(dim_size, kern_size, stride, padding=0):
    """
    Returns the number of patches that can be extracted from an image
    """
    return (dim_size + 2 * padding - kern_size) // stride + 1


class MGNDataset(Dataset):
    """ Load a single timestep from the dataset."""

    def __init__(self, load_dir, resolution: int, patch_size: tuple, stride: tuple, seq_len: int, seq_interval=1,
                 pad=True, mode="train", normalize=True, noise=None):
        super(MGNDataset, self).__init__()

        assert mode in ["train", "valid", "test"]

        self.mode = mode
        self.load_dir = load_dir
        self.resolution = resolution
        self.patch_size = patch_size
        self.stride = stride
        self.pad = pad
        self.seq_len = seq_len
        self.seq_interval = seq_interval
        self.max_step_num = 600 - self.seq_len * self.seq_interval
        self.normalize = normalize
        self.noise = noise

        self.save_files = sorted([f for f in os.listdir(f"{self.load_dir}/") if f.endswith('.pkl')])
        # Load a random file to get min and max values and patch size
        triang, tri_index, grid_x, grid_y, save_data = self._load_step(self.save_files[1])
        state, _ = self._get_step(triang, tri_index, grid_x, grid_y, save_data, step_num=20)

        # Get min and max values for each channel
        self.ds_min_max = [(state[0].min(), state[0].max()), (state[1].min(), state[1].max()), (state[2].min(), state[2].max())]

        # Calculate number of patches, assuming stride = patch_size
        x_px, y_px = state.shape[1:]

        self.N_x_patch, self.N_y_patch = num_patches(x_px, patch_size[0], stride[0]), num_patches(y_px, patch_size[1], stride[1])
        self.N_patch = self.N_x_patch * self.N_y_patch

    def __getitem__(self, idx):
        """
        Returns as all patches as a single sequence for file with index idx, ready to be encoded by the LLM as a single element of batch.
        Return:
             state.shape = ((seq_len - 1), num_patches, 3, H, W)
             diff.shape = ((seq_len - 1), num_patches, 3, H, W)
             patch_idx: [x_idx, y_idx, t_idx] for each patch

        """
        # Time sampling is random during training, but set to a fix value during test and valid, to ensure repeatability.
        step_num = random.randint(0, self.max_step_num)
        step_num = 100 if self.mode in ["test", "valid"] else step_num
        return self.ds_get(save_file=self.save_files[idx], step_num=step_num)

    def ds_get(self, save_file=None, step_num=None):
        """
        Returns as all patches as a single sequence, ready to be encoded by the LLM as a single element of batch.
        Return:
             states.shape = ((seq_len - 1), num_patches, 3, H, W)
             patch_idx: [x_idx, y_idx, t_idx] for each patch
        """

        to_patches = self._get_full_seq(save_file, step_num)
        to_patches = torch.from_numpy(to_patches).float()

        patches = self._patch(to_patches)
        states = patches[:, :-1]
        masks = patches[:, -1]
        # Permute to (seq_len, num_patches, C, H, W)
        states = torch.permute(states, [0, 4, 1, 2, 3])
        masks = torch.permute(masks, [0, 3, 1, 2])

        if self.normalize:
            states = self._normalize(states)

        diffs = states[1:] - states[:-1]  # shape = (seq_len, num_patches, C, H, W)
        next_state = states[1:]

        # Compute targets and discard last state that has no diff
        input_states = states[:-1]

        # Reshape into a continuous sequence
        masks = masks[1:].unsqueeze(2).repeat(1, 1, 3, 1, 1).bool()

        return input_states, next_state, diffs, masks, self._get_pos_id()

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

        if self.pad:
            step_state, P_mask = self._pad(step_state, P_mask)

        return step_state, P_mask

    def _patch(self, states: torch.Tensor):
        """
        Patches a batch of images.
        Returns a tensor of shape (bs, C, patch_h, patch_w, num_patches)
        """
        bs, C, _, _ = states.shape
        ph, pw = self.patch_size

        patches = F.unfold(states, kernel_size=self.patch_size, stride=self.stride)

        # Reshape patches to (bs, C, ph, pw, num_patches)
        patches_reshaped = patches.view(bs, C, ph, pw, patches.size(2))
        return patches_reshaped

    def _pad(self, state, mask):
        """ Pad state and mask so they can be evenly patched."""
        _, w, h = state.shape
        pad_width = (-w % self.patch_size[0])
        pad_height = (-h % self.patch_size[1])

        padding = (
            (0, 0),  # No padding on channel dimension
            (pad_width // 2, pad_width - pad_width // 2),  # Left, Right padding
            (pad_height // 2, pad_height - pad_height // 2),  # Top, Bottom padding
        )

        padding = np.array(padding)
        state_pad = np.pad(state, padding, mode='constant', constant_values=0)
        mask_pad = np.pad(mask, padding[1:], mode='constant', constant_values=1)
        return state_pad, mask_pad

    def _load_step(self, save_file):
        """ Load save file from disk and calculate mesh interpolation triangles"""

        with open(f"{self.load_dir}/{save_file}", 'rb') as f:
            save_data = pickle.load(f)  # ['faces', 'mesh_pos', 'velocity', 'pressure']
        pos = save_data['mesh_pos']
        faces = save_data['cells']

        triang, tri_index, grid_x, grid_y = get_mesh_interpolation(pos, faces, self.resolution)

        return triang, tri_index, grid_x, grid_y, save_data

    def _get_full_seq(self, save_file=None, step_num=None):
        """ Returns numpy arrays of sequence, ready to be patched.
            Return shape: (seq_len, C+1, H, W)
        """
        if save_file is None:
            save_file = random.choice(self.save_files)
        elif isinstance(save_file, int):
            save_file = self.save_files[save_file]

        if step_num is None:
            step_num = np.random.randint(0, self.max_step_num)
        if step_num > self.max_step_num:
            c_print(f"Step number {step_num} too high, setting to max step number {self.max_step_num}", 'red')
            step_num = self.max_step_num

        triang, tri_index, grid_x, grid_y, save_data = self._load_step(save_file)

        to_patches = []
        for i in range(step_num, step_num + self.seq_len * self.seq_interval, self.seq_interval):
            state, mask = self._get_step(triang, tri_index, grid_x, grid_y, save_data, step_num=i)

            # Patch mask with state
            to_patch = np.concatenate([state, mask[None, :, :]], axis=0)
            to_patches.append(to_patch)

        return np.stack(to_patches)

    def _normalize(self, states):
        """ states.shape = [seq_len, N_patch, 3, patch_x, patch_y] """
        # State 0:  0.8845, 0.5875
        # Diff 0: 1.78e-05, 0.02811
        # 0.21, 0.0198
        # State 1: -0.0002054, 0.1286
        # Diff 1: -9.47e-07, 0.02978
        # 0.0166, 0.0202
        # State 2:  0.04064, 0.2924
        # Diff 2: -0.00288, 0.04859
        # 0.0852, 0.0433

        s0_mean, s0_var = 0.823, 0.3315
        s1_mean, s1_var = 0.0005865, 0.01351
        s2_mean, s2_var = 0.04763, 0.07536

        means = torch.tensor([s0_mean, s1_mean, s2_mean]).reshape(1, 1, 3, 1, 1)
        stds = torch.tensor([0.275, 0.275, 0.275]).reshape(1, 1, 3, 1, 1)

        # Normalise states
        states = states - means
        states = states / stds

        return states

    def _get_pos_id(self):
        # Get positions / times for each patch
        seq_dim = (self.seq_len - 1) * self.N_patch
        arange = np.arange(seq_dim)
        x_idx = arange % self.N_x_patch
        y_idx = (arange // self.N_x_patch) % self.N_y_patch
        t_idx = arange // self.N_patch
        position_ids = np.stack([x_idx, y_idx, t_idx], axis=1).reshape(self.seq_len - 1, self.N_patch, 3)
        return torch.from_numpy(position_ids)

    def __len__(self):
        return len(self.save_files)


def plot_all_patches():
    patch_size = (16, 16)

    seq_dl = MGNDataset(load_dir="./ds/MGN/cylinder_dataset/train", resolution=238, patch_size=patch_size, stride=patch_size,
                        seq_len=10, seq_interval=2, normalize=False)

    ds = DataLoader(seq_dl, batch_size=8, shuffle=True)

    for batch in ds:
        state, next_state, diffs, mask, pos_id = batch
        print(f'{state.shape = }, {diffs.shape = }. {pos_id.shape = }')
        break

    N_x, N_y = seq_dl.N_x_patch, seq_dl.N_y_patch
    print(f'{N_x = }, {N_y = }')

    p_shows = state[0, 0, :, 0]
    plot_patches(p_shows, (seq_dl.N_x_patch, seq_dl.N_y_patch))

    p_shows = diffs[0, 0, :, 0]
    plot_patches(p_shows, (seq_dl.N_x_patch, seq_dl.N_y_patch))

    p_shows = next_state[0, 0, :, 0]
    plot_patches(p_shows, (seq_dl.N_x_patch, seq_dl.N_y_patch))


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from dataloader.mesh_utils import plot_patches
    from utils import set_seed

    set_seed()
    plot_all_patches()
