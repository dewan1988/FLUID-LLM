"""Plots a CFD trajectory rollout."""

import numpy as np
import os
import pickle
import random
from cprint import c_print
import time
from dataloader.synthetic.solver_node import WaveConfig, PDESolver2D
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader.mesh_utils import to_grid, get_mesh_interpolation


def num_patches(dim_size, kern_size, stride, padding=0):
    """
    Returns the number of patches that can be extracted from an image
    """
    return (dim_size + 2 * padding - kern_size) // stride + 1


class SynthDS(Dataset):
    """ Load a single timestep from the dataset."""

    def __init__(self, resolution: int, patch_size: tuple, stride: tuple, seq_len: int, fit_diffs: bool, seq_interval=1,
                 pad=True, mode="train", normalize=True):
        super().__init__()

        assert mode in ["train", "valid", "test"]

        self.mode = mode
        self.resolution = resolution
        self.patch_size = patch_size
        self.stride = stride
        self.pad = pad
        self.seq_len = seq_len
        self.seq_interval = seq_interval
        self.max_step_num = 600 - self.seq_len * self.seq_interval
        self.normalize = normalize
        self.fit_diffs = fit_diffs

        self.data_gen = PDESolver2D(WaveConfig())

        # Load a random file to get min and max values and patch size
        ys, bc_mask = self._load_step()
        state, _ = self._get_step(ys, bc_mask, 5)

        # Get min and max values for each channel
        self.ds_min_max = [(state[0].min(), state[0].max()), (state[1].min(), state[1].max()), (state[2].min(), state[2].max())]

        # Calculate number of patches, assuming stride = patch_size
        x_px, y_px = state.shape[1:]

        self.N_x_patch, self.N_y_patch = num_patches(x_px, patch_size[0], stride[0]), num_patches(y_px, patch_size[1], stride[1])
        self.N_patch = self.N_x_patch * self.N_y_patch

    def __getitem__(self, idx):
        """
        Returns as all patches as a single sequence for file with index idx, ready to be encoded by the LLM as a single element of batch.
        """
        return self.ds_get()

    def ds_get(self, step_num=None):
        """
        Returns as all patches as a single sequence, ready to be encoded by the LLM as a single element of batch.
        Return:
             state.shape = ((seq_len - 1) * num_patches, 3, H, W)
             diff.shape = ((seq_len - 1)  * num_patches, 3, H, W)
             patch_idx: [x_idx, y_idx, t_idx] for each patch
        """

        to_patches = self._get_full_seq(step_num)

        states, diffs, mask = self._ds_get_pt(to_patches)

        # Get positions / times for each patch
        seq_dim = (self.seq_len - 1) * self.N_patch
        arange = np.arange(seq_dim)
        x_idx = arange % self.N_x_patch
        y_idx = (arange // self.N_x_patch) % self.N_y_patch
        t_idx = arange // self.N_patch

        position_ids = np.stack([x_idx, y_idx, t_idx], axis=1)

        states, diffs = states.clamp(-5, 5), diffs.clamp(-5, 5)

        return states, diffs, mask, torch.from_numpy(position_ids)

    def _get_step(self, ys, bc_mask, step_num):
        """
        Returns all interpolated measurements for a given step, including padding.
        """

        ys = ys[step_num]
        ys = torch.cat((ys, ys[0:1]), dim=0)
        # bc_mask = np.repeat(bc_mask[None], 3, axis=0)

        if self.pad:
            step_state, P_mask = self._pad(ys, bc_mask)

        return step_state, P_mask

    def _patch(self, states: torch.Tensor):
        """
        Patches a batch of images.
        Returns a tensor of shape (bs, C, patch_h, patch_w, num_patches)
        """
        bs, C, _, _ = states.shape
        ph, pw = self.patch_size
        # states = states.unsqueeze(0)

        st = time.time()
        patches = torch.nn.functional.unfold(states, kernel_size=self.patch_size, stride=self.stride)

        if time.time() - st > 0.1:
            c_print(f"Time to patch: {time.time() - st:.3g}s", 'green')

        # Reshape patches to (bs, N, C, ph, pw, num_patches)
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

    def _load_step(self):
        """ Load save file from disk and calculate mesh interpolation triangles"""

        self.data_gen.set_init_cond()
        sol, bc_mask = self.data_gen.solve()
        return sol, bc_mask

    def _get_full_seq(self, step_num):
        """ Returns numpy arrays of sequence, ready to be patched.
            Required to avoid pytorch multiprocessing bug.

            Return shape: (seq_len, C+1, H, W)
        """
        sol, bc_mask = self._load_step()

        to_patches = []
        for i in range(0, self.seq_len * self.seq_interval, self.seq_interval):
            state, mask = self._get_step(sol, bc_mask, step_num=i)

            # Patch mask with state
            to_patch = np.concatenate([state, mask[None, :, :]], axis=0)
            to_patches.append(to_patch)

        return np.stack(to_patches)

    def _ds_get_pt(self, to_patches: np.ndarray):
        """ Pytorch section of ds_get to avoid multiprocessing bug.
            to_patches.shape = (seq_len, C+1, H, W) where last channel dim is mask.
        """

        to_patches = torch.from_numpy(to_patches).float()

        patches = self._patch(to_patches)

        states = patches[:, :-1]
        masks = patches[:, -1]

        # Permute to (seq_len, num_patches, C, H, W)
        states = torch.permute(states, [0, 4, 1, 2, 3])
        masks = torch.permute(masks, [0, 3, 1, 2])

        if self.fit_diffs:
            target = states[1:] - states[:-1]  # shape = (seq_len, num_patches, C, H, W)
        else:
            target = states[1:]
        # Compute targets and discard last state that has no diff
        states = states[:-1]

        # Reshape into a continuous sequence
        seq_dim = (self.seq_len - 1) * self.N_patch
        states = states.reshape(seq_dim, 3, self.patch_size[0], self.patch_size[1])
        target = target.reshape(seq_dim, 3, self.patch_size[0], self.patch_size[1])

        # Reshape mask. All masks are the same
        masks = masks[:-1].reshape(seq_dim, 1, self.patch_size[0], self.patch_size[1]).repeat(1, 3, 1, 1)

        return states, target, masks.bool()

    def __len__(self):
        if self.mode == "train":
            return 1000
        else:
            return 125


def plot_all_patches():
    patch_size, stride = (16, 16), (16, 16)

    seq_dl = SynthDS(resolution=240, patch_size=patch_size, stride=stride,
                     seq_len=10, seq_interval=2, normalize=False, fit_diffs=True)
    ds = DataLoader(seq_dl, batch_size=1, num_workers=0)

    for batch in ds:
        state, diffs, mask, pos_id = batch
        if state.max() > 100:
            print(state.max())
        break

    x_count, y_count = seq_dl.N_x_patch, seq_dl.N_y_patch
    N_patch = seq_dl.N_patch

    show_dim = 0
    p_shows = state[0, :, show_dim]    # shape = (N_patch * seq_len, 16, 16)
    p_shows = p_shows.reshape(-1, N_patch, 16, 16)
    vmin, vmax = p_shows[0].min(), p_shows[0].max()
    print(f'{vmin = :.2g}, {vmax = :.2g}')
    for show_step in range(0, 9, 2):
        fig, axes = plt.subplots(y_count, x_count, figsize=(16, 4))
        for i in range(y_count):
            for j in range(x_count):
                p_show = p_shows[show_step, i + j * y_count].numpy()
                p_show = p_show.T
                axes[i, j].imshow(p_show[:, :], vmin=vmin, vmax=vmax)
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

    # p_shows = diffs[0]
    # fig, axes = plt.subplots(y_count, x_count, figsize=(16, 4))
    # for i in range(y_count):
    #     for j in range(x_count):
    #         p_show = p_shows[i + j * y_count + show_step*N_patch].numpy()
    #         p_show = np.transpose(p_show, (2, 1, 0))
    #
    #         min, max = -0.005, 0.005  # seq_dl.ds_min_max[0]
    #
    #         axes[i, j].imshow(p_show[:, :, 0], vmin=min, vmax=max)
    #         axes[i, j].axis('off')
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from utils import set_seed

    # set_seed(6)
    # plot_patches(None, 10, 20)
    plot_all_patches()
