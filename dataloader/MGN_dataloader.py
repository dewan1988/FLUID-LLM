"""Plots a CFD trajectory rollout."""

import pickle

from visualisation.mesh_utils import to_grid
import numpy as np
import torch


class MGNDataloader:
    def __init__(self, load_dir, resolution=1024, patch_size=(64, 64), stride=(32, 32)):
        self.load_dir = load_dir
        self.resolution = resolution
        self.patch_size = patch_size
        self.stride = stride

    def ds_get(self, save_num=None, step_num=None):
        """ Returns image from a given save and step patched."""
        if save_num is None:
            save_num = np.random.randint(0, 10)
        if step_num is None:
            step_num = np.random.randint(0, 100)


        state, mask = self._get_step(save_num, step_num=step_num)

        # Patch mask with state
        to_patch = np.concatenate([state, mask[None, :, :]], axis=0)
        patches = self._patch(to_patch)[0]
        state, mask = patches[:-1], patches[-1]

        return state, mask.bool()

    def _get_step(self, save_num, step_num, interp_type='linear'):
        """
        Returns all interpolated measurements for a given step
        """

        with open(f"{self.load_dir}/save_{save_num}.pkl", 'rb') as f:
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

    def _patch(self, states):
        """
        Returns a list of patches of size patch_size from the states
        """
        C = states.shape[0]

        states = torch.from_numpy(states).unsqueeze(0)
        patches = torch.nn.functional.unfold(states, kernel_size=self.patch_size, stride=self.stride)

        # Reshape patches to (N, C, H, W, num_patches)
        h, w = self.patch_size
        patches_reshaped = patches.view(1, C, h, w, patches.size(2))

        return patches_reshaped


def plot_patches(save_num, step_num, patch_no):
    from matplotlib import pyplot as plt

    dl = MGNDataloader(load_dir="../ds/MGN/cylinder_dataset", resolution=512, patch_size=(32, 32), stride=(32, 32))

    state, mask = dl.ds_get(save_num, step_num)

    p_show = state[:, :, :, patch_no].numpy()
    p_show = np.transpose(p_show, (2, 1, 0))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        axes[i].imshow(p_show[:, :, i], vmin=-0.5, vmax=0.75)
    plt.show()


if __name__ == '__main__':
    plot_patches(0, 10, 20)
