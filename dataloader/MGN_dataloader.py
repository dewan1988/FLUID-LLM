"""Plots a CFD trajectory rollout."""

import pickle

from visualisation.mesh_utils import to_grid
import numpy as np
import torch


class MGNDataloader:
    def __init__(self, load_dir, patch_size=(64, 64), stride=(32, 32)):
        self.load_dir = load_dir
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

        Vx_interp, Vx_mask = to_grid(pos, Vx, faces, grid_res=1024, type=interp_type)
        Vy_interp, Vy_mask = to_grid(pos, Vy, faces, grid_res=1024, type=interp_type)
        P_interp, P_mask = to_grid(pos, P, faces, grid_res=1024, mask_interp=True, type=interp_type)

        Vx_interp, Vy_interp, P_interp = Vx_interp.astype(np.float32), Vy_interp.astype(np.float32), P_interp.astype(np.float32)
        step_state = np.stack([Vx_interp, Vy_interp, P_interp], axis=0)

        return step_state, P_mask

    def _patch(self, states):
        """
        Returns a list of patches of size patch_size from the states
        """
        C = states.shape[0]
        states = torch.from_numpy(states).unsqueeze(0)
        unfolder = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.stride, padding=(0, 0))
        patches = unfolder(states)

        # Reshape patches to (N, C, H, W, num_patches)
        h, w = self.patch_size
        patches_reshaped = patches.view(1, C, h, w, patches.size(2))

        return patches_reshaped


def plot_patches(save_num, step_num, patch_no):
    from matplotlib import pyplot as plt

    # rol = load_step(save_num)
    # state, mask = get_step(rol, step_num=step_num)
    state, mask = ds_get(save_num, step_num, patch_size=(64, 64), stride=(32, 32))

    # state = np.transpose(state, (2, 1, 0))
    # plt.imshow(state, origin='lower')
    # plt.show()

    p_show = state[:, :, :, patch_no].numpy()
    p_show = np.transpose(p_show, (2, 1, 0))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        axes[i].imshow(p_show[:, :, i])
    plt.show()


if __name__ == '__main__':
    DS = MGNDataloader(load_dir="../ds/MGN/cylinder_dataset")
    _state, _mask = DS.ds_get(0, 10, patch_size=(64, 64), stride=(32, 32))
    print(_state.shape, _mask.shape)

    # plot_patches(0, 10, 50)
