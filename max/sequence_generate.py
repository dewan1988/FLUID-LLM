import numpy as np
import torch

from dataloader.MGN_dataloader import MGNSeqDataloader
from dataloader.mesh_utils import plot_patches


def next_state(old_state, diff, mask) -> torch.Tensor:
    """ Model prediction for state t+1 by adding on diff. Using mask for BC.
        shape: (BS, N_patch, 3, H, W)
    """
    new_state = old_state + diff
    new_state[mask] = old_state[mask]
    return new_state


def main():
    ds = MGNSeqDataloader(load_dir="../ds/MGN/cylinder_dataset", resolution=240,
                          patch_size=(16, 16), stride=(16, 16), seq_len=5, seq_interval=20)
    patch_shape = ds.N_x_patch, ds.N_y_patch
    N_patch = ds.N_patch
    states, diffs, masks, pos_embed = ds.ds_get()

    print(f'{states.shape = }, {diffs.shape = }, {masks.shape = }, {pos_embed.shape = }')

    get_num = 0
    get_slice = slice(get_num * N_patch, (get_num + 1) * N_patch)
    test_state = states[get_slice]
    test_diff = diffs[get_slice]
    test_mask = masks[get_slice]

    new_state = next_state(test_state, test_diff, test_mask)

    print("Current state")
    plot_patches(states[get_slice, 0], patch_shape)
    print("Next state")
    plot_patches(new_state[:, 0], patch_shape)


if __name__ == '__main__':
    from utils import set_seed

    set_seed(2)
    main()
