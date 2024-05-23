import numpy as np
import torch
from functools import lru_cache
from cprint import c_print

c_print("Monkey patching matplotlib", color='yellow')

import matplotlib
# Monkey patch plt grid interpolator.
from _triinterpolate import TriInterpolator as CustomTriInterpolator
from _triinterpolate import LinearTriInterpolator as CustomLinearTriInterpolator

# Replace the default interpolator with the custom one
matplotlib.tri.TriInterpolator = CustomTriInterpolator
matplotlib.tri.LinearTriInterpolator = CustomLinearTriInterpolator

from matplotlib import tri as mtri
import matplotlib.pyplot as plt


def plot_patches(state: torch.Tensor, N_patch: tuple):
    """Plot a series of patches in a grid, single channel.
     state.shape = (N_patch, H, W)"""
    x_count, y_count = N_patch

    # state.clamp_(0, 1)
    state = state.detach().float().cpu().numpy()
    v_min, v_max = state.min(), state.max()
    print(f'min: {v_min:.2g}, max: {v_max:.2g}, mean: {state.mean():.2g}, std: {state.std():.2g}')

    # Normalize state to [0, 1]
    state = (state - v_min) / (v_max - v_min)

    fig, axes = plt.subplots(y_count, x_count, figsize=(x_count, y_count))
    for i in range(y_count):
        for j in range(x_count):
            patch = state[i + j * y_count].T
            axes[i, j].imshow(patch, vmin=0, vmax=1)
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()


def plot_full_patches(state, N_patch: tuple, ax):
    """ Plot patches as a single image"""
    x_count, y_count = N_patch
    x_px, y_px = state.shape[2], state.shape[1]

    state = state.detach().float().cpu().numpy()
    full_img = np.zeros((y_count * y_px, x_count * x_px))
    for i in range(y_count):
        for j in range(x_count):
            start_row = i * y_px
            start_col = j * x_px
            end_row = start_row + y_px
            end_col = start_col + x_px

            full_img[start_row:end_row, start_col:end_col] = state[i + j * y_count].T

    ax.imshow(full_img)
    ax.axis('off')


@lru_cache(maxsize=5)
def grid_pos(x_min, x_max, y_min, y_max, grid_res):
    # Scale grid so long axis is grid_res with square grid
    long_axis = max(x_max - x_min, y_max - y_min)
    short_axis = min(x_max - x_min, y_max - y_min)
    ratio = short_axis / long_axis

    if x_max - x_min > y_max - y_min:
        x_points = grid_res
        y_points = int(grid_res * ratio)
    else:
        y_points = grid_res
        x_points = int(grid_res * ratio)

    grid_x, grid_y = np.mgrid[x_min:x_max:x_points * 1j, y_min:y_max:y_points * 1j]  # Adjust 100j as needed for resolution
    return grid_x.astype(np.float32), grid_y.astype(np.float32)


def to_grid(val, grid_x, grid_y, triang, tri_index):
    interp = mtri.LinearTriInterpolator(triang, val)
    mask_data = interp(grid_x, grid_y, tri_index=tri_index)

    mask = mask_data.mask
    data = mask_data.data

    data[mask] = 0.

    return data, mask


def get_mesh_interpolation(pos, faces, grid_res=256):
    """ Returns mesh interpolation properties for a given mesh.
        Can be cached for efficiency if mesh is the same.
    """

    x_min, y_min = np.min(pos, axis=0)
    x_max, y_max = np.max(pos, axis=0)
    grid_x, grid_y = grid_pos(x_min, x_max, y_min, y_max, grid_res)

    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], triangles=faces)
    tri_index = triang.get_trifinder()(grid_x, grid_y)

    return triang, tri_index, grid_x, grid_y


def plot_mesh(pos, faces, val):
    """Plots triangular mesh from positions of nodes and triangles defined using faces."""
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)

    plt.figure(figsize=(8, 8))
    plt.tripcolor(triang, val)  # or contourf
    plt.triplot(triang, "k-", lw=0.1)

    # plt.ylim([0., 0.5])
    plt.axis("equal")

    # plt.axis("off")
    plt.tight_layout()

    plt.show()


#
#
# def plot_mesh_smooth(pos, val, faces, grid_res):
#     """Plots a 2D image from positions of nodes and values of the nodes."""
#     grid_z, _ = to_grid(pos, val, faces, grid_res)
#     plt.figure(figsize=(12, 4))
#     plt.imshow(grid_z.T, extent=(np.min(pos[:, 0]), np.max(pos[:, 0]), np.min(pos[:, 1]), np.max(pos[:, 1])),
#                origin='lower', cmap='viridis')
#
#     plt.tight_layout()
#     plt.show()
#
#
# def inpaint(img, mask, method='telea'):
#     """Inpaints mask to avoid rough edges. """
#
#     min_val = np.min(img)
#     max_val = np.max(img)
#
#     img = (img - min_val) / (max_val - min_val)
#     img = (img * 255).astype(np.uint8)
#     mask = (mask * 255).astype(np.uint8)
#     if method == 'telea':
#         img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
#     elif method == 'ns':
#         img = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
#     else:
#         raise ValueError(f"Unknown inpainting method: {method}")
#
#     img = (img / 255) * (max_val - min_val) + min_val
#     return img.astype(np.float32)

if __name__ == "__main__":
    import pickle
    import time

    for _ in range(10):
        a = test_time()

    a = test_time(save_no=0)

    plt.imshow(a.T)
    plt.tight_layout()
    plt.show()
