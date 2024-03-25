"""Plots a sample from MeshGraphNets."""

import pickle
from matplotlib import pyplot as plt
from matplotlib import tri as mtri
import numpy as np


def plot_mesh(pos, faces, val):
    """Plots triangular mesh from positions of nodes and triangles defined using faces."""
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)

    plt.figure(figsize=(10, 4))
    plt.tripcolor(triang, val)  # or contourf
    plt.ylim([0., 0.5])
    plt.axis("equal")

    # plt.axis("off")
    plt.tight_layout()

    plt.show()


def plot_mesh_smooth(pos, val, faces, grid_res):
    """Plots a 2D image from positions of nodes and values of the nodes."""
    grid_z, _ = to_grid(pos, val, faces, grid_res)
    plt.figure(figsize=(12, 4))
    plt.imshow(grid_z.T, extent=(np.min(pos[:, 0]), np.max(pos[:, 0]), np.min(pos[:, 1]), np.max(pos[:, 1])),
               origin='lower', cmap='viridis')

    plt.tight_layout()
    plt.show()


def main():
    with open("../ds/MGN/cylinder_dataset/save_0.pkl", 'rb') as f:
        rollout_data = pickle.load(f)  # pickle.load(fp)
    # ['faces', 'mesh_pos', 'gt_velocity', 'pred_velocity']

    # compute bounds
    bounds = []
    for trajectory in rollout_data['velocity']:
        bb_min = trajectory.min(axis=(0, 1))
        bb_max = trajectory.max(axis=(0, 1))
        bounds.append((bb_min, bb_max))

    pos = rollout_data['mesh_pos']
    faces = rollout_data['cells']
    plot_vals = rollout_data['velocity'][0][:, 0]

    # plot_mesh(pos, faces, plot_vals)
    grid_z = plot_mesh_smooth(pos, plot_vals, faces, 512)

    print(grid_z)


if __name__ == '__main__':
    main()
