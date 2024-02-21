"""Plots a sample from MeshGraphNets."""

import pickle

from mesh_utils import plot_mesh, plot_mesh_smooth


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
