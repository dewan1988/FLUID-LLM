from matplotlib import tri as mtri
import matplotlib.pyplot as plt
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
    grid_z = to_grid(pos, val, faces, grid_res)
    plt.figure(figsize=(12, 4))
    plt.imshow(grid_z.T, extent=(np.min(pos[:, 0]), np.max(pos[:, 0]), np.min(pos[:, 1]), np.max(pos[:, 1])),
               origin='lower', cmap='viridis')

    plt.tight_layout()
    plt.show()


def to_grid(pos, val, faces, grid_res, mask_val=-0.1):
    # Define your grid bounds and resolution
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)

    x_min, x_max = np.min(pos[:, 0]), np.max(pos[:, 0])
    y_min, y_max = np.min(pos[:, 1]), np.max(pos[:, 1])

    x_points = grid_res * 3
    y_points = grid_res
    grid_x, grid_y = np.mgrid[x_min:x_max:x_points * 1j, y_min:y_max:y_points * 1j]  # Adjust 100j as needed for resolution
    interp = mtri.CubicTriInterpolator(triang, val, kind='geom')
    # interp = mtri.LinearTriInterpolator(triang, val)

    mask_data = interp(grid_x, grid_y)
    mask = mask_data.mask
    data = mask_data.data
    data[mask] = mask_val
    return data
