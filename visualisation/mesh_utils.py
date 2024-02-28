from matplotlib import tri as mtri
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

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

def inpaint(img, mask, method='telea'):
    """Inpaints mask to avoid rough edges. """

    min_val = np.min(img)
    max_val = np.max(img)

    img = (img - min_val) / (max_val - min_val)
    img = (img * 255).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    if method == 'telea':
        img =  cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    elif method == 'ns':
        img =  cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    else:
        raise ValueError(f"Unknown inpainting method: {method}")

    img = (img / 255) * (max_val - min_val) + min_val
    return img.astype(np.float32)


def to_grid(pos, val, faces, grid_res, mask_interp=None, type='linear'):
    # Define your grid bounds and resolution
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)

    x_min, x_max = np.min(pos[:, 0]), np.max(pos[:, 0])
    y_min, y_max = np.min(pos[:, 1]), np.max(pos[:, 1])

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

    if type == 'cubic':
        interp = mtri.CubicTriInterpolator(triang, val, kind='geom')
    elif type == 'linear':
        interp = mtri.LinearTriInterpolator(triang, val)
    else:
        raise ValueError(f"Unknown interpolation algorithm: {type}")

    mask_data = interp(grid_x, grid_y)
    mask = mask_data.mask
    data = mask_data.data

    data[mask] = 0.

    if mask_interp is not None:
        inpaint_data = inpaint(data, mask, method='telea')
        data[mask] = inpaint_data[mask]
    #data = inpaint_data
    # data[mask] = mask_val
    return data, mask
