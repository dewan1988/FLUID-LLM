from mesh_utils import plot_mesh
import numpy as np
import os


def get_data(path):
    # Time sampling is random during training, but set to a fix value during test and valid, to ensure repeatability.
    data = np.load(os.path.join(path, 'sim.npz'), mmap_mode='r')

    mesh_pos = data["pointcloud"]

    cells = np.load(os.path.join(path, f"triangles.npy"))
    cells = cells

    Vx = data['VX']
    Vy = data['VY']

    Ps = data['PS']
    Pg = data['PG']

    velocity = np.stack([Vx, Vy], axis=-1)
    pressure = np.stack([Ps, Pg], axis=-1)
    node_type = data['mask']

    return mesh_pos, cells, node_type, velocity, pressure


def plot():
    mesh_pos, faces, node_type, velocity, pressure = get_data("../ds/Eagle/Step/1/1")
    for step in [14, 15, 16]:
        plot_mesh(mesh_pos[step], faces[step], pressure[step, :, 0])


if __name__ == "__main__":
    plot()
