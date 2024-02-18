from matplotlib import tri as mtri
import matplotlib.pyplot as plt
import numpy as np


def plot_mesh(pos, faces, val):
    """Plots triangular mesh from positions of nodes and triangles defined using faces."""

    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    plt.tripcolor(triang, val)      # or contourf
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

