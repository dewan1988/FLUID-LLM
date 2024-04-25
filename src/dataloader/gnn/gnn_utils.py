import torch
from cprint import c_print
from matplotlib import tri as mtri
import matplotlib.pyplot as plt


def faces_to_edges(faces):
    """ Given a triangular mesh, return all edges that make up mesh"""
    # faces.shape = [N_cells, 3]
    edges = torch.cat([faces[:, :2], faces[:, 1:], faces[:, ::2]], dim=0)

    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)

    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    unique_edges = torch.unique(packed_edges, dim=0)
    unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=0)

    return unique_edges


def plot_mesh(pos, faces, val: torch.Tensor):
    """Plots triangular mesh from positions of nodes and triangles defined using faces."""
    pos = pos.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    val = val.detach().cpu().numpy()

    min, max, std = val.min(), val.max(), val.std()
    print(f'{min = :.3g}, {max = :.3g}, {std = :.3g}')
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)

    plt.figure(figsize=(10, 4))
    plt.tripcolor(triang, val)  # or contourf
    plt.triplot(triang, "k-", lw=0.3)
    plt.ylim([0., 0.5])
    plt.axis("equal")

    plt.axis("off")
    plt.tight_layout()

    plt.show()


def unflatten_states(Xs: torch.Tensor, split_idx, seq_len: int) -> tuple[torch.Tensor]:
    """ Split a concatenated batch of graph nodes back into a list of graphs. """
    _, dim = Xs.shape
    split_batch_idx = split_idx * seq_len
    Xs = torch.split(Xs, split_batch_idx.tolist())
    Xs = [Xs[i].view(seq_len, -1, dim) for i in range(len(Xs))]
    return Xs