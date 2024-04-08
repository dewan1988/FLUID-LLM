import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from boundary_domain import BoundaryConditionGenerator
from initial_cond import InitialConditionGenerator, smooth_transition


class WaveConfig:
    nx, ny = 400, 400  # Grid size
    c = 1.0  # Wave speed
    dx = dy = 0.25  # Space step
    dt = 0.1  # Time step
    nt = 1500  # Number of time steps


def init_conditions(cfg: WaveConfig):
    nx, ny, dx, dy = cfg.nx, cfg.ny, cfg.dx, cfg.dy
    nt = cfg.nt

    init_cond_gen = InitialConditionGenerator(nx, ny)
    u = init_cond_gen.random_cond()

    return u


def make_bc_mask(cfg: WaveConfig):
    # Define a non-square boundary (for example, a circular domain)
    nx, ny = cfg.nx, cfg.ny
    bc_gen = BoundaryConditionGenerator(nx, ny)
    domain_mask = bc_gen.random_polygon_boundary()
    plt.imshow(domain_mask, origin='lower')
    plt.show()
    return domain_mask


def plot(cfg: WaveConfig, u_all: np.ndarray, bc_mask: np.ndarray):
    vmin, vmax = np.min(u_all), np.max(u_all)

    # Plot a sequence of timesteps on a single plot
    nx, ny, dx, dy = cfg.nx, cfg.ny, cfg.dx, cfg.dy
    nt = cfg.nt

    selected_timesteps = np.linspace(0, cfg.nt - 1, 8).astype(int)
    n_selected = len(selected_timesteps)

    fig, axs = plt.subplots(2, n_selected // 2, figsize=(15, 6))

    for i, t in enumerate(selected_timesteps):
        ax = axs[i // (n_selected // 2), i % (n_selected // 2)]
        ax.imshow(u_all[t, :, :], origin='lower', extent=[0, nx * dx, 0, ny * dy], vmin=vmin, vmax=vmax)
        ax.set_title(f"Step {t}")
        ax.axis('off')

    # Add a colorbar
    # fig.colorbar(im, ax=axs, orientation='horizontal', fraction=.1)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def main():
    cfg = WaveConfig()
    bc_mask = make_bc_mask(cfg)
    u_init = init_conditions(cfg)
    u_init = smooth_transition(u_init, bc_mask)
    u_init[bc_mask] = 0.

    # Initialize the 3D array to store the wave function at all time steps
    u_all = np.zeros((cfg.nt, cfg.nx, cfg.ny))  # Time, X, Y
    u_all[0, :, :] = u_init
    u_prev = u_init.copy()
    u = u_init.copy()

    # Convolution kernels for second derivatives
    kernel_dx2 = np.array([[1, -2, 1]]) / cfg.dx ** 2
    kernel_dy2 = np.array([[1], [-2], [1]]) / cfg.dy ** 2

    # Time evolution
    for t in range(1, cfg.nt):
        # Convolve with kernels to find second spatial derivatives
        u_xx = convolve2d(u, kernel_dx2, mode='same', boundary='fill', fillvalue=0)
        u_yy = convolve2d(u, kernel_dy2, mode='same', boundary='fill', fillvalue=0)

        # Update u using the discretized wave equation
        u_next = 2 * u - u_prev + cfg.c ** 2 * cfg.dt ** 2 * (u_xx + u_yy)
        # Implement fixed boundary conditions
        u_next[bc_mask] = 0.

        # Store the updated state
        u_all[t, :, :] = u_next

        # Update previous and current time steps
        u_prev = u.copy()
        u = u_next.copy()

    plot(cfg, u_all, bc_mask)


if __name__ == "__main__":
    main()
