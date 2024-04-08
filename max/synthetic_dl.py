import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Grid and wave properties
nx, ny = 400, 400  # Grid size
c = 1.0  # Wave speed
dx = dy = 0.25  # Space step
dt = 0.1  # Time step
nt = 500  # Number of time steps

# Initialize the 3D array to store the wave function at all time steps
u_all = np.zeros((nt, nx, ny))  # Time, X, Y

# Initial condition (a small disturbance in the middle)
u = np.zeros((nx, ny))  # u at current time step
u[nx // 4:3 * nx // 4, ny // 4:3 * ny // 4] = 1.0
# u_prev = np.zeros((nx, ny))  # u at previous time step
u_prev = u.copy()

# Save the initial state
u_all[0, :, :] = u

# Convolution kernels for second derivatives
kernel_dx2 = np.array([[1, -2, 1]]) / dx ** 2
kernel_dy2 = np.array([[1], [-2], [1]]) / dy ** 2

# Time evolution
for t in range(1, nt):
    # Convolve with kernels to find second spatial derivatives
    u_xx = convolve2d(u, kernel_dx2, mode='same', boundary='fill', fillvalue=0)
    u_yy = convolve2d(u, kernel_dy2, mode='same', boundary='fill', fillvalue=0)

    # Update u using the discretized wave equation
    u_next = 2 * u - u_prev + c ** 2 * dt ** 2 * (u_xx + u_yy)
    # Implement fixed boundary conditions
    u_next[0, :] = u_next[-1, :] = u_next[:, 0] = u_next[:, -1] = 0

    # Store the updated state
    u_all[t, :, :] = u_next

    # Update previous and current time steps
    u_prev = u.copy()
    u = u_next.copy()

# Plot a sequence of timesteps on a single plot
selected_timesteps = np.linspace(0, nt - 1, 8).astype(int)
n_selected = len(selected_timesteps)

fig, axs = plt.subplots(2, n_selected // 2, figsize=(15, 6))

for i, t in enumerate(selected_timesteps):
    ax = axs[i // (n_selected // 2), i % (n_selected // 2)]
    im = ax.imshow(u_all[t, :, :], origin='lower', extent=[0, nx * dx, 0, ny * dy], vmin=-1, vmax=1)
    ax.set_title(f"Step {t}")
    ax.axis('off')

# Add a colorbar
# fig.colorbar(im, ax=axs, orientation='horizontal', fraction=.1)

# Adjust layout
plt.tight_layout()
plt.show()
