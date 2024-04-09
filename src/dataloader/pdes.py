import numpy as np
from scipy.signal import convolve2d

class PDEs:
    def __init__(self, dx, dy, bc_mask):
        self.dx, self.dy = dx, dy
        self.bc_mask = bc_mask

        # Convolution kernels for second derivatives
        self.kernel_dx2 = np.array([[1, -2, 1]]) / self.dx ** 2
        self.kernel_dy2 = np.array([[1], [-2], [1]]) / self.dy ** 2
        self.kernel_dxdy = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / self.dx ** 2

    def diffusion(self, u, dudt):
        """
        Returns the Laplacian of the input field.
        """
        d2u_dx2 = convolve2d(u, self.kernel_dxdy, mode='same', boundary='symm')
        dudt = 0.05 * d2u_dx2
        # Boundary Conditions
        dudt[self.bc_mask] = 0

        return dudt, np.zeros_like(dudt)

    def wave_equation(self, u, dudt):
        """
        Returns the time derivative of the input field according to the wave equation.
        """
        dampening = 0.2

        d2u_dx2 = convolve2d(u, self.kernel_dxdy, mode='same', boundary='symm')
        d2udt2 = d2u_dx2 - dampening * dudt

        # Boundary Conditions
        dudt[self.bc_mask] = 0
        d2udt2[self.bc_mask] = 0

        return dudt, d2udt2

