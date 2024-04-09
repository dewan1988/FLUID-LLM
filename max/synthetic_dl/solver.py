import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from initial_cond import InitialConditionGenerator, smooth_transition
from boundary_domain import BoundaryConditionGenerator
from pdes import PDEs


class PDESolver2D:
    def __init__(self, Lx, Ly, Nx, Ny, T):
        """
        Initializes the 2D PDE solver.

        Parameters:
        - Lx, Ly: Length of the domain in the x and y dimensions.
        - Nx, Ny: Number of grid points in the x and y dimensions.
        - T: Final time for the simulation.
        """
        self.Nt = 100
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        self.T = T
        self.x = np.linspace(0, Lx, Nx)
        self.y = np.linspace(0, Ly, Ny)
        self.t_eval = np.linspace(0, self.T, self.Nt)  # Time points to evaluate

        self.solution = None

        # Convolution kernels for second derivatives
        self.kernel_dx2 = np.array([[1, -2, 1]]) / self.dx ** 2
        self.kernel_dy2 = np.array([[1], [-2], [1]]) / self.dy ** 2
        self.kernel_dxdy = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / self.dx ** 2

        self.u0, self.bc_mask = self._initial_condition()
        self.pde = PDEs(self.dx, self.dy, self.bc_mask)

    def _initial_condition(self):
        """
        Sets the initial condition and boundary conditions for the PDE.

        Parameters:
        - func: A function of two variables (x and y) that returns the initial state of the system.
        """
        init_cond_gen = InitialConditionGenerator(self.Nx, self.Ny)
        u0 = init_cond_gen.multiple_gaussian_pulses()
        dudt0 = np.zeros_like(u0)

        bc_gen = BoundaryConditionGenerator(self.Nx, self.Ny)
        bc_mask = bc_gen.random_polygon_boundary()
        plt.imshow(bc_mask, origin='lower')
        plt.show()

        u_init = smooth_transition(u0, bc_mask)
        u_init = np.stack((u_init, dudt0), axis=0)
        return u_init, bc_mask

    def pde_wrapper(self, t, state_flat, equation_func):
        """
        Returns the RHS of the system of ODEs equivalent to the original PDE.

        Parameters:
        - t: The current time.
        - state_flat: The current state of the system, flattened to 1D.
        - equation_func: A function representing the PDE, taking the current state, spatial steps (dx, dy),
          and any necessary parameters, and returning the time derivative of the state.

        Returns:
        - A flat array representing the time derivative of the system's state.
        """
        state_flat = state_flat.reshape((2, self.Nx * self.Ny))
        u_f, dudt_f = state_flat[0], state_flat[1]

        u = u_f.reshape((self.Nx, self.Ny))  # Reshape to 2D for processing
        dudt = dudt_f.reshape((self.Nx, self.Ny))

        dudt, d2udt2 = equation_func(u, dudt)

        pred_state = np.stack((dudt, d2udt2), axis=0).ravel()  # Stack and flatten
        return pred_state

    def solve(self):
        """
        Solves the PDE using the method of lines.

        Parameters:
        - equation_func: The function representing the PDE to be solved.
        - t_eval: Optional array of time points at which to store the solution.
        """
        t_span = (0, self.T)
        self.solution = solve_ivp(self.pde_wrapper, t_span, self.u0.ravel(), args=(self.pde.wave_equation,), t_eval=self.t_eval, method='RK45')
        return self.solution

    def plot_solution(self):
        """
        Plots the solution at a given time index.

        Parameters:
        - at_time_index: The index of the time step at which to plot the solution.
        """

        solution = self.solution.y
        solution = solution.reshape((2, self.Nx, self.Ny, self.Nt))
        u_sol = solution[0]

        vmin, vmax = np.min(u_sol), np.max(u_sol)

        selected_timesteps = np.linspace(0, self.Nt - 1, 8).astype(int)
        n_selected = len(selected_timesteps)
        fig, axs = plt.subplots(2, n_selected // 2, figsize=(15, 6))

        for i, t in enumerate(selected_timesteps):
            ax = axs[i // (n_selected // 2), i % (n_selected // 2)]
            u_sol_t = u_sol[:, :, t]
            ax.imshow(u_sol_t, origin='lower', extent=[0, self.Nx * self.dx, 0, self.Ny * self.dy], vmin=vmin, vmax=vmax)
            ax.set_title(f"Step {t}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    solver = PDESolver2D(Lx=3.0, Ly=1.0, Nx=150, Ny=100, T=3)
    solver.solve()
    solver.plot_solution()  # Plot the solution at the final time step
