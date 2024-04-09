import numpy as np
from scipy.integrate import solve_ivp, ode
import matplotlib.pyplot as plt

from dataloader.initial_cond import InitialConditionGenerator, smooth_transition
from dataloader.boundary_domain import BoundaryConditionGenerator
from dataloader.pdes import PDEs


class WaveConfig:
    Nx, Ny = 240, 60  # Grid size
    Lx, Ly = 2.4, 0.6
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    # dt = 0.1  # Time step
    T = 0.5  # Final time
    Nt = 21  # Number of time steps


class PDESolver2D:
    u0: np.ndarray
    bc_mask: np.ndarray

    def __init__(self, cfg: WaveConfig):
        """
        Initializes the 2D PDE solver.

        Parameters:
        - Lx, Ly: Length of the domain in the x and y dimensions.
        - Nx, Ny: Number of grid points in the x and y dimensions.
        - T: Final time for the simulation.
        """
        self.Lx, self.Ly = cfg.Lx, cfg.Ly
        self.Nx, self.Ny = cfg.Nx, cfg.Ny
        self.Nt = cfg.Nt
        self.dx, self.dy = cfg.dx, cfg.dy
        self.T = cfg.T

        self.x = np.linspace(0, self.Lx, self.Nx)
        self.y = np.linspace(0, self.Ly, self.Ny)
        self.t_eval = np.linspace(0, self.T, self.Nt)  # Time points to evaluate
        self.solution = None

        # Convolution kernels for second derivatives
        self.kernel_dx2 = np.array([[1, -2, 1]]) / self.dx ** 2
        self.kernel_dy2 = np.array([[1], [-2], [1]]) / self.dy ** 2
        self.kernel_dxdy = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / self.dx ** 2

        # self.u0, self.bc_mask = self._initial_condition()

    def set_init_cond(self):
        """
        Sets the initial condition and boundary conditions for the PDE.

        Parameters:
        - func: A function of two variables (x and y) that returns the initial state of the system.
        """
        init_cond_gen = InitialConditionGenerator(self.Nx, self.Ny)
        u0 = init_cond_gen.random_cond()
        dudt0 = np.zeros_like(u0)

        bc_gen = BoundaryConditionGenerator(self.Nx, self.Ny)
        bc_mask = bc_gen.random_boundary()

        u_init = smooth_transition(u0, bc_mask)
        u_init = np.stack((u_init, dudt0), axis=0)
        self.u0, self.bc_mask = u_init, bc_mask
        self.pde = PDEs(self.dx, self.dy, self.bc_mask)

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
        self.solution = solve_ivp(self.pde_wrapper, t_span, self.u0.ravel(), args=(self.pde.wave_equation,), t_eval=self.t_eval,
                                  method='DOP853')  # , max_step=0.1, rtol=0.001, atol=0.01)

        solution = self.solution.y
        solution = solution.reshape((2, self.Nx, self.Ny, self.Nt))

        # us_init = self.u0.ravel().reshape((2, self.Nx, self.Ny))
        # plt.imshow(us_init[0], origin='lower')
        # plt.show()
        #
        # plt.imshow(solution[0, :, :, 0], origin='lower')
        # plt.show()
        # print(self.solution.t)
        # exit(6)
        return solution, self.bc_mask

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
            ax.imshow(u_sol_t, origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(f"Step {t}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    cfg = WaveConfig()
    solver = PDESolver2D(cfg)
    solver.set_init_cond()
    print("Initial cond set")
    solver.solve()
    solver.plot_solution()  # Plot the solution at the final time step
