import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torch

from dataloader.synthetic.initial_cond import InitialConditionGenerator, smooth_transition
from dataloader.synthetic.boundary_domain import BoundaryConditionGenerator
from dataloader.synthetic.pdes import PDEs


class WaveConfig:
    Nx, Ny = 240, 60  # Grid size
    Lx, Ly = 2.4, 0.6  # Domain size
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    T = 0.2  # Final time
    Nt = 21  # Number of time steps
    stepsize = 0.005


class PDESolver2D:
    u0: torch.Tensor
    bc_mask: torch.Tensor
    solution: torch.Tensor

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
        self.stepsize = cfg.stepsize

        self.t_eval = torch.linspace(0, self.T, self.Nt)  # Time points to evaluate

    def set_init_cond(self):
        """
        Sets the initial condition and boundary conditions for the PDE.

        Parameters:
        - func: A function of two variables (x and y) that returns the initial state of the system.
        """
        init_cond_gen = InitialConditionGenerator(self.Nx, self.Ny)
        u_init = init_cond_gen.random_cond()

        bc_gen = BoundaryConditionGenerator(self.Nx, self.Ny)
        bc_mask = bc_gen.random_boundary()

        u_init = smooth_transition(u_init, bc_mask, smooth=True)
        dudt0 = torch.zeros_like(u_init)
        u_init = torch.stack((u_init, dudt0), dim=0)

        bc_mask = torch.from_numpy(bc_mask).bool()
        self.u0, self.bc_mask = u_init, bc_mask
        self.pde = PDEs(self.dx, self.dy, self.bc_mask)

    def pde_wrapper(self, t, state_flat):
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
        u, dudt = state_flat[0], state_flat[1]

        dudt, d2udt2 = self.pde.wave_equation(u, dudt)
        pred_state = torch.stack((dudt, d2udt2), dim=0)  # Stack and flatten
        return pred_state

    def solve(self):
        """
        Solves the PDE using the method of lines.

        Parameters:
        - equation_func: The function representing the PDE to be solved.
        - t_eval: Optional array of time points at which to store the solution.

        solution.shape = (Nt, 2, Nx, Ny)
        """
        with torch.no_grad():
            self.solution = odeint(self.pde_wrapper, self.u0, self.t_eval,
                                   method='rk4', options={'step_size': self.stepsize})  #
        return self.solution, self.bc_mask

    def plot_solution(self):
        """
        Plots the solution at a given time index.

        Parameters:
        - at_time_index: The index of the time step at which to plot the solution.
        """

        solution = self.solution
        u_sol = solution[:, 0]

        vmin, vmax = torch.min(u_sol), torch.max(u_sol)

        selected_timesteps = np.linspace(0, self.Nt - 1, 8).astype(int)
        n_selected = len(selected_timesteps)
        fig, axs = plt.subplots(2, n_selected // 2, figsize=(15, 6))

        for i, t in enumerate(selected_timesteps):
            ax = axs[i // (n_selected // 2), i % (n_selected // 2)]
            u_sol_t = u_sol[t].T
            ax.imshow(u_sol_t, origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(f"Step {t}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    import time
    from utils import set_seed

    cfg = WaveConfig()
    solver = PDESolver2D(cfg)
    solver.set_init_cond()
    print("Initial cond set")
    st = time.time()
    solver.solve()
    print(f'{time.time() - st:.2f} s')
    solver.plot_solution()  # Plot the solution at the final time step
