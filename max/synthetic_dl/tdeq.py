import torch
from torchdiffeq import odeint
from matplotlib import pyplot as plt


class ODEFunc():
    def __init__(self):
        super(ODEFunc, self).__init__()

    def forward(self, t, y):
        # This method computes f'(t, y)
        return -y


# Example initial condition
y0 = torch.tensor([2., 0.]).float()

# Time points where we want the solution
t = torch.linspace(0., 25., 100)

# Create an instance of your ODE function
ode_func = ODEFunc()

# Solve the ODE
solution = odeint(ode_func.forward, y0, t, method='rk4')

plt.plot(solution[:, 0])
plt.show()
