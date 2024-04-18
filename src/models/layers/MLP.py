import torch.nn as nn


class MLP(nn.Module):
    """ MLP Decoder"""

    def __init__(self, in_dim, out_dim, hid_dim, num_layers, act: str, zero_last=False):
        super().__init__()
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU()
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "gelu":
            self.act = nn.GELU()
        elif act == 'softplus':
            self.act = nn.Softplus()
        elif act == "linear":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Activation {self.act} not supported")

        self.layers = nn.ModuleList()  # Use ModuleList to store layers

        # Create the layers based on the configuration
        if num_layers > 1:
            # Input to first hidden layer
            self.layers.append(nn.Linear(in_dim, hid_dim))
            for _ in range(1, num_layers - 1):
                # Hidden layers
                self.layers.append(nn.Linear(hid_dim, hid_dim))
            # Last hidden to output layer

            if zero_last:
                # Zero initialization for last layer
                self.layers.append(nn.Linear(hid_dim, out_dim))
                self.layers[-1].weight.data.fill_(0)
                self.layers[-1].bias.data.fill_(0)
            else:
                self.layers.append(nn.Linear(hid_dim, out_dim))
        else:
            # Directly connect input to output if only one layer
            self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        # Manually apply layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation up to before the last layer
                x = self.act(x)
        return x
