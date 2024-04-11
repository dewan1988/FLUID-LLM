import torch.nn as nn


class CNN(nn.Module):
    """ CNN"""

    def __init__(self, in_dim, out_dim, hid_dim, num_layers, act: str, zero_last=False, pool_output=True, conv_type='2d'):
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
        elif act == "linear":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Activation {self.act} not supported")

        self.conv_layer = nn.Conv2d if conv_type == '2d' else nn.Conv1d
        self.pool_output = pool_output
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.layers = nn.ModuleList()  # Use ModuleList to store layers

        # Create the layers based on the configuration
        if num_layers > 1:
            # Input to first hidden layer
            self.layers.append(self.conv_layer(in_channels=in_dim, out_channels=hid_dim, kernel_size=3, padding=1))

            for _ in range(1, num_layers - 1):
                # Hidden layers
                self.layers.append(self.conv_layer(in_channels=hid_dim, out_channels=hid_dim, kernel_size=3, padding=1))

            # Last hidden to output layer
            if zero_last:
                # Zero initialization for last layer
                self.layers.append(self.conv_layer(in_channels=hid_dim, out_channels=out_dim, kernel_size=3, padding=1))
                self.layers[-1].weight.data.fill_(0)
                self.layers[-1].bias.data.fill_(0)
            else:
                self.layers.append(self.conv_layer(in_channels=hid_dim, out_channels=out_dim, kernel_size=3, padding=1))
        else:
            # Directly connect input to output if only one layer
            self.layers.append(self.conv_layer(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1))

    def forward(self, x):
        # Manually apply layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation up to before the last layer
                x = self.act(x)

        if self.pool_output:
            x = self.pool(x)

        return x
