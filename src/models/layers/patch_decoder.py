from torch import nn


class PatchDecoder(nn.Module):
    """Patch decoder layer to project from feature space to patch space."""

    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        z = self.decoder(x)
        return z
