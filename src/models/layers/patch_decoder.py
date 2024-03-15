from torch import nn


class PatchDecoder(nn.Module):
    """Patch decoder layer to project from feature space to patch space."""

    def __init__(self, llm_dim, out_dim, hid_dim=256):
        super().__init__()

        self.dec_in = nn.Linear(llm_dim, hid_dim)
        self.dec_out = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.dec_in(x)
        z = self.dec_out(x)
        return z
