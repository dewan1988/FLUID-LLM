from torch import nn


class PatchEmbeddings(nn.Module):
    """Patch embeddings layer used to project patches into feature space."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.embeddings = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        seq_len, num_patch, C, H, W = x.shape
        x = x.view(seq_len, num_patch, -1)
        z = self.embeddings(x)
        return z
