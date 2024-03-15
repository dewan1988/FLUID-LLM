from torch import nn
import torch

class PatchEmbeddings(nn.Module):
    """Patch embeddings layer used to project patches into feature space."""

    def __init__(self, in_dim, llm_dim, hidden_dim=256):
        super().__init__()
        self.embed_in = nn.Linear(in_dim, hidden_dim)
        self.embed_out = nn.Linear(hidden_dim, llm_dim)

    def forward(self, x):
        seq_len, num_patch, C, H, W = x.shape
        x = x.view(seq_len, num_patch, -1)
        x = self.embed_in(x)
        z = self.embed_out(x)
        return z
