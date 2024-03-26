from torch import nn
from models.layers.MLP import MLP
import torch


class PatchEmbeddings(nn.Module):
    """Patch embeddings layer used to project patches into feature space."""

    def __init__(self, in_dim, llm_dim, params: dict):
        super().__init__()
        if params['type'] == 'MLP':
            hid_dim, num_layers, act = params["hidden_dim"], params["num_layers"], params["activation"]
            self.encoder = MLP(in_dim=in_dim, out_dim=llm_dim, hid_dim=hid_dim, num_layers=num_layers, act=act)
        else:
            raise ValueError(f"Unknown patch embedding type: {params['type']}")

    def forward(self, x):
        seq_len, num_patch, C, H, W = x.shape
        x = x.view(seq_len, num_patch, -1)

        embeddings = self.encoder(x)
        return embeddings
