import torch
from torch import nn

from models.layers.patch_embeddings import PatchEmbeddings
from models.layers.positional_embeddings import PositionalEmbeddings


class InputEmbeddings(nn.Module):
    """Input embeddings layer adapter for time series data."""

    def __init__(self, patch_in_dim, hidden_size, hidden_dropout_prob, layer_norm_eps, max_pos_embeddings):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(patch_in_dim, hidden_size)
        self.position_embeddings = PositionalEmbeddings(hidden_size, max_pos_embeddings)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
            self,
            x,
            position_ids,
    ):
        """
        Here x.shape = (seq_len, num_patches, C, H, W)
        """

        inputs_embeds = self.patch_embeddings(x)

        embeddings = inputs_embeds

        # Add positional embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
