import torch
from torch import nn

from models.layers.patch_embeddings import PatchEmbeddings


class InputEmbeddings(nn.Module):
    """Input embeddings layer adapter for time series data."""

    def __init__(self, patch_in_dim, hidden_size, hidden_dropout_prob, layer_norm_eps, max_pos_embeddings):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(patch_in_dim, hidden_size)
        self.position_embeddings = nn.Embedding(max_pos_embeddings, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(max_pos_embeddings).expand((1, -1)),
                             persistent=False)

    def forward(
        self,
        x,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length: int = 0,
    ):
        """
        Here x.shape = (seq_len, num_patches, C, H, W)
        """
        if x is not None:
            input_shape = x.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        inputs_embeds = self.patch_embeddings(x)

        embeddings = inputs_embeds

        # Add positional embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
