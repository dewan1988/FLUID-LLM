import torch
from torch import nn

from models.layers.patch_embeddings import PatchEmbeddings
from models.layers.positional_encodings.positional_embeddings import PositionalEmbeddings
from models.layers.positional_encodings.rotary_3d_positional_embeddings import Rotary3DPositionalEmbeddings
from models.layers.self_attention import SelfAttention


class InputEmbeddings(nn.Module):
    """Input embeddings layer adapter for time series data."""

    def __init__(self, patch_in_dim, hidden_size, hidden_dropout_prob,
                 layer_norm_eps, max_pos_embeddings, pos_embedding_type="rope", use_self_attn=True):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(in_dim=patch_in_dim, llm_dim=hidden_size, hidden_dim=256)

        if pos_embedding_type == "rope":
            self.position_embeddings = Rotary3DPositionalEmbeddings(hidden_size)
        elif pos_embedding_type == "pos":
            self.position_embeddings = PositionalEmbeddings(hidden_size, max_pos_embeddings)
        else:
            raise ValueError(f"Unknown positional embedding type: {pos_embedding_type}")

        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.patches_attn = SelfAttention(hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x, position_ids):
        """
        Here x.shape = (seq_len, num_patches, C, H, W)
        """
        inputs_embeds = self.patch_embeddings(x)

        # Apply self attention
        if self.use_self_attn:
            inputs_embeds = self.patches_attn(inputs_embeds)

        embeddings = inputs_embeds

        # Add positional embeddings
        embeddings = self.position_embeddings(embeddings, position_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
