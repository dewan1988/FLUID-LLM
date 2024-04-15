import torch
from torch import nn

from models.layers.patch_encoder import PatchEmbeddings
from models.layers.positional_encodings.positional_embeddings import PositionalEmbeddings
from models.layers.positional_encodings.rotary_3d_positional_embeddings import Rotary3DPositionalEmbeddings
from models.layers.self_attention import SelfAttention


class InputEmbeddings(nn.Module):
    """Input embeddings layer adapter for time series data."""

    def __init__(self, patch_dim, llm_dim, enc_params: dict, hidden_dropout_prob,
                 layer_norm_eps, max_pos_embeddings, init_pos_embed,
                 pos_embedding_type="rope", use_self_attn=True,
                 ):
        super().__init__()

        # Patch embedding
        self.patch_embeddings = PatchEmbeddings(in_dim=patch_dim, llm_dim=llm_dim, params=enc_params)

        # Positional Embeddings
        if pos_embedding_type == "rope":
            self.position_embeddings = Rotary3DPositionalEmbeddings(llm_dim)
        elif pos_embedding_type == "pos":
            self.position_embeddings = PositionalEmbeddings(llm_dim, max_pos_embeddings, init_pos_embed)
        else:
            raise ValueError(f"Unknown positional embedding type: {pos_embedding_type}")

        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.patches_attn = SelfAttention(llm_dim)

        # self.LayerNorm = nn.LayerNorm(llm_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x, position_ids):
        """
        x.shape = (seq_len, num_patches, 3, H, W)

        return shape = (seq_len, num_patches, llm_dim)
        """
        # Apply patch embeddings
        inputs_embeds = self.patch_embeddings(x)

        # Apply self attention
        if self.use_self_attn:
            inputs_embeds = self.patches_attn(inputs_embeds)

        embeddings = inputs_embeds

        # Add positional embeddings
        embeddings = self.position_embeddings(embeddings, position_ids)

        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
