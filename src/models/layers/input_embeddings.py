import torch
from torch import nn

from models.layers.patch_encoder import PatchEmbeddings
from models.layers.positional_encodings.positional_embeddings import PositionalEmbeddings
from models.layers.positional_encodings.rotary_3d_positional_embeddings import Rotary3DPositionalEmbeddings
from models.layers.self_attention import SelfAttention


class InputEmbeddings(nn.Module):
    """Input embeddings layer adapter for time series data."""

    def __init__(self, patch_dim, llm_dim, enc_params: dict, emb_cfg: dict):
        super().__init__()

        # Patch embedding
        self.patch_embeddings = PatchEmbeddings(in_dim=patch_dim, llm_dim=llm_dim, params=enc_params)

        # Positional Embeddings
        if emb_cfg['pos_embedding_type'] == "rope":
            self.position_embeddings = Rotary3DPositionalEmbeddings(llm_dim)
        elif emb_cfg['pos_embedding_type'] == "pos":
            self.position_embeddings = PositionalEmbeddings(llm_dim, emb_cfg['max_num_embed'], emb_cfg['init_pos_embed'])
        else:
            raise ValueError(f"Unknown positional embedding type: {emb_cfg['pos_embedding_type']}")

        if enc_params['in_emb_ln_eps'] is not None:
            self.LayerNorm = nn.LayerNorm(llm_dim, eps=enc_params['in_emb_ln_eps'])
        else:
            self.LayerNorm = nn.Identity()

        if enc_params['input_emb_layer_dropout'] is not None:
            self.dropout = nn.Dropout(emb_cfg['input_emb_layer_dropout'])
        else:
            self.dropout = nn.Identity()

    def forward(self, x, position_ids):
        """
        x.shape = (seq_len, num_patches, 3, H, W)

        return shape = (seq_len, num_patches, llm_dim)
        """
        # Apply patch embeddings
        inputs_embeds = self.patch_embeddings(x)

        embeddings = inputs_embeds

        # Add positional embeddings
        embeddings = self.position_embeddings(embeddings, position_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
