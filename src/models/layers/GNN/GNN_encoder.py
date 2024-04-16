import torch
from torch import nn

from models.layers.GNN.GATConv import GATNet
from models.layers.positional_encodings.positional_embeddings import PositionalEmbeddings
from models.layers.positional_encodings.rotary_3d_positional_embeddings import Rotary3DPositionalEmbeddings
from models.layers.self_attention import SelfAttention


class GNNEncoder(nn.Module):
    """Input embeddings layer adapter for time series data."""

    def __init__(self, vertex_dim, edge_dim, llm_dim, enc_params: dict, hidden_dropout_prob,
                 layer_norm_eps, max_pos_embeddings, init_pos_embed,
                 pos_embedding_type="rope"
                 ):
        super().__init__()

        # Patch embedding
        self.GNN_encoder = GATNet(vertex_dim, edge_dim, out_dim=llm_dim, cfg=enc_params)

        # Positional Embeddings
        if pos_embedding_type == "rope":
            self.position_embeddings = Rotary3DPositionalEmbeddings(llm_dim)
        elif pos_embedding_type == "pos":
            self.position_embeddings = PositionalEmbeddings(llm_dim, max_pos_embeddings, init_pos_embed)
        else:
            raise ValueError(f"Unknown positional embedding type: {pos_embedding_type}")

        self.LayerNorm = nn.LayerNorm(llm_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, data):
        """
        data: BS*seq_len graphs

        return shape = (seq_len, num_patches, llm_dim)
        """
        # Apply GNN encoding
        inputs_embeds = self.GNN_encoder.forward(data)

        # # Add positional embeddings
        # embeddings = self.position_embeddings(embeddings, position_ids)
        #
        # embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return inputs_embeds
