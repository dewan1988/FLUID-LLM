import torch
import torch.nn as nn
import math


class Rotary3DPositionalEmbeddings(nn.Module):
    def __init__(self, hidden_dim):
        super(Rotary3DPositionalEmbeddings, self).__init__()
        self.hidden_dim = hidden_dim
        self.cache = {}

    def generate_pos_embedding(self, x, position_ids):
        bs, seq_len, N_patch, channel = x.shape

        x = x.view(bs, seq_len*N_patch, channel)
        position_ids = position_ids.view(bs, seq_len*N_patch, 3)

        # All position ids may be 0
        max_vals = position_ids.max(dim=1, keepdim=True)[0]
        safe_max_vals = torch.where(max_vals > 0, max_vals, torch.tensor(1.0, device=max_vals.device))
        # Normalize position_ids to [0, 2*pi]
        position_ids = (position_ids / safe_max_vals) * 2 * math.pi

        if torch.isnan(position_ids).any():
            print(safe_max_vals)
            raise ValueError("Position ids contain NaN values.")

        # Calculate Rotary embeddings for x, y, and z separately
        dim_t = torch.arange(self.hidden_dim // 3).to(x.device)
        dim_t = torch.pow(10000, 2 * dim_t / self.hidden_dim)

        # Apply encoding to each dimension
        pos_embedding = torch.zeros_like(x)
        for i in range(3):  # Iterate over x, y, z
            pos_i = position_ids[:, :, i][:, :, None] / dim_t
            pos_embedding_i = torch.stack((pos_i.sin(), pos_i.cos()), dim=2).flatten(start_dim=2)

            # Assign to the correct third of the hidden dimension
            d_start = i * (self.hidden_dim // 3)
            d_end = (i + 1) * (self.hidden_dim // 3)
            pos_embedding[:, :, d_start:d_end] = pos_embedding_i[:, :, :d_end - d_start]

        pos_embedding = pos_embedding.view(bs, seq_len, N_patch, self.hidden_dim)

        return pos_embedding

    def forward(self, x, position_ids):
        # x shape: [batch_size, seq_len, N_patch, hidden_dim]
        # position_ids shape: [batch_size, seq_len, N_patch, 3] containing (x, y, z) coordinates

        # Create a cache key based on the positions' unique values and their shape
        cache_key = (tuple(position_ids.unique().tolist()), position_ids.shape)
        if cache_key not in self.cache:
            self.cache[cache_key] = self.generate_pos_embedding(x, position_ids)

        pos_embedding = self.cache[cache_key]

        # Apply positional embeddings
        x_pe = x + pos_embedding

        return x_pe
