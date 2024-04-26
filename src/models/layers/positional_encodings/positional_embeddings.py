import torch.nn as nn
import math
import random


class PositionalEmbeddings(nn.Module):
    """Positional embeddings layer, for time, x and y."""

    def __init__(self, hidden_size, max_embeds, init_pos_embed):
        super().__init__()

        max_x, max_y, max_t = max_embeds

        self.x_embeddings = nn.Embedding(max_x, hidden_size)
        self.y_embeddings = nn.Embedding(max_y, hidden_size)
        self.time_embeddings = nn.Embedding(max_t, hidden_size)

        if init_pos_embed == "zero":
            # Zero init embeddings
            nn.init.zeros_(self.x_embeddings.weight)
            nn.init.zeros_(self.y_embeddings.weight)
            nn.init.zeros_(self.time_embeddings.weight)
        elif init_pos_embed == "scaled":
            # Scaled init embeddings to norm 1
            std = 1 / math.sqrt(hidden_size)
            nn.init.normal_(self.x_embeddings.weight, mean=0, std=std)
            nn.init.normal_(self.y_embeddings.weight, mean=0, std=std)
            nn.init.normal_(self.time_embeddings.weight, mean=0, std=std)
        elif init_pos_embed == "normal":
            pass

    def forward(self, x, position_ids):
        x_embeddings = self.x_embeddings(position_ids[..., 0])
        y_embeddings = self.y_embeddings(position_ids[..., 1])
        time_embeddings = self.time_embeddings(position_ids[..., 2])

        return x + (x_embeddings + y_embeddings + time_embeddings)
