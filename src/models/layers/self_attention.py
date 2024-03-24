import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        # Assuming the same size for queries, keys, and values for simplicity
        self.query_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.key_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.value_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, x):
        # x shape: [seq_len, num_patches, hidden_dim]
        seq_len, num_embedding, hidden_dim = x.shape

        # Prepare queries, keys, values
        Q = torch.matmul(x, self.query_weight)
        K = torch.matmul(x, self.key_weight)
        V = torch.matmul(x, self.value_weight)

        # Compute attention scores, shape: [seq_len, num_patches, num_embedding]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Apply attention scores to values
        attention_output = torch.matmul(attention_scores, V)

        return attention_output
