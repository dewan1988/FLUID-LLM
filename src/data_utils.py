"""
Various utility functions for handling data
"""

import torch


def generate_dummy_ts_dataset(multivariate=False, batch_size=32, horizon=10, n=100, m=5, in_dim=32):
    # Expected dimensions for univariate is BxTxNxin_dim
    # Expected dimensions for multivariate is BxTx(NxM)xin_dim
    # B = batch_size
    # T = horizon (time steps)
    # in_dim = patch dimensionality

    if not multivariate:
        data = torch.rand(batch_size, horizon, n, in_dim, dtype=torch.float)
    else:
        data = torch.rand(batch_size, horizon, n, m, in_dim, dtype=torch.float)

    return data
