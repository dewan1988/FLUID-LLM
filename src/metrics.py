"""
Module defining metrics
"""
import torch


def aux_calc_n_rmse(preds: torch.Tensor, target: torch.Tensor, bc_mask: torch.Tensor = None):
    if bc_mask:
        error = (preds - target) * (~bc_mask)
    else:
        error = (preds - target)

    mse = error.pow(2).mean(dim=(-1, -2, -3, -4))

    rmse = torch.sqrt(mse)
    N_rmse = rmse.mean()
    return N_rmse


def calc_n_rmse(preds: torch.Tensor, target: torch.Tensor, bc_mask: torch.Tensor = None):
    # shape = (bs, seq_len, N_patch, 1/2, 16, 16)
    v_pred = preds[:, :, :, :2, :]
    v_target = target[:, :, :, :2, :]
    if bc_mask:
        v_mask = bc_mask[:, :, :, :2, :]
    else:
        v_mask = None

    p_pred = preds[:, :, :, 2:, :]
    p_target = target[:, :, :, 2:, :]

    if bc_mask:
        p_mask = bc_mask[:, :, :, 2:, :]
    else:
        p_mask = None

    v_N_rmse = aux_calc_n_rmse(v_pred, v_target, v_mask)
    p_N_rmse = aux_calc_n_rmse(p_pred, p_target, p_mask)

    N_rmse = v_N_rmse + p_N_rmse
    return N_rmse
