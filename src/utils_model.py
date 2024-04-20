"""
Module defining metrics
"""
import torch
import torch.nn.functional as F
from dataloader.ds_props import DSProps


def aux_calc_n_rmse(preds: torch.Tensor, target: torch.Tensor, bc_mask: torch.Tensor = None):
    error = (preds - target) * (~bc_mask)
    # RMSE of each state
    mse = error.pow(2).mean(dim=(-1, -2, -3))
    rmse = torch.sqrt(mse)

    # Average over seq_len and batches
    N_rmse = rmse.mean()
    return N_rmse


def calc_n_rmse(preds: torch.Tensor, target: torch.Tensor, bc_mask: torch.Tensor = None):
    # shape = (bs, seq_len, channel, px, py)
    v_pred = preds[:, :, :2, :]
    v_target = target[:, :, :2, :]
    v_mask = bc_mask[:, :, :2, :]

    p_pred = preds[:, :, 2:, :]
    p_target = target[:, :, 2:, :]
    p_mask = bc_mask[:, :, 2:, :]

    v_N_rmse = aux_calc_n_rmse(v_pred, v_target, v_mask)
    p_N_rmse = aux_calc_n_rmse(p_pred, p_target, p_mask)

    N_rmse = v_N_rmse + p_N_rmse
    return N_rmse


def patch_to_img(patches, ds_props: DSProps):
    """ Convert patches to image
        return.shape = (bs, seq_len, 3, tot_px/S, tot_py/S)
    """
    bs, seq_len, N_patch, channel, px, py = patches.shape

    px_patch, py_patch = ds_props.patch_size
    channel = ds_props.channel
    tot_px, tot_py = ds_props.input_tot_size
    N_patch = ds_props.N_patch

    patches = patches.view(-1, N_patch, channel * px_patch * py_patch).transpose(-1, -2)  # (bs*seq_len, channel*px*py, N_patch)
    img = F.fold(patches, output_size=(tot_px, tot_py), kernel_size=(px_patch, py_patch), stride=(px_patch, py_patch))

    img = img.view(bs, seq_len, channel, tot_px, tot_py)
    # img = img[:, :, :, ::2, ::2]
    return img


def img_to_patch(img, ds_props: DSProps):
    """ Convert image to patches
        return.shape = (bs, seq_len, N_patch, channel, px, py)
    """

    bs, seq_len, channel, tot_px, tot_py = img.shape

    px_patch, py_patch = ds_props.patch_size
    channel = ds_props.channel
    N_patch = ds_props.N_patch

    img = img.view(-1, channel, tot_px, tot_py)  # (bs*seq_len, channel, tot_px, tot_py)
    patches = F.unfold(img, kernel_size=(px_patch, py_patch), stride=(px_patch, py_patch))
    patches = patches.view(bs, seq_len, channel, px_patch, py_patch, N_patch).permute(0, 1, 5, 2, 3, 4)
    return patches
