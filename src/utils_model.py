import torch
import torch.nn.functional as F
from dataloader.ds_props import DSProps
from dataloader.simple_dataloader import MGNDataset
from torch.utils.data import DataLoader


def get_data_loader(config, mode="train"):
    ds = MGNDataset(load_dir=f'{config["load_dir"]}/{mode}',
                    resolution=config['resolution'],
                    patch_size=config['patch_size'],
                    stride=config['stride'],
                    seq_len=config['seq_len'],
                    seq_interval=config['seq_interval'],
                    mode=mode,
                    normalize=config['normalize_ds']
                    )

    dl = DataLoader(ds,
                    batch_size=config['batch_size'],
                    num_workers=config['num_workers'],
                    prefetch_factor=2,
                    pin_memory=True)

    N_x_patch, N_y_patch = ds.N_x_patch, ds.N_y_patch
    seq_len = ds.seq_len - 1
    ds_props = DSProps(Nx_patch=N_x_patch, Ny_patch=N_y_patch, patch_size=ds.patch_size,
                       seq_len=seq_len)
    return dl, ds_props


def aux_calc_n_rmse(preds: torch.Tensor, target: torch.Tensor, bc_mask: torch.Tensor):
    error = (preds - target) * (~bc_mask)
    # RMSE of each state
    mse = error.pow(2).mean(dim=(-1, -2, -3))
    rmse = torch.sqrt(mse)

    # Average over batches
    N_rmse = rmse  # .mean(dim=0)
    return N_rmse


def calc_n_rmse(preds: torch.Tensor, target: torch.Tensor, bc_mask: torch.Tensor):
    # shape = (bs, seq_len, channel, px, py)
    v_pred = preds[:, :, :2]
    v_target = target[:, :, :2]
    v_mask = bc_mask[:, :, :2]

    p_pred = preds[:, :, 2:]
    p_target = target[:, :, 2:]
    p_mask = bc_mask[:, :, 2:]

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

    patches = patches.reshape(-1, N_patch, channel * px_patch * py_patch).transpose(-1, -2)  # (bs*seq_len, channel*px*py, N_patch)
    img = F.fold(patches, output_size=(tot_px, tot_py), kernel_size=(px_patch, py_patch), stride=(px_patch, py_patch))

    img = img.view(bs, seq_len, channel, tot_px, tot_py)
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


def normalise_diffs(targs, preds, norm_const, channel_indep):
    """ Normalise differences.
        Scale predictions and targets between batch based on true targets

        shape = [bs, seq_len, 3, tot_px, tot_py]
    """
    # print(f'{targs.shape = }, {preds.shape = }')

    if channel_indep:
        targ_std = targs.std(dim=(-1, -2, -4), keepdim=True)  # Std pixels and seq_len
    else:
        targ_std = targs.std(dim=(-1, -2, -3, -4), keepdim=True)    # Std pixels, channels and seq_len
    targs = targs / (targ_std + norm_const)
    preds = preds / (targ_std + norm_const)

    return targs, preds


def normalise_states(diffs, targs, preds, norm_const, channel_indep):
    """ Normalise states.
        Scale predictions and targets between batch based on diffs
        diffs.shape = (bs, seq_len, N_patch, channel, px, py)
        targs.shape = (bs, seq_len, channel, tot_px, tot_py)
    """
    # print(f'{diffs.shape = }, {targs.shape = }, {preds.shape = }')
    if channel_indep:
        diff_std = diffs.std(dim=(-1, -2, -4, -5), keepdim=True).squeeze(1)
    else:
        diff_std = diffs.std(dim=(-1, -2, -3, -4, -5), keepdim=True).squeeze(-1)  # Std pixels, channels and seq_len]

    targs = targs / (diff_std + norm_const)
    preds = preds / (diff_std + norm_const)

    return targs, preds
