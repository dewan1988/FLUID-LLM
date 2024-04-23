"""
Testing loading a saved model and running the test_generate function
"""
import os
import logging
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from cprint import c_print

from utils import set_seed, load_yaml_from_file, get_available_device, get_save_folder, get_accelerator
from utils_model import calc_n_rmse, patch_to_img, get_data_loader
from models.model import MultivariateTimeLLM
import torch.nn.functional as F

from dataloader.simple_dataloader import MGNDataset

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def get_eval_dl(model, bs, seq_len):
    ds = MGNDataset(load_dir=f"./ds/MGN/cylinder_dataset/valid",
                    resolution=model.config['resolution'],
                    patch_size=model.config['patch_size'],
                    stride=model.config['stride'],
                    seq_len=seq_len,
                    seq_interval=model.config['seq_interval'],
                    mode='valid',
                    fit_diffs=model.config['fit_diffs'],
                    normalize=model.config['normalize_ds'])

    dl = DataLoader(ds, batch_size=bs, pin_memory=True)
    return dl


def test_step(model: MultivariateTimeLLM, dl, plot_step, batch_num=0):

    model.eval()
    # Get batch and run through model
    batch_data = next(iter(dl))

    with torch.inference_mode():
        states, target, bc_mask, position_ids = batch_data
        bs = target.shape[0]
        decoder_out = model.forward(states.cuda(), position_ids.cuda())

        # Reshape targets to images and downsample
        target = target.view(bs, -1, 60, 3, 16, 16)
        target = target.view(-1, 60, 3 * 16 * 16).transpose(-1, -2)

        targ_img = F.fold(target, output_size=(240, 64), kernel_size=(16, 16), stride=(16, 16))
        targ_img = targ_img.view(bs, -1, 3, 240, 64)

    fig, axs = plt.subplots(3, 2, figsize=(20, 8))
    for i, ax in enumerate(axs):
        plot_dim = i

        plot_targ = targ_img[batch_num, plot_step, plot_dim]
        plot_preds = decoder_out[batch_num, plot_step, plot_dim]

        targ_min, targ_max = plot_targ.min(), plot_targ.max()
        pred_min, pred_max = plot_preds.min(), plot_preds.max()

        print()
        print(f'Target min: {targ_min:.4g}, max: {targ_max:.4g}, std: {plot_targ.std():.4g}')
        print(f'Pred min: {pred_min:.4g}, max: {pred_max:.4g}, std: {plot_preds.std():.4g}')

        ax[0].imshow(plot_targ.cpu().T)
        ax[1].imshow(plot_preds.cpu().T)

    fig.tight_layout()
    plt.show()

    # Calculate normalised RMSE
    decoder_out = decoder_out.cpu()
    targ_std = targ_img.std(dim=(-1, -2, -3, -4), keepdim=True)  # Std over each batch item
    targ_img_red = targ_img / (targ_std)
    decoder_out = decoder_out / (targ_std)

    loss_fn = torch.nn.L1Loss()
    loss = loss_fn(decoder_out, targ_img_red)
    print(targ_std.squeeze())
    print(loss)


def test_generate(model: MultivariateTimeLLM, dl, plot_step, batch_num=0):
    model.eval()
    # Get batch and run through model
    batch = next(iter(dl))

    with torch.inference_mode():
        states, target, bc_mask, position_ids = batch
        states, target, bc_mask, position_ids = (states.cuda(), target.cuda(), bc_mask.cuda(), position_ids.cuda())
        batch = (states, target, bc_mask, position_ids)

        bs, seq_len, N_patch, channel, px, py = states.shape
        pred_states, pred_diffs = model.gen_seq(batch, pred_steps=seq_len - 1)

        true_states = patch_to_img(states, model.ds_props)
        true_diffs = patch_to_img(target, model.ds_props)
        bc_mask = patch_to_img(bc_mask.float(), model.ds_props).bool()
        pred_states = pred_states[:, :-1]
        print(f'{pred_states.shape = }, {pred_diffs.shape = }')
        print(f'{true_states.shape = }, {true_diffs.shape = }')

    # Plot diffs
    fig, axs = plt.subplots(3, 2, figsize=(20, 9))
    fig.suptitle(f'Differences, step {plot_step}')
    for i, ax in enumerate(axs):
        img_1 = true_diffs[batch_num, plot_step, i].cpu()
        img_2 = pred_diffs[batch_num, plot_step, i].cpu()

        ax[0].imshow(img_1.T)  # Initial image
        ax[1].imshow(img_2.T)  # Predictions
        ax[0].axis('off'), ax[1].axis('off')
    fig.tight_layout()
    fig.show()

    # Plot states
    fig, axs = plt.subplots(3, 2, figsize=(20, 9))
    fig.suptitle(f'States, step {plot_step}')
    for i, ax in enumerate(axs):
        img_1 = true_states[batch_num, plot_step, i].cpu()
        img_2 = pred_states[batch_num, plot_step, i].cpu()

        ax[0].imshow(img_1.T)  # Initial image
        ax[1].imshow(img_2.T)  # Predictions
        ax[0].axis('off'), ax[1].axis('off')
    fig.tight_layout()
    fig.show()

    N_rmse = calc_n_rmse(pred_states, true_states, bc_mask)
    c_print(f"Standard N_RMSE: {N_rmse}, Mean: {N_rmse.mean().item():.3g}", color='cyan')

    targ_std = true_diffs.std(dim=(-1, -2, 0, -4), keepdim=True)  # Std pixels, channels and seq_len
    # true_diffs = true_diffs / (targ_std + 0.0005)
    # pred_diffs = pred_diffs / (targ_std + 0.0005)

    print(targ_std.squeeze())


@torch.inference_mode()
def get_ds_stats(model: MultivariateTimeLLM, dl):
    all_states, all_targets = [], []
    for batch in dl:
        states, target, _, _ = batch
        all_states.append(states)
        all_targets.append(target)

    all_states = torch.cat(all_states, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    print(all_targets.shape)
    diff_stds = all_targets.std(dim=(0, 1, 2, 4, 5))
    diff_mean = all_targets.mean(dim=(0, 1, 2, 4, 5))
    print(f'STD over channels {diff_stds = }')

    all_stds = all_targets.std()
    print(f'STD over entire dataset: {all_stds = }')


def main():
    load_no = -1
    save_epoch = 160
    seq_len = 27
    bs = 100

    plot_step = 8
    batch_num = 0

    set_seed()

    # Load the checkpoint
    load_path = get_save_folder("./model_checkpoints", load_no=load_no)
    checkpoint_file_path = os.path.join(load_path, f'step_{save_epoch}.pth')
    logging.info(f"Loading checkpoint from: {checkpoint_file_path}")

    if not os.path.exists(checkpoint_file_path):
        raise ValueError(f"Checkpoint file not found at {checkpoint_file_path}")

    ckpt = torch.load(checkpoint_file_path)
    ckpt_params = ckpt['params']
    ckpt_state_dict = ckpt['state_dict']

    # Get dataloader
    _, ds_props = get_data_loader(ckpt_params, mode="valid")

    # Get the model
    model = MultivariateTimeLLM(ckpt_params, ds_props=ds_props, device_map=get_available_device())
    # Load weights
    model.load_state_dict(ckpt_state_dict)
    accelerator = get_accelerator(use_deepspeed=False, precision='bf16')
    model = accelerator.prepare(model)

    # Val dataloader
    dl = get_eval_dl(model, bs, seq_len)

    # Run test_generate
    test_generate(model, dl, plot_step, batch_num)
    # test_step(model, dl, plot_step, batch_num)
    # get_ds_stats(model, dl)


if __name__ == '__main__':
    main()
