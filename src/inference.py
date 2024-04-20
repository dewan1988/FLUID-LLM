"""
Testing loading a saved model and running the test_generate function
"""
import os
import sys
import argparse
import logging
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

from tqdm import tqdm

from utils import set_seed, load_yaml_from_file, get_available_device, get_save_folder, get_accelerator
from utils_model import calc_n_rmse
from models.model import MultivariateTimeLLM
import torch.nn.functional as F
from trainer import get_data_loader

from dataloader.simple_dataloader import MGNDataset
from dataloader.mesh_utils import plot_full_patches

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def rmse_loss(preds, targets):
    """ state.shape = (bs, num_steps, N_patch, 3, 16, 16)"""
    assert preds.shape == preds.shape
    preds = preds.to(torch.float32)
    targets = targets.to(torch.float32)

    v_pred = preds[:, :, :, :2, :]
    v_target = targets[:, :, :, :2, :]

    p_pred = preds[:, :, :, 2:, :]
    p_target = targets[:, :, :, 2:, :]

    # MSE over all patches in each step
    v_mse = torch.mean((v_pred - v_target) ** 2, dim=(-1, -2, -3, -4))
    p_mse = torch.mean((p_pred - p_target) ** 2, dim=(-1, -2, -3, -4))

    # Average RMSE over batch
    v_rmse = torch.sqrt(v_mse).mean(dim=0)
    p_rmse = torch.sqrt(p_mse).mean(dim=0)
    # print(v_rmse)
    # exit(9)
    rmse = v_rmse + p_rmse
    return rmse


def test_generate(model: MultivariateTimeLLM, eval_cfg, plot_step, batch_num=0):
    bs = eval_cfg['batch_size']
    ds = MGNDataset(load_dir=eval_cfg['load_dir'],
                    resolution=model.config['resolution'],
                    patch_size=model.config['patch_size'],
                    stride=model.config['stride'],
                    seq_len=eval_cfg['seq_len'],
                    seq_interval=model.config['seq_interval'],
                    mode='valid',
                    fit_diffs=model.config['fit_diffs'],
                    normalize=model.config['normalize_ds'])
    N_patch = ds.N_patch

    dl = DataLoader(ds, batch_size=bs, pin_memory=True)
    N_x_patch, N_y_patch = ds.N_x_patch, ds.N_y_patch
    x_px, y_px = model.config['patch_size']

    model.eval()
    # Get batch and run through model
    batch_data = next(iter(dl))

    with torch.inference_mode():
        states, target, bc_mask, position_ids = batch_data
        decoder_out = model.forward(states.cuda(), position_ids.cuda())

        # Reshape targets to images and downsample
        target = target.view(bs, -1, 60, 3, 16, 16)
        target = target.view(-1, 60, 3 * 16 * 16).transpose(-1, -2)

        targ_img = F.fold(target, output_size=(240, 64), kernel_size=(16, 16), stride=(16, 16))
        targ_img = targ_img.view(bs, -1, 3, 240, 64)
        # targ_img_red = targ_img[:, :, :, ::2, ::2]

    fig, axs = plt.subplots(3, 2, figsize=(20, 8))
    for i, ax in enumerate(axs):
        plot_dim = i

        plot_targ = targ_img[batch_num, plot_step, plot_dim]
        plot_preds = decoder_out[batch_num, plot_step, plot_dim]

        targ_min, targ_max = plot_targ.min(), plot_targ.max()
        pred_min, pred_max = plot_preds.min(), plot_preds.max()

        print(f'Target min: {targ_min:.4g}, max: {targ_max:.4g}, std: {plot_targ.std():.4g}')
        print(f'Pred min: {pred_min:.4g}, max: {pred_max:.4g}')

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
    # print(f"Time taken: {time.time() - st:.4g}")
    #
    # # Split into steps
    # true_states = true_states.view(bs, eval_cfg['seq_len'] - 1, N_patch, 3,x_px, y_px).to(torch.float32)
    # pred_states = pred_states.view(bs, eval_cfg['seq_len'], N_patch, 3, x_px, y_px).cpu()
    # pred_states = pred_states[:, :-1]
    #
    # true_diffs = true_diffs.view(bs, eval_cfg['seq_len'] - 1, N_patch, 3, x_px, y_px)
    # pred_diffs = pred_diffs.view(bs, eval_cfg['seq_len'] - 1, N_patch, 3, x_px, y_px).cpu()
    #
    # loss = rmse_loss(pred_states, true_states)
    # N_rmse = calc_n_rmse(pred_states, true_states)
    #
    # logging.info(f"Loss: {loss}")
    # logging.info(f"N_RMSE: {N_rmse.item():.7g}")
    #
    # # Plot diffs
    # fig, axs = plt.subplots(3, 2, figsize=(20, 8))
    # for i, ax in enumerate(axs):
    #     img_1 = true_diffs[batch_num, plot_step, :, i]
    #     img_2 = pred_diffs[batch_num, plot_step, :, i]
    #
    #     # Initial image
    #     plot_full_patches(img_1, (N_x_patch, N_y_patch), ax[0])
    #     # Predictions
    #     plot_full_patches(img_2, (N_x_patch, N_y_patch), ax[1])
    # fig.tight_layout()
    # fig.show()
    #
    # # Plot states
    # fig, axs = plt.subplots(3, 2, figsize=(20, 8))
    # for i, ax in enumerate(axs):
    #     img_1 = true_states[batch_num, plot_step, :, i]
    #     img_2 = pred_states[batch_num, plot_step, :, i]
    #
    #     # Initial image
    #     plot_full_patches(img_1, (N_x_patch, N_y_patch), ax[0])
    #     # Predictions
    #     plot_full_patches(img_2, (N_x_patch, N_y_patch), ax[1])
    # fig.tight_layout()
    # fig.show()


def main(args):
    load_no = -1
    plot_step = 0
    batch_num = 0
    save_epoch = 300

    set_seed()
    inference_params = load_yaml_from_file(args.config_path)
    logging.info(f"Parameters for inference: {inference_params}")

    # Load the checkpoint
    load_path = get_save_folder(inference_params['checkpoint_save_path'], load_no=load_no)
    checkpoint_file_path = os.path.join(load_path, f'step_{save_epoch}.pth')
    logging.info(f"Loading checkpoint from: {checkpoint_file_path}")

    if not os.path.exists(checkpoint_file_path):
        raise ValueError(f"Checkpoint file not found at {checkpoint_file_path}")

    ckpt = torch.load(checkpoint_file_path)
    ckpt_params = ckpt['params']
    ckpt_state_dict = ckpt['state_dict']

    # Get dataloader
    dl, ds_props = get_data_loader(ckpt_params, mode="valid")

    # Get the model
    model = MultivariateTimeLLM(ckpt_params, ds_props=ds_props, device_map=get_available_device())
    # Load weights
    model.load_state_dict(ckpt_state_dict)

    accelerator = get_accelerator(use_deepspeed=False, precision='bf16')
    model = accelerator.prepare(model)

    # Run test_generate
    test_generate(model, inference_params, plot_step, batch_num)

    # Run evaluation
    # evaluate_model(model, inference_params, mode='valid')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/inference1.yaml",
                        help='Path to the json config for inference')

    args = parser.parse_args(sys.argv[1:])
    main(args)
