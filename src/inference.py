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

torch._dynamo.config.cache_size_limit = 1

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def get_eval_dl(model, bs, seq_len):
    ds = MGNDataset(load_dir=f"./ds/MGN/cylinder_dataset/valid",
                    resolution=model.config['resolution'],
                    patch_size=model.config['patch_size'],
                    stride=model.config['stride'],
                    seq_len=seq_len,
                    seq_interval=model.config['seq_interval'],
                    mode='test',
                    normalize=model.config['normalize_ds'])

    dl = DataLoader(ds, batch_size=bs, pin_memory=True)
    return dl


@torch.inference_mode()
def test_generate(model: MultivariateTimeLLM, dl, plot_step, batch_num=0):
    model.eval()

    # Keep the first batch for plotting
    first_batch = None
    N_rmses = []
    # Get batch and run through model
    for i, batch in enumerate(dl):
        print(f"Batch {i}")
        batch = [t.cuda() for t in batch]
        states, _, diffs, bc_mask, position_ids = batch

        bs, seq_len, N_patch, channel, px, py = states.shape
        pred_states, pred_diffs = model.gen_seq(batch, pred_steps=seq_len - 1)
        pred_states = pred_states[:, :-1]

        true_states = patch_to_img(states, model.ds_props)
        true_diffs = patch_to_img(diffs, model.ds_props)
        bc_mask = patch_to_img(bc_mask.float(), model.ds_props).bool()
        # print(f'{pred_states.shape = }, {pred_diffs.shape = }')
        # print(f'{true_states.shape = }, {true_diffs.shape = }')

        N_rmse = calc_n_rmse(pred_states, true_states, bc_mask)
        N_rmses.append(N_rmse)

        if first_batch is None:
            first_batch = (true_states, true_diffs, pred_states, pred_diffs)

    N_rmses = torch.cat(N_rmses, dim=0)
    N_rmse = torch.mean(N_rmses, dim=0)
    first_rmses = torch.mean(N_rmse[:15])
    c_print(f"First 15 N_RMSE: {first_rmses:.3g}", color='green')
    c_print(f"Standard N_RMSE: {N_rmse}, Mean: {N_rmse.mean().item():.3g}", color='cyan')

    # Plotting
    if True:
        true_states, true_diffs, pred_states, pred_diffs = first_batch
        # Plot diffs
        fig, axs = plt.subplots(3, 2, figsize=(20, 9))
        fig.suptitle(f'Differences, step {plot_step}')
        for i, ax in enumerate(axs):
            img_1 = true_diffs[batch_num, plot_step, i].cpu()
            img_2 = pred_diffs[batch_num, plot_step, i].cpu()

            vmin, vmax = img_1.min(), img_1.max()

            ax[0].imshow(img_1.T, vmin=vmin, vmax=vmax)  # Initial image
            ax[1].imshow(img_2.T, vmin=vmin, vmax=vmax)  # Predictions
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


def main():
    load_no = -1
    save_epoch = 500
    seq_len = 251
    bs = 16

    plot_step = 25
    plot_batch_num = 0

    set_seed()

    # Load the checkpoint
    load_path = get_save_folder("./model_checkpoints", load_no=load_no)
    checkpoint_file_path = os.path.join(load_path, f'step_{save_epoch}.pth')
    logging.info(f"Loading checkpoint from: {checkpoint_file_path}")

    if not os.path.exists(checkpoint_file_path):
        raise ValueError(f"Checkpoint file not found at {checkpoint_file_path}")

    ckpt = torch.load(checkpoint_file_path)
    ckpt_state_dict = ckpt['state_dict']
    ckpt_params = load_yaml_from_file(f'{load_path}/training1.yaml')
    # ckpt_params['compile'] = False

    # Get dataloader
    ckpt_params['seq_len'] = ckpt_params['autoreg_seq_len']
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
    test_generate(model, dl, plot_step, plot_batch_num)
    # test_step(model, dl, plot_step, batch_num)
    # get_ds_stats(model, dl)


if __name__ == '__main__':
    main()
