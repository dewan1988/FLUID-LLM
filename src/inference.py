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

from utils import set_seed, load_yaml_from_file, get_available_device, get_save_folder
from metrics import calc_n_rmse
from models.model import MultivariateTimeLLM

from dataloader.simple_dataloader import MGNDataset
from dataloader.mesh_utils import plot_full_patches

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def rmse_loss(pred_state, true_state):
    """ state.shape = (bs, num_steps, N_patch, 3, 16, 16)"""
    assert pred_state.shape == true_state.shape
    pred_state = pred_state.to(torch.float32)
    true_state = true_state.to(torch.float32)

    mse_loss = torch.mean((pred_state - true_state) ** 2, dim=(0, 2, 3, 4, 5))

    return torch.sqrt(mse_loss)


@torch.no_grad()
def evaluate_model(model: MultivariateTimeLLM, eval_cfg, mode='valid'):
    bs = eval_cfg['batch_size']
    ds = MGNDataset(load_dir=f"{eval_cfg['load_dir']}/{mode}",
                    resolution=model.config['resolution'],
                    patch_size=model.config['patch_size'],
                    stride=model.config['stride'],
                    seq_len=eval_cfg['seq_len'],
                    seq_interval=model.config['seq_interval'],
                    mode=mode,
                    fit_diffs=model.config['fit_diffs'],
                    normalize=model.config['normalize_ds'])
    N_patch = ds.N_patch

    dl = DataLoader(ds, batch_size=bs, pin_memory=True)

    model.eval()

    all_nmrse = []

    # Get batch and run through model
    for eval_batch in tqdm(dl, desc="Evaluating model"):
        true_states, true_diffs, mask = eval_batch[0], eval_batch[1], eval_batch[2]
        bs, tot_patch, channel, px, py = true_states.shape
        seq_len = tot_patch // N_patch

        pred_states, _ = model.gen_seq(eval_batch, N_patch, pred_steps=seq_len - 1)

        true_states = true_states.view(bs, seq_len, N_patch, channel, px, py)
        pred_states = pred_states.view(bs, seq_len, N_patch, channel, px, py).cpu()
        mask = mask.view(bs, seq_len, N_patch, channel, px, py)

        # Split into steps
        #true_states = true_states.view(bs, eval_cfg['seq_len'] - 1, N_patch, 3, 16, 16).to(torch.float32)
        #pred_states = pred_states.view(bs, eval_cfg['seq_len'], N_patch, 3, 16, 16).cpu()
        #pred_states = pred_states[:, :-1]

        #print(true_states.shape)
        #print(pred_states.shape)
        #exit(1)

        loss = rmse_loss(pred_states, true_states)
        N_rmse = calc_n_rmse(pred_states, true_states, mask)
        all_nmrse.append(N_rmse.item())

        #logging.info(f"Loss: {loss.mean():.7g}")
        #logging.info(f"N_RMSE: {N_rmse.item():.7g}")

    # Get mean N_rmse
    mean_nrmse = sum(all_nmrse) / len(all_nmrse)
    logging.info(f"N_RMSE: {mean_nrmse:.7g}")


def test_generate(model: MultivariateTimeLLM, eval_cfg):
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

    model.eval()

    # Get batch and run through model
    batch_data = next(iter(dl))
    true_states, true_diffs = batch_data[0], batch_data[1]

    """ NEW VERSION"""
    st = time.time()
    pred_states, pred_diffs = model.gen_seq(batch_data, N_patch, pred_steps=eval_cfg['seq_len'] - 1)
    print(f"Time taken: {time.time() - st:.4g}")

    # Split into steps
    true_states = true_states.view(bs, eval_cfg['seq_len'] - 1, N_patch, 3, 16, 16).to(torch.float32)
    pred_states = pred_states.view(bs, eval_cfg['seq_len'], N_patch, 3, 16, 16).cpu()
    pred_states = pred_states[:, :-1]

    true_diffs = true_diffs.view(bs, eval_cfg['seq_len'] - 1, N_patch, 3, 16, 16)
    pred_diffs = pred_diffs.view(bs, eval_cfg['seq_len'] - 1, N_patch, 3, 16, 16).cpu()

    # """ OLD VERSION"""
    # st = time.time()
    # pred_states, pred_diffs = model.generate(batch_data, N_patch)
    # print(f'Time taken: {time.time() - st:.4g}')
    # # Split into steps
    # pred_states = pred_states.view(bs, eval_cfg['seq_len'] - 1, N_patch, 3, 16, 16).cpu()
    # true_states = true_states.view(bs, eval_cfg['seq_len'] - 1, N_patch, 3, 16, 16).to(torch.float32)
    # pred_diffs = pred_diffs.view(bs, eval_cfg['seq_len'] - 1, N_patch, 3, 16, 16).cpu()
    # true_diffs = true_diffs.view(bs, eval_cfg['seq_len'] - 1, N_patch, 3, 16, 16)

    loss = rmse_loss(pred_states, true_states)
    N_rmse = calc_n_rmse(pred_states, true_states)

    logging.info(f"Loss: {loss.mean():.7g}")
    logging.info(f"N_RMSE: {N_rmse.item():.7g}")

    # Plotting
    plot_step = -1
    batch_num = 0

    # Plot diffs
    fig, axs = plt.subplots(3, 2, figsize=(20, 8))
    for i, ax in enumerate(axs):
        img_1 = true_diffs[batch_num, plot_step, :, i]
        img_2 = pred_diffs[batch_num, plot_step, :, i]

        # Initial image
        plot_full_patches(img_1, (15, 4), ax[0])
        # Predictions
        plot_full_patches(img_2, (15, 4), ax[1])
    fig.tight_layout()
    fig.show()

    # Plot states
    fig, axs = plt.subplots(3, 2, figsize=(20, 8))
    for i, ax in enumerate(axs):
        img_1 = true_states[batch_num, plot_step, :, i]
        img_2 = pred_states[batch_num, plot_step, :, i]

        # Initial image
        plot_full_patches(img_1, (15, 4), ax[0])
        # Predictions
        plot_full_patches(img_2, (15, 4), ax[1])
    fig.tight_layout()
    fig.show()


def main(args):
    set_seed()
    inference_params = load_yaml_from_file(args.config_path)
    logging.info(f"Parameters for inference: {inference_params}")

    # Load the checkpoint
    load_path = get_save_folder(inference_params['checkpoint_save_path'], load_no=-1)
    checkpoint_file_path = os.path.join(load_path, f'step_{inference_params["step_to_load"]}.pth')
    logging.info(f"Loading checkpoint from: {checkpoint_file_path}")

    if not os.path.exists(checkpoint_file_path):
        raise ValueError(f"Checkpoint file not found at {checkpoint_file_path}")

    checkpoint = torch.load(checkpoint_file_path)
    checkpoints_params = checkpoint['params']
    checkpoint_state_dict = checkpoint['state_dict']

    # Get the model
    model = MultivariateTimeLLM(checkpoints_params, device_map=get_available_device())

    # Load weights
    model.load_state_dict(checkpoint_state_dict)

    # Run test_generate
    #test_generate(model, inference_params)

    # Run evaluation
    evaluate_model(model, inference_params, mode='valid')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/inference1.yaml",
                        help='Path to the json config for inference')

    args = parser.parse_args(sys.argv[1:])
    main(args)
