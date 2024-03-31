"""
Testing loading a saved model and running the test_generate function
"""
import os
import sys
import argparse
import logging
import torch
import matplotlib.pyplot as plt

from utils import set_seed, load_yaml_from_file, get_available_device, get_save_folder
from models.model import MultivariateTimeLLM

from dataloader.MGN_dataloader import MGNSeqDataloader
from dataloader.parallel_dataloader import ParallelDataGenerator, SingleDataloader
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


def test_generate(model: MultivariateTimeLLM, cfg):
    bs = cfg['batch_size']
    ds = MGNSeqDataloader(load_dir=cfg['load_dir'],
                          resolution=cfg['resolution'],
                          patch_size=cfg['patch_size'],
                          stride=cfg['stride'],
                          seq_len=cfg['seq_len'],
                          seq_interval=cfg['seq_interval'])
    N_patch = ds.N_patch

    dl = SingleDataloader(ds, bs=bs)

    model.eval()

    # Get batch and run through model
    batch_data = dl.get_batch()
    true_states, true_diffs = batch_data[0], batch_data[1]
    # model.precision = torch.float32
    pred_states, pred_diffs = model.generate(batch_data, N_patch)

    # Split into steps
    pred_states = pred_states.view(bs, -1, N_patch, 3, 16, 16).cpu()
    true_states = true_states.view(bs, -1, N_patch, 3, 16, 16).to(torch.float32)
    pred_diffs = pred_diffs.view(bs, -1, N_patch, 3, 16, 16).cpu()
    true_diffs = true_diffs.view(bs, -1, N_patch, 3, 16, 16)

    loss = rmse_loss(pred_states, true_states)
    print(loss)

    # Plotting
    plot_step = -1
    batch_num = 1

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

    print(f'{pred_states.shape = }, {true_states.shape = }')


def load_model(checkpoint_file_path):
    checkpoint = torch.load(checkpoint_file_path)
    checkpoints_params = checkpoint['params']
    checkpoint_state_dict = checkpoint['state_dict']

    # Get the model
    precision = torch.bfloat16 if checkpoints_params['half_precision'] else torch.float32
    model = MultivariateTimeLLM(checkpoints_params, device_map=get_available_device(), precision=precision)

    # Load weights
    model.load_state_dict(checkpoint_state_dict)
    return model


def analyse_weight(p):
    # w = p.input_embeddings.patch_embeddings.encoder.layers[1].weight
    w = p.input_embeddings.position_embeddings.x_embeddings.weight
    w = w.to(torch.float32)
    w = w[:7]

    sum = w.sum().item()
    std = w.std().item()
    norm = w.norm().item()
    print(f'{sum = :.4g}, {std = :.4g} {norm = :.4g}')


def main(args):
    set_seed()
    inference_params = load_yaml_from_file(args.config_path)
    logging.info(f"Parameters for inference: {inference_params}")

    # Load the checkpoint
    load_path = get_save_folder(inference_params['checkpoint_save_path'], load_no=-1)

    checkpoint_file_path = f'{load_path}/step_{0}.pth'
    model = load_model(checkpoint_file_path)
    analyse_weight(model)

    checkpoint_file_path = f'{load_path}/step_{20}.pth'
    model = load_model(checkpoint_file_path)
    analyse_weight(model)

    exit(4)

    analyse_weight(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/inference1.yaml",
                        help='Path to the json config for inference')

    args = parser.parse_args(sys.argv[1:])
    main(args)
