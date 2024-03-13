"""
Testing
"""

import sys
import argparse
import logging
import torch

from utils import set_seed, load_params_from_file, get_available_device
from models.model import MultivariateTimeLLM

from dataloader.MGN_dataloader import MGNSeqDataloader
from dataloader.parallel_dataloader import ParallelDataGenerator
from dataloader.mesh_utils import plot_patches
from sequence_generate import next_state

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')
DEVICE = get_available_device()
DTYPE = torch.float16


def loss_fn(preds: torch.Tensor, diffs: torch.Tensor, bc_mask: torch.Tensor):
    # When calculating loss, need to mask out BCs

    mse_error = (preds - diffs) ** 2
    mse_error = mse_error * torch.logical_not(bc_mask)

    print(f'{mse_error.shape = }')
    loss = mse_error.mean()
    return loss


def test_generate(model: MultivariateTimeLLM, cfg):
    # Init dataloader
    patch_size = cfg['patch_size']
    resolution = cfg['resolution']
    dl = MGNSeqDataloader(load_dir="./ds/MGN/cylinder_dataset", resolution=resolution,
                          patch_size=patch_size, stride=patch_size, seq_len=5, seq_interval=2)
    # parallel_dl = ParallelDataGenerator(dl, bs=cfg['batch_size'])
    # parallel_dl.run()
    N_patch = dl.N_patch

    # Get batch from dataloader
    # states, diffs, bc_mask, position_ids = parallel_dl.get_batch()

    states, diffs, bc_mask, position_ids = dl.ds_get()
    states = states.unsqueeze(0)
    diffs = diffs.unsqueeze(0)
    bc_mask = bc_mask.unsqueeze(0)
    position_ids = position_ids.unsqueeze(0)

    states, diffs = states.to(DTYPE), diffs.to(DTYPE)
    states, diffs, position_ids, bc_mask = states.to(DEVICE), diffs.to(DEVICE), position_ids.to(DEVICE), bc_mask.to(DEVICE)

    # Initial states
    all_states = states[:, :N_patch]
    print(f'{states.shape = }, {diffs.shape = }, {bc_mask.shape = }, {position_ids.shape = }')

    # Send to model
    for i in range(N_patch):
        print(i)
        pos_id = position_ids[:, :N_patch + i]
        # Need patch and mask at t-1
        last_patch = all_states[:, i:i + 1]
        mask = bc_mask[:, i + N_patch:i + 1 + N_patch]

        with torch.no_grad():
            _, pred_diff = model.forward(all_states, pos_id)
        pred_diff = pred_diff[:, -1:] * 0.01

        new_state = next_state(last_patch, pred_diff, mask)
        all_states = torch.cat([all_states, new_state], dim=1)

    # Plotting
    img_1 = all_states[0, :N_patch, 0]
    img_2 = all_states[0, N_patch:2 * N_patch, 0]

    # Initial image
    plot_patches(img_1, (15, 4))

    # Predictions
    plot_patches(img_2, (15, 4))
    # parallel_dl.stop()
    return


def test_loop(model: MultivariateTimeLLM, cfg):
    # Init dataloader
    patch_size = cfg['patch_size']
    resolution = cfg['resolution']
    dl = MGNSeqDataloader(load_dir="./ds/MGN/cylinder_dataset", resolution=resolution,
                          patch_size=patch_size, stride=patch_size, seq_len=5, seq_interval=2)
    # parallel_dl = ParallelDataGenerator(dl, bs=cfg['batch_size'])
    # parallel_dl.run()

    # Get batch from dataloader
    # states, diffs, bc_mask, position_ids = parallel_dl.get_batch()
    states, diffs, bc_mask, position_ids = dl.ds_get()
    states = states.unsqueeze(0)
    diffs = diffs.unsqueeze(0)
    bc_mask = bc_mask.unsqueeze(0)
    position_ids = position_ids.unsqueeze(0)

    print(f'{states.shape = }, {diffs.shape = }, {bc_mask.shape = }, {position_ids.shape = }')

    states, diffs = states.to(DTYPE), diffs.to(DTYPE)
    states, diffs, position_ids, bc_mask = states.to(DEVICE), diffs.to(DEVICE), position_ids.to(DEVICE), bc_mask.to(DEVICE)

    # Send to model
    backbone_out, preds = model.forward(states, position_ids)

    # Backward pass
    loss = loss_fn(preds, diffs, bc_mask)
    print(f'Loss: {loss.item()}')
    loss.backward()

    print()
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f'{n = }, {p.grad.shape = }')

    # parallel_dl.stop()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/training1.json",
                        # required=True,
                        help='Path to the json config for training')

    args = parser.parse_args(sys.argv[1:])

    set_seed()
    training_params = load_params_from_file(args.config_path)
    logging.info(f"Parameters for training: {training_params}")

    # Test dummy data
    N, M = training_params["patch_size"]

    # Test model forward pass
    model = MultivariateTimeLLM(training_params, device_map=DEVICE).to(DEVICE).to(DTYPE)
    test_loop(model, training_params)
