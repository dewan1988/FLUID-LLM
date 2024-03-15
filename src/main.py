"""
Testing
"""

import sys
import argparse
import logging
import torch
import time
from utils import set_seed, load_params_from_file, get_available_device
from models.model import MultivariateTimeLLM

from dataloader.MGN_dataloader import MGNSeqDataloader
from dataloader.parallel_dataloader import ParallelDataGenerator, SingleDataloader
from dataloader.mesh_utils import plot_patches
from sequence_generate import next_state

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')
DEVICE = get_available_device()
DTYPE = torch.float32


def loss_fn(preds: torch.Tensor, diffs: torch.Tensor, bc_mask: torch.Tensor):
    # When calculating loss, need to mask out BCs

    error = (preds - diffs)
    mse_error = error ** 2
    mae = torch.abs(error)

    loss = mse_error + 0.05 * mae
    loss = loss # * torch.logical_not(bc_mask)

    loss = loss.mean()
    return loss


def test_generate(model: MultivariateTimeLLM, cfg, show_dim=2):
    # Init dataloader
    patch_size = cfg['patch_size']
    resolution = cfg['resolution']
    ds = MGNSeqDataloader(load_dir="./ds/MGN/cylinder_dataset", resolution=resolution,
                          patch_size=patch_size, stride=patch_size, seq_len=10, seq_interval=2)
    N_patch = ds.N_patch

    if cfg['multiprocess']:
        dl = ParallelDataGenerator(ds, num_producers=2,bs=1)
        dl.run()
    else:
        dl = SingleDataloader(ds, bs=1)

    # Get batch from dataloader
    states, diffs, bc_mask, position_ids = dl.get_batch()
    print(f'{states.shape = }, {diffs.shape = }, {bc_mask.shape = }, {position_ids.shape = }')

    states, diffs = states.to(DTYPE), diffs.to(DTYPE)
    states, diffs, position_ids, bc_mask = states.to(DEVICE), diffs.to(DEVICE), position_ids.to(DEVICE), bc_mask.to(DEVICE)

    # Start with initial patches, and extrapolate for 1 patch
    init_patch = N_patch * 5
    seq_states = states[:, :init_patch]

    # Model reconstructs autoregressively
    pred_diffs = []
    for i in range(N_patch):
        pos_id = position_ids[:, :init_patch + i]
        # Need patch and mask at t-1
        last_patch = seq_states[:, -N_patch:-N_patch + 1]
        mask = bc_mask[:, init_patch + i: init_patch + i + 1]

        with torch.no_grad():
            _, pred_diff = model.forward(seq_states, pos_id)
        pred_diff = pred_diff[:, -1:]

        new_state = next_state(last_patch, pred_diff, mask)
        seq_states = torch.cat([seq_states, new_state], dim=1)

        pred_diffs.append(pred_diff)

    # Plotting
    img_1 = diffs[0, init_patch:init_patch + N_patch, show_dim]  # seq_states[0, init_patch - N_patch:init_patch, 0]
    img_2 = torch.stack(pred_diffs).squeeze()[:, show_dim]  # seq_states[0, init_patch:init_patch + N_patch, 0]

    # Initial image
    plot_patches(img_1, (15, 4))

    # Predictions
    plot_patches(img_2, (15, 4))

    dl.stop()
    return


def test_loop(model: MultivariateTimeLLM, cfg):
    # Init dataloader
    patch_size = cfg['patch_size']
    resolution = cfg['resolution']
    ds = MGNSeqDataloader(load_dir="./ds/MGN/cylinder_dataset", resolution=resolution,
                          patch_size=patch_size, stride=patch_size, seq_len=5, seq_interval=2)

    if cfg['multiprocess']:
        dl = ParallelDataGenerator(ds, bs=cfg['batch_size'])
        dl.run()
    else:
        dl = SingleDataloader(ds, bs=cfg['batch_size'])

    # Get batch from dataloader
    states, diffs, bc_mask, position_ids = dl.get_batch()
    print(f'{states.shape = }, {diffs.shape = }, {bc_mask.shape = }, {position_ids.shape = }')

    states, diffs = states.to(DTYPE), diffs.to(DTYPE)
    states, diffs, position_ids, bc_mask = states.to(DEVICE), diffs.to(DEVICE), position_ids.to(DEVICE), bc_mask.to(DEVICE)

    # Send to model
    backbone_out, preds = model.forward(states, position_ids)

    # Backward pass
    loss = loss_fn(preds, diffs, bc_mask)
    print(f'Loss: {loss.item()}')
    loss.backward()

    params = model.get_parameters()
    model_parameters_count = sum(p.numel() for p in params if p.requires_grad)
    print(f"The model has {model_parameters_count} trainable parameters")

    return


def train_loop(model: MultivariateTimeLLM, cfg):
    params = model.get_parameters()
    model_parameters_count = sum(p.numel() for p in params if p.requires_grad)
    print(f"The model has {model_parameters_count} trainable parameters")

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.numel())

    # Init dataloader
    patch_size = cfg['patch_size']
    resolution = cfg['resolution']
    ds = MGNSeqDataloader(load_dir="./ds/MGN/cylinder_dataset", resolution=resolution,
                          patch_size=patch_size, stride=patch_size, seq_len=10, seq_interval=2)

    if cfg['multiprocess']:
        dl = ParallelDataGenerator(ds, bs=cfg['batch_size'])
        dl.run()
    else:
        dl = SingleDataloader(ds, bs=cfg['batch_size'])

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Train loop
    sum_loss = 0.
    for i in range(250):
        states, diffs, bc_mask, position_ids = dl.get_batch()

        states, diffs = states.to(DTYPE), diffs.to(DTYPE)
        states, diffs, position_ids, bc_mask = states.to(DEVICE), diffs.to(DEVICE), position_ids.to(DEVICE), bc_mask.to(DEVICE)

        # Forward pass
        backbone_out, preds = model.forward(states, position_ids)

        # Backward pass
        loss = loss_fn(preds, diffs, bc_mask)
        loss.backward()

        optim.step()
        optim.zero_grad()

        sum_loss += loss.item()
        if i % 5 == 0:
            print(i)
            print(f'Loss: {sum_loss / 5:.2g}')
            sum_loss = 0.
    dl.stop()
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

    N, M = training_params["patch_size"]

    # Test model forward pass
    model = MultivariateTimeLLM(training_params, device_map=DEVICE).to(DEVICE).to(DTYPE)

    train_loop(model, training_params)
    test_generate(model, training_params)
    test_generate(model, training_params)
    test_generate(model, training_params)