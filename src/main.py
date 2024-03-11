"""
Testing
"""

import sys
import argparse
import logging
import torch
import torch.nn.functional as F

from utils import set_seed, load_params_from_file
from models.model import MultivariateTimeLLM

from dataloader.MGN_dataloader import MGNSeqDataloader

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')
DEVICE = 'cuda'
DTYPE = torch.float16


def test_loop(model: MultivariateTimeLLM, cfg):
    patch_size = cfg['patch_size']
    resolution = cfg['resolution']
    dl = MGNSeqDataloader(load_dir="../ds/MGN/cylinder_dataset", resolution=resolution,
                          patch_size=patch_size, stride=patch_size, seq_len=5, seq_interval=2)

    states, diffs, bc_mask, position_ids = dl.get_sequence()

    states, diffs = states.to(DTYPE), diffs.to(DTYPE)
    states, diffs, position_ids, bc_mask = states.to(DEVICE), diffs.to(DEVICE), position_ids.to(DEVICE), bc_mask.to(DEVICE)
    backbone_out, preds = model.forward(states, position_ids)

    # When calculating loss, need to mask out BCs
    print(f'{preds.shape = }, {diffs.shape = }, {bc_mask.shape = }')

    mse_error = (preds - diffs) ** 2
    mse_error = mse_error * torch.logical_not(bc_mask)
    loss = mse_error.mean()

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
    model = MultivariateTimeLLM(training_params, N=N, M=M, device_map=DEVICE).to(DEVICE).to(DTYPE)
    test_loop(model, training_params)
