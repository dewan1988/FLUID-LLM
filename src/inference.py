"""
Testing loading a saved model and running the test_generate function
"""
import os
import sys
import argparse
import logging
import torch

from utils import set_seed, load_yaml_from_file, get_available_device
from models.model import MultivariateTimeLLM

from dataloader.MGN_dataloader import MGNSeqDataloader
from dataloader.parallel_dataloader import ParallelDataGenerator, SingleDataloader

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')

def rmse_loss(pred_state, true_state):
    """ state.shape = (bs, num_steps, N_patch, 3, 16, 16)"""
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
    pred_state = model.generate(batch_data, N_patch).cpu()
    true_state = batch_data[0]

    # Split into steps
    pred_state = pred_state.view(bs, -1, N_patch, 3, 16, 16)
    true_state = true_state.view(bs, -1, N_patch, 3, 16, 16)

    loss = rmse_loss(pred_state, true_state)
    print(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/inference1.yaml",
                        help='Path to the json config for inference')

    args = parser.parse_args(sys.argv[1:])

    set_seed()
    inference_params = load_yaml_from_file(args.config_path)
    logging.info(f"Parameters for inference: {inference_params}")

    # Load the checkpoint
    checkpoint_file_path = os.path.join(inference_params['checkpoint_save_path'],
                                        f'llm4multivariatets_step_{inference_params["step_to_load"]}.pth')

    if not os.path.exists(checkpoint_file_path):
        raise ValueError(f"Checkpoint file not found at {checkpoint_file_path}")

    checkpoint = torch.load(checkpoint_file_path)
    checkpoints_params = checkpoint['params']
    checkpoint_state_dict = checkpoint['state_dict']

    # Get the model
    precision = torch.bfloat16 if checkpoints_params['half_precision'] else torch.float32
    model = MultivariateTimeLLM(checkpoints_params, device_map=get_available_device(), precision=precision)

    # Load weights
    model.load_state_dict(checkpoint_state_dict)

    # Run test_generate
    test_generate(model, inference_params)
