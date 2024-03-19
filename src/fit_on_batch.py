"""
Testing
"""

import sys
import argparse
import logging
import torch
from tqdm import trange, tqdm
from cprint import c_print

from trainer import Trainer, get_data_loader
from utils import set_seed, load_params_from_file, get_available_device
from models.model import MultivariateTimeLLM

from dataloader.MGN_dataloader import MGNSeqDataloader
from dataloader.parallel_dataloader import ParallelDataGenerator, SingleDataloader

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def test_generate(model: MultivariateTimeLLM, cfg, batch):
    ds = MGNSeqDataloader(load_dir=cfg['load_dir'],
                          resolution=cfg['resolution'],
                          patch_size=cfg['patch_size'],
                          stride=cfg['stride'],
                          seq_len=cfg['seq_len'],
                          seq_interval=cfg['seq_interval'])
    N_patch = ds.N_patch

    # Get batch and run through model
    states, diffs, bc_mask, position_ids = batch
    model.generate(states, diffs, bc_mask, position_ids, N_patch, show_num=1)


def run_train_epoch(trainer: Trainer, optimizer, batch):
    trainer.model.train()

    states, diffs, bc_mask, position_ids = batch

    loss, log_metrics_dict = trainer.run_train_step(states, diffs, bc_mask, position_ids)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
    optimizer.step()

    return log_metrics_dict


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

    # Get the model
    precision = torch.bfloat16 if training_params['half_precision'] else torch.float32
    model = MultivariateTimeLLM(training_params, device_map=get_available_device(), precision=precision)

    # Get the train data loader
    training_params['multiprocess'] = False
    train_dataloader = get_data_loader(training_params)

    trainer = Trainer(params=training_params,
                      model=model,
                      precision=precision,
                      device=get_available_device())

    optimizer = trainer.prepare_optimizers()

    batch = next(iter(train_dataloader))

    for epoch in trange(100, desc="Training"):
        train_log_metrics = run_train_epoch(trainer=trainer,
                                            optimizer=optimizer,
                                            batch=batch)

        if epoch % 5 == 0:
            logging.info(f'[TRAIN]: Epoch [{epoch + 1}] Metrics: {train_log_metrics}')

    test_generate(model, training_params, batch)
