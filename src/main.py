"""
Testing
"""

import sys
import argparse
import logging
import torch
from tqdm import trange, tqdm

from trainer import Trainer, get_data_loader
from utils import set_seed, load_params_from_file, get_available_device
from models.model import MultivariateTimeLLM

from dataloader.MGN_dataloader import MGNSeqDataloader
from dataloader.parallel_dataloader import ParallelDataGenerator, SingleDataloader

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def test_generate(model: MultivariateTimeLLM, cfg):
    ds = MGNSeqDataloader(load_dir=cfg['load_dir'],
                          resolution=cfg['resolution'],
                          patch_size=cfg['patch_size'],
                          stride=cfg['stride'],
                          seq_len=cfg['seq_len'],
                          seq_interval=cfg['seq_interval'])
    N_patch = ds.N_patch

    if cfg['multiprocess']:
        dl = ParallelDataGenerator(ds, bs=1, num_procs=1)
        dl.run()
    else:
        dl = SingleDataloader(ds, bs=1)

    trainer.model.eval()

    # Get batch and run through model
    states, diffs, bc_mask, position_ids = dl.get_batch()
    model.generate(states, diffs, bc_mask, position_ids, N_patch, show_num=0)
    model.generate(states, diffs, bc_mask, position_ids, N_patch, show_num=1)
    model.generate(states, diffs, bc_mask, position_ids, N_patch, show_num=2)


def run_train_epoch(dataloader, trainer: Trainer, optimizer):
    trainer.model.train()

    metrics_per_epoch = []
    dataloader_iterator = tqdm(dataloader, desc="Iterating batches", leave=False)
    for batch_idx, batch in enumerate(dataloader_iterator):
        states, diffs, bc_mask, position_ids = batch

        loss, log_metrics_dict = trainer.run_train_step(states, diffs, bc_mask, position_ids)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
        optimizer.step()

        dataloader_iterator.set_description(f"Iterating batches (Batch Idx: {batch_idx+1} | Loss: {log_metrics_dict['train_loss']:.3g})")
        dataloader_iterator.refresh()

        # Keep track of metrics
        metrics_per_epoch.append(log_metrics_dict)

        # === Aggregate metrics across iterations in the epoch ===
    metrics_names = metrics_per_epoch[0].keys()
    metrics_agg = {f"train/{metric_name}": sum(d[metric_name]
                                               for d in metrics_per_epoch) / len(metrics_per_epoch)
                   for metric_name in metrics_names}
    return metrics_agg


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
    train_dataloader = get_data_loader(training_params)

    trainer = Trainer(params=training_params,
                      model=model,
                      precision=precision,
                      device=get_available_device())

    optimizer = trainer.prepare_optimizers()

    epoch_iterator = trange(training_params["num_epochs"], desc="Training", position=0, leave=True)
    for epoch_idx, epoch in enumerate(epoch_iterator):
        train_log_metrics = run_train_epoch(dataloader=train_dataloader,
                                            trainer=trainer,
                                            optimizer=optimizer)

        epoch_iterator.set_description(f"Training (Epoch: {epoch_idx+1} | Loss: {train_log_metrics['train/train_loss']})")
        epoch_iterator.refresh()

    test_generate(model, training_params)

