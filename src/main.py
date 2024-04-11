"""
Main entrypoint for training
"""
import os
import sys
import argparse
import logging
from random import random
from statistics import mean
from cprint import c_print
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from tqdm import trange, tqdm

from trainer import Trainer, get_data_loader
from utils import set_seed, load_yaml_from_file, get_available_device, get_accelerator, make_save_folder, save_cfg
from models.model import MultivariateTimeLLM

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def get_model(training_params, dl):
    # Get the model
    model = MultivariateTimeLLM(training_params, device_map=get_available_device())

    # Get the Trainer
    trainer = Trainer(params=training_params,
                      model=model,
                      N_patch=dl.dataset.N_patch)

    optimizer, scheduler = trainer.prepare_optimizers()

    return model, optimizer, scheduler, trainer


def select_run_mode(trainer: Trainer, gen_cfg, epoch):
    if gen_cfg['ratio'] < 0.0 or gen_cfg['ratio'] > 1.0:
        raise ValueError("Invalid teacher forcing ratio. Must be between 0 and 1.")

    if gen_cfg['start_epoch'] != 0 and epoch < gen_cfg['start_epoch']:
        return trainer.run_train_step, "Autoreg"

    use_teacher_forcing = random() < gen_cfg['ratio']
    if not use_teacher_forcing:
        return trainer.run_gen_train_step, "Gen"
    else:
        return trainer.run_train_step, "Autoreg"


def process_metrics(metrics_per_epoch, epoch_len, run_mode, mode: str):
    # === Aggregate metrics across iterations in the epoch ===
    metrics_names = metrics_per_epoch[0].keys()
    metrics_agg = {f"{mode}/{run_mode}_{metric_name}": sum(d[metric_name] for d in metrics_per_epoch)
                                                       / epoch_len
                   for metric_name in metrics_names}

    return metrics_agg, metrics_agg[f"{mode}/{run_mode}_loss"], metrics_agg[f"{mode}/{run_mode}_N_RMSE"]


def run_train_epoch(run_fn: callable, dataloader, trainer: Trainer, optimizer, scheduler, accelerator: Accelerator):
    metrics_per_epoch = []
    dataloader_iterator = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(dataloader_iterator):
        states, diffs, bc_mask, position_ids = batch
        batch = (states.to(accelerator.device), diffs.to(accelerator.device),
                 bc_mask.to(accelerator.device), position_ids.to(accelerator.device))

        optimizer.zero_grad(set_to_none=True)
        with accelerator.accumulate([trainer.model]):
            loss, log_metrics = run_fn(batch)

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()

            if batch_idx % 5 == 0:
                dataloader_iterator.set_description(
                    f"Iterating batches (Batch Idx: {batch_idx + 1} | Loss: {log_metrics['loss']:.3g} | N_RMSE: {log_metrics['N_RMSE']:.3g})")
                dataloader_iterator.refresh()

        # Keep track of metrics
        metrics_per_epoch.append(log_metrics)

    scheduler.step()

    return metrics_per_epoch


def val_epoch(val_dl, trainer, accelerator: Accelerator):
    val_metrics_ep = []
    dl_iterator = tqdm(val_dl, desc="Validation", leave=False)
    for batch_idx, batch in enumerate(dl_iterator):
        states, diffs, bc_mask, position_ids = batch
        batch = (states.to(accelerator.device), diffs.to(accelerator.device),
                 bc_mask.to(accelerator.device), position_ids.to(accelerator.device))

        loss, log_metrics_dict = trainer.run_gen_val_step(batch)

        val_metrics_ep.append(log_metrics_dict)
    return val_metrics_ep


def train_run(train_params, save_path, train_dataloader, valid_dataloader, trainer, optimizer, scheduler, accelerator, start_ep=0):
    epoch_iterator = trange(train_params["num_epochs"], desc="Training", position=0, leave=True)
    for epoch_idx, epoch in enumerate(epoch_iterator):

        # Train Step
        ep_train_fn, run_mode = select_run_mode(trainer, train_params['teacher_forcing'], epoch + start_ep)

        train_log_metrics = run_train_epoch(run_fn=ep_train_fn,
                                            dataloader=train_dataloader,
                                            trainer=trainer,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            accelerator=accelerator)

        train_log, loss, nrmse = process_metrics(train_log_metrics, len(train_dataloader), run_mode, "train")
        wandb.log(train_log, step=epoch_idx + start_ep)

        # Validation Step
        val_metrics = val_epoch(valid_dataloader, trainer, accelerator)
        val_log, val_loss, val_nmrse = process_metrics(val_metrics, len(valid_dataloader), "Gen", "val")
        wandb.log(val_log, step=epoch_idx + start_ep)

        epoch_iterator.set_description(
            f"Epoch: {epoch_idx + 1}: "
            f"Training (Loss: {loss:.4g} | N_RMSE: {nrmse:.7g}) - "
            f"Validation (Loss: {val_loss:.4g} | N_RMSE: {val_nmrse:.7g})")
        epoch_iterator.refresh()

        # Save model checkpoint
        if train_params['save_on'] and train_params['save_model_each'] > 0 and epoch_idx % train_params['save_model_each'] == 0 and epoch_idx > 0:
            accelerator.wait_for_everyone()
            checkpoint_file_path = os.path.join(save_path, f'step_{epoch_idx}.pth')

            checkpoint = {'params': train_params,
                          'state_dict': trainer.model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict()}

            logging.info(f"Saving model checkpoint at epoch {epoch_idx} to {checkpoint_file_path}")
            torch.save(checkpoint, checkpoint_file_path)


def run_everything(training_params, train_dataloader, valid_dataloader, model_components, args, start_ep=0):
    model, optimizer, scheduler, trainer = model_components

    # Prepare accelerator
    accelerator = get_accelerator(use_deepspeed=training_params['use_deepspeed'], precision='bf16')
    if training_params['use_deepspeed']:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = training_params[
                                                                                                    'batch_size'] // accelerator.state.num_processes
    # Prepare model, optimizer and dataloader for accelerate training
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    trainer.model = model

    # Wandb, save and logging
    if training_params['enable_wandb'] is False:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project="llm4multivariatets", entity="adrianbzgteam", config=training_params)
    save_path = make_save_folder(training_params['checkpoint_save_path'], args.save_folder, save_on=training_params['save_on'])
    logging.info(f"Saving checkpoints to: {save_path}")
    save_cfg(args.config_path, save_path)  # WandB saves it, but make another copy anyway.

    train_run(training_params, save_path, train_dataloader, valid_dataloader, trainer, optimizer, scheduler, accelerator, start_ep=start_ep)

    # Close wandb
    wandb.finish()


def main(args):
    set_seed()
    training_params = load_yaml_from_file(args.config_path)
    logging.info(f"Parameters for training: {training_params}")

    train_dataloader = get_data_loader(training_params, mode="train")
    valid_dataloader = get_data_loader(training_params, mode="valid")
    model_components = get_model(training_params, train_dataloader)

    run_everything(training_params, train_dataloader, valid_dataloader, model_components, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/training1.yaml",
                        help='Path to the json config for training')
    parser.add_argument('--save_folder',
                        help='Path to save model checkpoints. Defaults to time', default=None)

    args = parser.parse_args(sys.argv[1:])
    main(args)
