"""
Main entrypoint for training
"""
import os
import sys
import argparse
import logging
from random import random
from cprint import c_print
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from tqdm import trange, tqdm

from trainer import Trainer
from utils_model import get_data_loader
from utils import set_seed, load_yaml_from_file, get_available_device, get_accelerator, make_save_folder, save_cfg, process_metrics
from models.model import MultivariateTimeLLM
from dataloader.ds_props import DSProps

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def get_model(training_params, ds_props):
    # Get the model
    model = MultivariateTimeLLM(training_params, ds_props, device_map=get_available_device())

    # Get the Trainer
    trainer = Trainer(params=training_params,
                      model=model,
                      ds_props=ds_props)

    optimizer, scheduler = trainer.prepare_optimizers()

    return model, optimizer, scheduler, trainer


def select_run_mode(trainer: Trainer, gen_cfg, train_dl, gen_dl, epoch):
    gen_settings = (trainer.run_gen_train_step, gen_dl, 'Gen')
    autoreg_settings = (trainer.run_train_step, train_dl, 'Autoreg')

    if gen_cfg['start_epoch'] != 0 and epoch < gen_cfg['start_epoch']:
        return autoreg_settings

    use_teacher_forcing = random() < gen_cfg['tf_prob']
    if use_teacher_forcing:
        return gen_settings
    else:
        return autoreg_settings


def run_train_epoch(run_fn: callable, dataloader, trainer: Trainer, optimizer, scheduler, accelerator: Accelerator):
    metrics_per_epoch = []
    dataloader_iterator = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(dataloader_iterator):

        optimizer.zero_grad(set_to_none=True)
        with accelerator.accumulate([trainer.model]):
            loss, log_metrics = run_fn(batch)

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()

            if batch_idx % 5 == 0:
                dataloader_iterator.set_description(
                    f"Iterating batches (Batch Idx: {batch_idx + 1} | Loss: {log_metrics['loss']:.3g} | N_RMSE: {log_metrics['N_RMSE'].mean():.3g})")
                dataloader_iterator.refresh()
        # Keep track of metrics
        metrics_per_epoch.append(log_metrics)

    scheduler.step()

    return metrics_per_epoch


def val_epoch(val_dl, trainer, accelerator: Accelerator):
    val_metrics_ep = []
    dl_iterator = tqdm(val_dl, desc="Validation", leave=False)
    for batch_idx, batch in enumerate(dl_iterator):
        log_metrics_dict = trainer.run_val_step(batch)

        val_metrics_ep.append(log_metrics_dict)
    return val_metrics_ep


def train_run(train_cfg, save_path, autoreg_dl, gen_dl, valid_dl, trainer, optimizer, scheduler, accelerator, start_ep=0):
    val_steps = len(valid_dl)

    epoch_iterator = trange(train_cfg["num_epochs"], desc="Training", position=0, leave=True)
    for epoch_idx, epoch in enumerate(epoch_iterator):
        print()
        # Train Step
        run_fn, run_dl, run_mode = select_run_mode(trainer, train_cfg['teacher_forcing'], autoreg_dl, gen_dl, epoch + start_ep)

        train_log_metrics = run_train_epoch(run_fn=run_fn,
                                            dataloader=run_dl,
                                            trainer=trainer,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            accelerator=accelerator)

        train_log, loss, nrmse = process_metrics(train_log_metrics, len(run_dl), run_mode, "train")
        train_log['lr'] = optimizer.param_groups[0]['lr']
        wandb.log(train_log, step=epoch_idx + start_ep)

        # Validation Step
        val_metrics = val_epoch(valid_dl, trainer, accelerator)
        val_log, val_loss, val_nmrse = process_metrics(val_metrics, val_steps, "Gen", "val")
        wandb.log(val_log, step=epoch_idx + start_ep)

        epoch_iterator.set_description(
            f"Epoch: {epoch_idx + 1}: "
            f"Training (Loss: {loss:.4g} | N_RMSE: {nrmse:.5g}) - "
            f"Validation (Loss: {val_loss:.4g} | N_RMSE: {val_nmrse:.5g})"
        )
        epoch_iterator.refresh()

        # Save model checkpoint
        if train_cfg['save_on'] and train_cfg['save_model_each'] > 0 and epoch_idx % train_cfg['save_model_each'] == 0 and epoch_idx > 0:
            accelerator.wait_for_everyone()
            checkpoint_file_path = os.path.join(save_path, f'step_{epoch_idx}.pth')

            checkpoint = {'params': train_cfg,
                          'state_dict': trainer.model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict()}

            logging.info(f"Saving model checkpoint at epoch {epoch_idx} to {checkpoint_file_path}")
            torch.save(checkpoint, checkpoint_file_path)


def run_everything(train_cfg, autoreg_dl, gen_dl, valid_dl, model_components, args, start_ep=0):
    model, optimizer, scheduler, trainer = model_components

    # Prepare accelerator
    accelerator = get_accelerator(precision='bf16')
    model, optimizer, autoreg_dl, gen_dl, valid_dl, scheduler = accelerator.prepare(model, optimizer, autoreg_dl, gen_dl, valid_dl, scheduler)
    trainer.model = model

    # Wandb, save and logging
    if train_cfg['enable_wandb'] is False:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project="llm4multivariatets", entity="adrianbzgteam", tags=["NewValidation"], config=train_cfg)
    wandb.save(args.config_path)
    run_name = wandb.run.name

    save_path = make_save_folder(train_cfg['checkpoint_save_path'], args.save_folder, save_on=train_cfg['save_on'])
    logging.info(f"Saving checkpoints to: {save_path}")
    save_cfg(args.config_path, save_path)  # WandB saves it, but make another copy anyway.
    with open(f"{save_path}/{run_name}", 'w') as f:
        f.write(run_name)

    train_run(train_cfg, save_path, autoreg_dl, gen_dl, valid_dl, trainer, optimizer, scheduler, accelerator, start_ep=start_ep)

    # Close wandb
    wandb.finish()


def main(args):
    set_seed()
    train_cfg = load_yaml_from_file(args.config_path)
    logging.info(f"Parameters for training: {train_cfg}")

    # Modify configs for dataloaders
    autoreg_cfg = dict(train_cfg)
    autoreg_cfg['seq_len'] = train_cfg['autoreg_seq_len']
    # gen_cfg = dict(train_cfg)
    # gen_cfg['seq_len'] = train_cfg['gen_seq_len']
    val_cfg = dict(train_cfg)
    val_cfg['seq_len'] = train_cfg['val_seq_len']

    autoreg_dl, ds_props = get_data_loader(autoreg_cfg, mode="train")  # Main dataloader for model params max_seq_len.
    gen_dl = autoreg_dl  # get_data_loader(gen_cfg, mode="train")
    valid_dl, _ = get_data_loader(val_cfg, mode="valid")

    model_components = get_model(train_cfg, ds_props)
    run_everything(train_cfg, autoreg_dl, gen_dl, valid_dl, model_components, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/training1.yaml",
                        help='Path to the json config for training')
    parser.add_argument('--save_folder',
                        help='Path to save model checkpoints. Defaults to time', default=None)

    args = parser.parse_args(sys.argv[1:])
    main(args)
