"""
Main entrypoint for training
"""
import os
import sys
import argparse
import logging
from statistics import mean

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from tqdm import trange, tqdm
from cprint import c_print
import pickle

from trainer import Trainer, get_data_loader
from utils import set_seed, load_yaml_from_file, get_available_device, get_accelerator, make_save_folder, save_cfg
from models.model import MultivariateTimeLLM
from main import run_train_epoch, run_no_force_epoch, select_run_mode, train_run

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def load_model(save_file, training_params):
    # Get the model
    model = MultivariateTimeLLM(training_params, device_map=get_available_device())
    model.load_state_dict(save_file['state_dict'])

    # Get the train data loader
    train_dataloader = get_data_loader(training_params)

    # Get trainer
    trainer = Trainer(params=training_params,
                      model=model,
                      N_patch=train_dataloader.dataset.N_patch)

    # Load optimizer and scheduler
    optimizer, scheduler = trainer.prepare_optimizers()
    optimizer.load_state_dict(save_file['optimizer'])
    scheduler.load_state_dict(save_file['scheduler'])

    return model, optimizer, train_dataloader, scheduler, trainer


def main(args):
    set_seed()
    load_dir = f"./model_checkpoints"
    load_file = "04-06_03-52-33"
    load_num = 80

    save_file = torch.load(f'{load_dir}/{load_file}/step_{load_num}.pth')
    # Use saved .yaml config for easier editing
    training_params = load_yaml_from_file(f'{load_dir}/{load_file}/training1.yaml')
    c_print("Loading Config", color='bright_green')
    c_print(training_params, color='green')

    # Prepare accelerator
    accelerator = get_accelerator(use_deepspeed=training_params['use_deepspeed'])
    if training_params['use_deepspeed']:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = training_params['batch_size'] // accelerator.state.num_processes

    model, optimizer, train_dl, scheduler, trainer = load_model(save_file, training_params)
    valid_dataloader = get_data_loader(training_params, mode="valid")

    # Wandb
    if training_params['enable_wandb'] is False:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project="llm4multivariatets", entity="adrianbzgteam", config=training_params)

    # Prepare model, optimizer and dataloader for accelerate training
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dl, scheduler)
    trainer.model = model

    # Make save folder and save config
    save_path = make_save_folder(f'{training_params["checkpoint_save_path"]}',
                                 f'{load_file}_cont', save_on=training_params['save_on'])
    logging.info(f"Saving checkpoints to: {save_path}")
    save_cfg(args.config_path, save_path)  # WandB saves it, but make another copy anyway.

    train_run(training_params, save_path, train_dataloader, valid_dataloader, trainer, optimizer, scheduler, accelerator, start_ep=load_num)

    # epoch_iterator = trange(training_params["num_epochs"], desc="Training", position=0, leave=True)
    # for epoch_idx, epoch in enumerate(epoch_iterator):
    #     train_log_metrics = run_train_epoch(dataloader=train_dataloader,
    #                                         trainer=trainer,
    #                                         optimizer=optimizer,
    #                                         scheduler=scheduler,
    #                                         accelerator=accelerator)
    #
    #     wandb.log(train_log_metrics, step=epoch_idx)
    #     epoch_iterator.set_description(
    #         f"Training (Epoch: {epoch_idx + 1} | Loss: {train_log_metrics['train/train_loss']:.4g} | N_RMSE: {train_log_metrics['train/N_RMSE']:.4g})")
    #     epoch_iterator.refresh()
    #
    #     # Save model checkpoint
    #     if training_params['save_model_each'] > 0 and epoch_idx % training_params['save_model_each'] == 0:
    #         accelerator.wait_for_everyone()
    #         checkpoint_file_path = os.path.join(save_path, f'step_{epoch_idx}.pth')
    #
    #         checkpoint = {'params': training_params,
    #                       'state_dict': trainer.model.state_dict(),
    #                       'optimizer': optimizer.state_dict()}
    #
    #         logging.info(f"Saving model checkpoint at epoch {epoch_idx} to {checkpoint_file_path}")
    #         torch.save(checkpoint, checkpoint_file_path)

    # Close wandb
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/training1.yaml",
                        help='Path to the json config for training')
    parser.add_argument('--save_folder',
                        help='Path to save model checkpoints. Defaults to time', default=None)

    args = parser.parse_args(sys.argv[1:])
    main(args)
