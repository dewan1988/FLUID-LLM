"""
Main entrypoint for training
"""
import sys
import argparse
import logging

import torch

from cprint import c_print

from trainer import Trainer, get_data_loader
from utils import set_seed, load_yaml_from_file, get_available_device
from models.model import MultivariateTimeLLM
from main import run_everything

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def load_model(save_file, training_params, train_dl):
    # Get the model
    model = MultivariateTimeLLM(training_params, device_map=get_available_device())
    model.load_state_dict(save_file['state_dict'])

    # Get trainer
    trainer = Trainer(params=training_params,
                      model=model,
                      N_patch=train_dl.dataset.N_patch)

    # Load optimizer and scheduler
    optimizer, scheduler = trainer.prepare_optimizers()
    optimizer.load_state_dict(save_file['optimizer'])
    scheduler.load_state_dict(save_file['scheduler'])

    return model, optimizer, scheduler, trainer


def main(args):
    set_seed()
    load_dir = f"./model_checkpoints"
    load_file = "04-11_04-40-56"
    load_num = 80

    save_file = torch.load(f'{load_dir}/{load_file}/step_{load_num}.pth')
    # Use saved .yaml config for easier editing
    training_params = load_yaml_from_file(f'{load_dir}/{load_file}/training1.yaml')
    c_print("Loading Config", color='bright_green')
    c_print(training_params, color='green')

    train_dataloader = get_data_loader(training_params, mode="train")
    valid_dataloader = get_data_loader(training_params, mode="valid")
    model_components = load_model(save_file, training_params, train_dataloader)

    run_everything(training_params, train_dataloader, valid_dataloader, model_components, args, start_ep=load_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/training1.yaml",
                        help='Path to the json config for training')
    parser.add_argument('--save_folder',
                        help='Path to save model checkpoints. Defaults to time', default=None)

    args = parser.parse_args(sys.argv[1:])
    main(args)
