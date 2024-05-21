"""
Main entrypoint for training
"""
import sys
import argparse
import logging

import torch

from cprint import c_print

from trainer import Trainer
from utils_model import get_data_loader
from utils import set_seed, load_yaml_from_file, get_available_device
from models.model import MultivariateTimeLLM
from main import run_everything

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def load_model(save_file, training_params, ds_props):
    # Get the model
    model = MultivariateTimeLLM(training_params, ds_props, device_map=get_available_device())
    model.load_state_dict(save_file['state_dict'])

    # Get trainer
    trainer = Trainer(params=training_params,
                      model=model,
                      ds_props=ds_props)

    # Load optimizer and scheduler
    optimizer, scheduler = trainer.prepare_optimizers()
    optimizer.load_state_dict(save_file['optimizer'])
    scheduler.load_state_dict(save_file['scheduler'])

    return model, optimizer, scheduler, trainer


def main(args):
    set_seed()
    load_dir = f"./model_checkpoints"
    load_file = "05-15_14-27-01"
    load_num = 120

    save_file = torch.load(f'{load_dir}/{load_file}/step_{load_num}.pth')
    # Use saved .yaml config for easier editing
    train_cfg = load_yaml_from_file(f'{load_dir}/{load_file}/training1.yaml')
    c_print("Loading Config", color='bright_green')
    c_print(train_cfg, color='green')

    autoreg_cfg = dict(train_cfg)
    autoreg_cfg['seq_len'] = train_cfg['autoreg_seq_len']
    # gen_cfg = dict(train_cfg)
    # gen_cfg['seq_len'] = train_cfg['tf_seq_len']
    val_cfg = dict(train_cfg)
    val_cfg['seq_len'] = train_cfg['val_seq_len']

    autoreg_dl, ds_props = get_data_loader(autoreg_cfg, mode="train")  # Dataloader target diffs
    gen_dl = autoreg_dl # , _ = get_data_loader(gen_cfg, mode="train")  # Dataloader returns next state
    valid_dl, _ = get_data_loader(val_cfg, mode="valid")

    model_components = load_model(save_file, train_cfg, ds_props)
    run_everything(train_cfg, autoreg_dl, gen_dl, valid_dl, model_components, args, start_ep=load_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/training1.yaml",
                        help='Path to the json config for training')
    parser.add_argument('--save_folder',
                        help='Path to save model checkpoints. Defaults to time', default=None)

    args = parser.parse_args(sys.argv[1:])
    main(args)
