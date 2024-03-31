import json
import yaml
import os
import pickle
import torch
import random
import logging
import numpy as np
from datetime import datetime
import shutil

from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator

GLOBAL_SEED = 123
ACCELERATOR = None
logging.basicConfig(level=logging.INFO,
                    format='[utils:%(levelname)s] %(message)s')


def set_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def freeze_model(model_to_freeze):
    for param in model_to_freeze.parameters():
        param.requires_grad = False


def unfreeze_model(model_to_unfreeze):
    for param in model_to_unfreeze.parameters():
        param.requires_grad = True


def get_model_size(model):
    num_parameters = sum(p.numel() for p in model.parameters())
    return num_parameters


def get_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_available_device():
    accelerator = get_accelerator()
    return accelerator.device


def get_accelerator():
    global ACCELERATOR
    if not ACCELERATOR:
        ACCELERATOR = Accelerator()

    return ACCELERATOR


def get_huggingface_model(model_name, args=None):
    if args:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not tokenizer.is_fast:
        raise ValueError('Only fast tokenizers are supported.')

    if args:
        model = AutoModel.from_pretrained(model_name, **args).to(get_available_device())
    else:
        model = AutoModel.from_pretrained(model_name).to(get_available_device())

    return {"tokenizer": tokenizer,
            "model": model}


def get_file_number_of_lines(file_path):
    with open(file_path) as fp:
        num_lines = sum(1 for _ in fp)

    return num_lines


def save_to_pickle(data, save_path):
    with open(save_path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_params_from_file(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f'Path to config {file_path} does not exist.')

    logging.info(f"Loading parameters from: {file_path}")
    with open(file_path) as file:
        try:
            params = json.load(file)
        except Exception as e:
            logging.error(e)

    return params


def load_yaml_from_file(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f'Path to config {file_path} does not exist.')

    logging.info(f"Loading parameters from: {file_path}")
    with open(file_path, 'r') as file:
        try:
            params = yaml.safe_load(file)
        except Exception as e:
            logging.error(e)
            exit(e)

    return params


def save_cfg(cfg_path, save_path):
    shutil.copy(cfg_path, f'{save_path}/')


def make_save_folder(save_dir, save_name=None):
    """ Make save folder. If no name is given, create a folder with the current date and time."""
    if save_name is None:
        save_name = datetime.now().strftime("%m-%d_%H-%M-%S")

    save_path = f'{save_dir}/{save_name}'
    if os.path.exists(save_path):
        raise ValueError(f"Folder {save_path} already exists.")

    os.mkdir(save_path)

    return save_path


def get_save_folder(save_dir, load_name=None, load_no=-1):
    """ Return save folder. Either by save folder name, or date order."""
    if load_name is not None:
        save_path = f'{save_dir}/{load_name}'
        if not os.path.exists(save_path):
            raise ValueError(f"Folder {save_path} does not exist.")
        return save_path

    # Get nth folder in directory
    all_items = os.listdir(save_dir)
    folders = [item for item in all_items if os.path.isdir(os.path.join(save_dir, item))]
    folders = sorted(folders)

    return os.path.join(save_dir, folders[load_no])

