import json
import os
import pickle
import torch
import random
import logging
import numpy as np

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
