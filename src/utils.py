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
from cprint import c_print
import natsort

from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator, DeepSpeedPlugin

GLOBAL_SEED = 1234
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
    accelerator = get_accelerator(precision='bf16')
    return accelerator.device


def get_accelerator(precision, use_deepspeed=False):
    global ACCELERATOR
    if not ACCELERATOR:
        if use_deepspeed:
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='configs/deepspeed_zero2.json')
            ACCELERATOR = Accelerator(deepspeed_plugin=deepspeed_plugin)
        else:
            ACCELERATOR = Accelerator(mixed_precision='bf16')

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


def make_save_folder(save_dir, save_name: str, save_on: bool):
    """ Make save folder. If no name is given, create a folder with the current date and time."""

    if not save_on:
        c_print("Warning: Save is off. Dumping save files to /tmp.", color='red')
        save_dir = '/tmp'
        save_name = 'LLM4TS_data_dump'

    if save_name is None:
        save_name = datetime.now().strftime("%m-%d_%H-%M-%S")

    save_path = f'{save_dir}/{save_name}'
    if os.path.exists(save_path):
        c_print(f"Warning: Folder {save_path} already exists.", color='red')

    os.makedirs(save_path, exist_ok=True)

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
    folders = natsort.natsorted(folders)
    return os.path.join(save_dir, folders[load_no])


def process_metrics(metrics_per_epoch, epoch_len, run_mode, mode: str):
    # === Aggregate metrics across iterations in the epoch ===
    metrics_names = metrics_per_epoch[0].keys()

    metrics_agg = {}
    # Iterate through each metric name provided in metrics_names
    for metric_name in metrics_names:
        all_values = [d[metric_name] for d in metrics_per_epoch]

        if metric_name != 'N_RMSE':
            all_values = torch.stack(all_values)
            average_metric = torch.mean(all_values)
        else:
            all_values = torch.cat(all_values)
            all_values = all_values.mean(dim=0)
            average_metric = all_values.mean()
        # Construct the key using the specified format and store the result in the dictionary
        key = f"{mode}/{run_mode}_{metric_name}"
        metrics_agg[key] = average_metric

    return metrics_agg, metrics_agg[f"{mode}/{run_mode}_loss"], metrics_agg[f"{mode}/{run_mode}_N_RMSE"]
