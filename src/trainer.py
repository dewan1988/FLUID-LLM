"""
Module defining a trainer for a LLM on a given dataset.
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataloader.simple_dataloader import MGNDataset
from dataloader.synth_dl import SynthDS
from utils import get_available_device, get_trainable_parameters
from metrics import calc_n_rmse
from losses import CombinedLoss, RMSELoss
from models.model import MultivariateTimeLLM
from dataloader.mesh_utils import plot_patches


def get_data_loader(config, mode="train"):
    if config['task_name'] == "cylinder_task":
        ds = MGNDataset(load_dir=f'{config["load_dir"]}/{mode}',
                        resolution=config['resolution'],
                        patch_size=config['patch_size'],
                        stride=config['stride'],
                        seq_len=config['seq_len'],
                        seq_interval=config['seq_interval'],
                        mode=mode,
                        fit_diffs=config['fit_diffs'],
                        normalize=config['normalize_ds']
                        )
    elif config['task_name'] == "synthetic_task":
        ds = SynthDS(resolution=config['resolution'],
                     patch_size=config['patch_size'],
                     stride=config['stride'],
                     seq_len=config['seq_len'],
                     seq_interval=config['seq_interval'],
                     mode=mode,
                     fit_diffs=config['fit_diffs'],
                     normalize=config['normalize_ds']
                     )
    else:
        raise ValueError(f"Unknown task name: {config['task_name']}")

    dl = DataLoader(ds,
                    batch_size=config['batch_size'],
                    num_workers=config['num_workers'] if config['task_name'] != "synthetic_task" else 1,
                    prefetch_factor=2,
                    pin_memory=True)
    return dl


class Trainer:
    def __init__(self, params, model: MultivariateTimeLLM, N_patch):
        """
        params (dict): A dict with the configuration parameters (e.g., learning rate, optimizer, etc.)
        """
        super().__init__()

        self.params = params
        self.model = model
        self.N_patch = N_patch
        self.loss_fn = CombinedLoss(params['loss_function'], params['loss_weighting'])

    def calculate_metrics(self, preds: torch.Tensor, target: torch.Tensor, bc_mask: torch.Tensor):
        """ input.shape = (bs, seq_len*N_patch, 3, pat)
        """
        BS, _, channel, px, py = preds.shape

        preds = preds.view(BS, -1, self.N_patch, channel, px, py)
        target = target.view(BS, -1, self.N_patch, channel, px, py)
        bc_mask = bc_mask.view(BS, -1, self.N_patch, channel, px, py)

        N_rmse = calc_n_rmse(preds, target, bc_mask)

        return N_rmse  # .item()

    def run_train_step(self, batch):
        """
        Returns
        - loss (torch.Tensor): The total loss, used for backpropagation
        - metrics_to_log (dict): A dictionary with the calculated metrics (detached from the computational graph)
        """
        states, target, bc_mask, position_ids = batch
        # If fitting diffs, target is diffs. Otherwise, target is next state
        self.model.train()

        # Forward pass
        if self.params['see_init_state']:
            model_out = self.model.forward_duplicate(states, position_ids, self.N_patch)
        else:
            _, model_out = self.model(states, position_ids)
        loss, all_losses = self.loss_fn(preds=model_out, target=target, mask=bc_mask)

        # Find predicted next state and true next state
        if self.params['fit_diffs']:
            true_state = states + target
            preds = states + model_out
        else:
            true_state = states
            preds = model_out

        # Calculate metrics
        with torch.no_grad():
            N_rmse = self.calculate_metrics(preds, true_state, bc_mask)

        # Log metrics
        log_metrics = {"loss": loss}
        log_metrics.update(all_losses)
        log_metrics['N_RMSE'] = N_rmse

        return loss, log_metrics

    def run_gen_train_step(self, batch):
        """
        No teacher forcing version.
        """
        self.model.train()

        states, target, bc_mask, position_ids = batch
        bs, tot_patch, channel, px, py = states.shape
        seq_len = tot_patch // self.N_patch

        _, model_out = self.model.gen_seq(batch, self.N_patch, pred_steps=seq_len)
        loss, all_losses = self.loss_fn(preds=model_out, target=target, mask=bc_mask)

        # Calculate loss
        if self.params['fit_diffs']:
            true_state = states + target
            preds = states + model_out
        else:
            true_state = states
            preds = model_out

        # Calculate metrics
        with torch.no_grad():
            BS, _, channel, px, py = preds.shape

            N_rmse = calc_n_rmse(preds.view(BS, -1, self.N_patch, channel, px, py),
                                 true_state.view(BS, -1, self.N_patch, channel, px, py),
                                 bc_mask.view(BS, -1, self.N_patch, channel, px, py))

        # Log metrics
        log_metrics = {"loss": loss}
        log_metrics.update(all_losses)
        log_metrics['N_RMSE'] = N_rmse

        return loss, log_metrics

    @torch.no_grad()
    def run_gen_val_step(self, batch):
        """
        No teacher forcing version with no grad.
        """
        states, target, bc_mask, position_ids = batch
        bs, tot_patch, channel, px, py = states.shape
        seq_len = tot_patch // self.N_patch

        _, model_out = self.model.gen_seq(batch, self.N_patch, pred_steps=seq_len)
        loss, all_losses = self.loss_fn(preds=model_out, target=target, mask=bc_mask)

        # Calculate loss
        if self.params['fit_diffs']:
            true_state = states + target
            preds = states + model_out
        else:
            true_state = states
            preds = model_out

        # Calculate metrics
        with torch.no_grad():
            BS, _, channel, px, py = preds.shape

            N_rmse = calc_n_rmse(preds.view(BS, -1, self.N_patch, channel, px, py),
                                 true_state.view(BS, -1, self.N_patch, channel, px, py),
                                 bc_mask.view(BS, -1, self.N_patch, channel, px, py))

        # Log metrics
        log_metrics = {"loss": loss}
        log_metrics.update(all_losses)
        log_metrics['N_RMSE'] = N_rmse

        return loss, log_metrics

    def prepare_optimizers(self):
        params = self.model.parameters()
        print(f"The model has {get_trainable_parameters(self.model)} trainable parameters")

        optimizer_type = self.params['optimizer']
        self.params['learning_rate'] = float(self.params['learning_rate'])
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(list(params),
                                          lr=self.params['learning_rate'],
                                          weight_decay=self.params['weight_decay'])
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(list(params),
                                         lr=self.params['learning_rate'],
                                         weight_decay=self.params['weight_decay'])
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(list(params),
                                        lr=self.params['learning_rate'],
                                        weight_decay=self.params['weight_decay'])
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=self.params['schedule_epoch'],
                                                    gamma=self.params['schedule_gamma'])

        return optimizer, scheduler
