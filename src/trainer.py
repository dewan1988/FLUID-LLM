"""
Module defining a trainer for a LLM on a given dataset.
"""

import torch
from torch.utils.data import DataLoader
from dataloader.simple_dataloader import MGNDataloader
from utils import get_available_device, get_trainable_parameters
from losses import CombinedLoss
from models.model import MultivariateTimeLLM


def get_data_loader(config):
    ds = MGNDataloader(load_dir=config['load_dir'],
                       resolution=config['resolution'],
                       patch_size=config['patch_size'],
                       stride=config['stride'],
                       seq_len=config['seq_len'],
                       seq_interval=config['seq_interval'],
                       step_per_ep=config['epoch_size'] * config['batch_size']
                       )

    dl = DataLoader(ds, batch_size=config['batch_size'], num_workers=config['num_workers'], prefetch_factor=2, pin_memory=True)
    return dl


class Trainer:
    def __init__(self, params, model: MultivariateTimeLLM, device):
        """
        params (dict): A dict with the configuration parameters (e.g., learning rate, optimizer, etc.)
        """
        super().__init__()

        self.params = params
        self.model = model
        self.loss_fn = CombinedLoss(params['loss_function'], params['loss_weighting'])

        self.device = device

    def calculate_loss(self, preds: torch.Tensor, diffs: torch.Tensor, bc_mask: torch.Tensor):
        loss, all_losses = self.loss_fn(preds=preds, target=diffs, mask=bc_mask)
        return loss, all_losses

    def prepare_optimizers(self):
        params = self.model.parameters()
        print(f"The model has {get_trainable_parameters(self.model)} trainable parameters")

        optimizer_type = self.params['optimizer']
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

    def run_train_step(self, states, diffs, bc_mask, position_ids):
        """
        Returns
        - loss (torch.Tensor): The total loss, used for backpropagation
        - metrics_to_log (dict): A dictionary with the calculated metrics (detached from the computational graph)
        """
        self.model.train()

        # Rescale diffs
        diffs = diffs * self.params['diff_scale_factor']

        # Forward pass
        backbone_out, preds = self.model(states, position_ids)

        # Calculate loss
        loss, all_losses = self.calculate_loss(preds, diffs, bc_mask)

        # Calculate metrics
        log_metrics = {"train_loss": loss.detach().item()}
        log_metrics.update(all_losses)

        return loss, log_metrics

    @torch.no_grad()
    def run_eval_step(self, batch):
        self.model.eval()

        states, diffs, bc_mask, position_ids = batch

        # Forward pass
        backbone_out, preds = self.model(states, position_ids)

        # Calculate loss
        loss = self.calculate_loss(preds, diffs, bc_mask)

        # Calculate metrics
        log_metrics = {"eval_loss": loss["loss"].detach().item()}

        return log_metrics
