"""
Module defining a trainer for a LLM on a given dataset.
"""

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from utils import get_available_device, get_trainable_parameters
from utils_model import calc_n_rmse, patch_to_img, normalise_diffs
from losses import CombinedLoss
from models.model import MultivariateTimeLLM
from dataloader.ds_props import DSProps


class Trainer:
    def __init__(self, params, model: MultivariateTimeLLM, ds_props: DSProps):
        """
        params (dict): A dict with the configuration parameters (e.g., learning rate, optimizer, etc.)
        """
        super().__init__()

        self.params = params
        self.model = model
        self.ds_props = ds_props

        self.N_patch = ds_props.N_patch
        self.loss_fn = CombinedLoss(params['loss_function'], params['loss_weighting'], params['pressure_weight'])

        self.loss_norm_eps = params['loss_norm_eps']
        if self.loss_norm_eps is not None:
            self.loss_norm_eps = torch.nn.Parameter(torch.tensor(self.loss_norm_eps, device='cuda'), requires_grad=False)

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
            preds = self.model.forward_duplicate(states, position_ids)
        else:
            preds = self.model(states, position_ids)

        # Reshape targets to images and downsample
        targs = patch_to_img(target, self.ds_props)
        bc_mask = patch_to_img(bc_mask.float(), self.ds_props).bool()

        # Normalise predictions so loss is well scaled
        if self.loss_norm_eps is not None:
            targs, preds = normalise_diffs(targs, preds, self.loss_norm_eps)

        loss, all_losses = self.loss_fn.forward(preds=preds, target=targs, mask=bc_mask)

        # Calculate metrics
        with torch.no_grad():
            N_rmse = calc_n_rmse(preds, targs, bc_mask).mean()  # self.calculate_metrics(model_out, targ_imgs, bc_mask)

        # Log metrics
        all_losses["loss"] = loss
        all_losses['N_RMSE'] = N_rmse

        return loss, all_losses

    # def run_gen_train_step(self, batch):
    #     """ No teacher forcing. Model makes predictions for a sequence, then tries to predict diffs given generated sequence.
    #         No grad when making predictions.
    #     """
    #
    #     states, diffs, bc_mask, position_ids = batch
    #     bs, tot_patch, channel, px, py = states.shape
    #     seq_len = tot_patch // self.N_patch
    #
    #     # 1) Model makes prediction of the sequence as guide
    #     self.model.eval()
    #     with torch.no_grad():
    #         guide_states, _ = self.model.gen_seq(batch, self.N_patch, pred_steps=seq_len - 1)
    #
    #     # 2) Model tries to predict diffs between generated sequence and next step to true sequence
    #     # Reshape to be easier to work with
    #     f_states = states.view(bs, seq_len, self.N_patch, channel, px, py)
    #     f_guide_states = guide_states.view(bs, seq_len, self.N_patch, channel, px, py)
    #     # Difference to predict
    #     f_guide_error = f_states[:, 1:] - f_guide_states[:, :-1]
    #     guide_error = f_guide_error.view(bs, -1, channel, px, py)
    #     # Last guide state has no diff to predict anymore. Delete last state
    #     guide_states = f_guide_states[:, :-1].view(bs, -1, channel, px, py)
    #     bc_mask = bc_mask[:, :-self.N_patch]
    #     position_ids = position_ids[:, :-self.N_patch]
    #
    #     # Forward pass like normal
    #     self.model.train()
    #     guide_batch = (guide_states, guide_error, bc_mask, position_ids)
    #     loss, log_metrics = self.run_train_step(guide_batch)
    #
    #     return loss, log_metrics

    @torch.inference_mode()
    def run_val_step(self, batch):
        """ Make predictions.
        """
        self.model.eval()

        states, target, bc_mask, position_ids = batch

        bs, seq_len, N_patch, channel, px, py = states.shape
        _, pred_diffs = self.model.gen_seq(batch, pred_steps=seq_len - 1)

        targ_imgs = patch_to_img(target, self.ds_props)
        bc_mask = patch_to_img(bc_mask.float(), self.ds_props).bool()

        # Calculate metrics
        loss, all_losses = self.loss_fn.forward(preds=pred_diffs, target=targ_imgs, mask=bc_mask)
        N_rmse = calc_n_rmse(pred_diffs, targ_imgs, bc_mask).mean()

        # Log metrics
        all_losses["loss"] = loss
        all_losses['N_RMSE'] = N_rmse

        return all_losses

    def prepare_optimizers(self):
        params = self.model.parameters()
        print(f"The model has {get_trainable_parameters(self.model)} trainable parameters")

        optimizer_type = self.params['optimizer']
        self.params['learning_rate'] = float(self.params['learning_rate'])
        if optimizer_type == "adamw":
            print(self.params['weight_decay'])
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
