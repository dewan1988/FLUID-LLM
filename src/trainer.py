"""
Module defining a trainer for a LLM on a given dataset.
"""

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from utils import get_available_device, get_trainable_parameters
from utils_model import calc_n_rmse, patch_to_img, normalise_diffs, normalise_states, img_to_patch
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
        self.norm_channel_independent = params['channel_independent']
    #
    # def run_train_step(self, batch):
    #     """
    #     Returns
    #     - loss (torch.Tensor): The total loss, used for backpropagation
    #     - metrics_to_log (dict): A dictionary with the calculated metrics (detached from the computational graph)
    #     """
    #     states, _, diffs, bc_mask, position_ids = batch
    #
    #     # If fitting diffs, target is diffs. Otherwise, target is next state
    #     self.model.train()
    #     # Forward pass
    #     if self.params['see_init_state']:
    #         pred_diff = self.model.forward_see_init(states, position_ids)
    #     else:
    #         pred_diff = self.model(states, position_ids)
    #
    #     # Reshape targets to images and downsample
    #     diffs = patch_to_img(diffs, self.ds_props)
    #     bc_mask = patch_to_img(bc_mask.float(), self.ds_props).bool()
    #
    #     # Normalise predictions so loss is well scaled
    #     if self.loss_norm_eps is not None:
    #         norm_true_diffs, norm_pred_diff = normalise_diffs(diffs, pred_diff, self.loss_norm_eps, self.norm_channel_independent)
    #         loss, all_losses = self.loss_fn.forward(preds=norm_pred_diff, target=norm_true_diffs, mask=bc_mask)
    #     else:
    #         loss, all_losses = self.loss_fn.forward(preds=pred_diff, target=diffs, mask=bc_mask)
    #
    #     # Calculate metrics
    #     with torch.no_grad():
    #         N_rmse = calc_n_rmse(pred_diff, diffs, bc_mask)
    #
    #     # Log metrics
    #     all_losses["loss"] = loss
    #     all_losses['N_RMSE'] = N_rmse
    #
    #     return loss, all_losses

    def run_train_step(self, batch):
        """
        Returns
        - loss (torch.Tensor): The total loss, used for backpropagation
        - metrics_to_log (dict): A dictionary with the calculated metrics (detached from the computational graph)
        """
        states, next_state, diffs, bc_mask, position_ids = batch

        # If fitting diffs, target is diffs. Otherwise, target is next state
        self.model.train()
        # Forward pass
        if self.params['noise'] is not None:
            noise = torch.randn_like(states) * (~bc_mask) * diffs.std(dim=(-1, -2, -3, -4, -5), keepdim=True) * self.params['noise']
            input_states = states + noise
        else:
            input_states = states

        if self.params['see_init_state']:
            pred_diff = self.model.forward_see_init(input_states, position_ids)
        else:
            pred_diff = self.model(input_states, position_ids)

        # print(f'{noisy_states.shape = }, {pred_diff.shape = }, {diffs.shape = }')
        input_states = patch_to_img(input_states, self.ds_props)
        pred_state = input_states + pred_diff
        next_state = patch_to_img(next_state, self.ds_props)
        bc_mask = patch_to_img(bc_mask.float(), self.ds_props).bool()

        # Normalise predictions so loss is well scaled
        if self.loss_norm_eps is not None:
            norm_next_state, norm_pred_states = normalise_states(diffs, next_state, pred_state, self.loss_norm_eps, self.norm_channel_independent)
            loss, all_losses = self.loss_fn.forward(preds=norm_pred_states, target=norm_next_state, mask=bc_mask)
        else:
            loss, all_losses = self.loss_fn.forward(preds=pred_diff, target=diffs, mask=bc_mask)

        # Calculate metrics
        with torch.no_grad():
            N_rmse = calc_n_rmse(pred_state, next_state, bc_mask)

        # Log metrics
        all_losses["loss"] = loss
        all_losses['N_RMSE'] = N_rmse

        return loss, all_losses

    def run_gen_train_step(self, batch):
        """ No teacher forcing. Model makes predictions for a sequence, then tries to predict diffs given generated sequence.
            No grad when making predictions.
        """

        states, next_state, diffs, bc_mask, position_ids = batch
        bs, seq_len, N_patch, channel, px, py = states.shape

        self.model.eval()
        with torch.no_grad():
            # 1) Model makes prediction of the sequence as guide
            guide_states, _ = self.model.gen_seq(batch, seq_len - 1)
            guide_states = guide_states[:, :-1].contiguous()  # Predictions for current states.
            guide_states_patch = img_to_patch(guide_states, self.ds_props)

        # 2) Model tries to predict true state based on guide_state
        self.model.train()
        pred_diffs = self.model.forward_see_init(guide_states_patch, position_ids)  # Predictions for next states
        pred_states = guide_states + pred_diffs

        # 3) Calculate loss based on pred_state and true_state
        next_state = patch_to_img(next_state, self.ds_props)
        bc_mask = patch_to_img(bc_mask.float(), self.ds_props).bool()

        if self.loss_norm_eps is not None:
            norm_next_state, norm_pred_states = normalise_states(diffs, next_state, pred_states, self.loss_norm_eps, self.norm_channel_independent)
            loss, all_losses = self.loss_fn(preds=norm_pred_states, target=norm_next_state, mask=bc_mask)
        else:
            loss, all_losses = self.loss_fn(preds=pred_states, target=next_state, mask=bc_mask)

        # Calculate metrics
        with torch.no_grad():
            N_rmse = calc_n_rmse(pred_states, next_state, bc_mask)

        # Log metrics
        all_losses["loss"] = loss
        all_losses['N_RMSE'] = N_rmse

        return loss, all_losses

    def run_notf_train_step(self, batch):
        """
        No teacher forcing version.
        """
        self.model.train()

        states, next_state, diffs, bc_mask, position_ids = batch
        bs, seq_len, N_patch, channel, px, py = states.shape

        # Model predictions
        pred_states, _ = self.model.gen_seq(batch, seq_len - 1)
        pred_states = pred_states[:, 1:]

        # Reshape targets + mask
        next_state = patch_to_img(next_state, self.ds_props)
        bc_mask = patch_to_img(bc_mask.float(), self.ds_props).bool()

        if self.loss_norm_eps is not None:
            norm_next_state, norm_pred_states = normalise_states(diffs, next_state, pred_states, self.loss_norm_eps, self.norm_channel_independent)
            loss, all_losses = self.loss_fn(preds=norm_pred_states, target=norm_next_state, mask=bc_mask)
        else:
            loss, all_losses = self.loss_fn(preds=pred_states, target=next_state, mask=bc_mask)

        # Calculate metrics
        with torch.no_grad():
            N_rmse = calc_n_rmse(pred_states, next_state, bc_mask)

        # Log metrics
        all_losses["loss"] = loss
        all_losses['N_RMSE'] = N_rmse

        return loss, all_losses

    @torch.inference_mode()
    def run_val_step(self, batch):
        """ Make predictions.
        """
        self.model.eval()

        states, _, _, bc_mask, position_ids = batch

        bs, seq_len, N_patch, channel, px, py = states.shape
        pred_states, pred_diffs = self.model.gen_seq(batch, pred_steps=seq_len - 1)
        pred_states = pred_states[:, :-1]

        states_img = patch_to_img(states, self.ds_props)
        bc_mask = patch_to_img(bc_mask.float(), self.ds_props).bool()

        # Calculate metrics
        loss, all_losses = self.loss_fn.forward(preds=pred_states, target=states_img, mask=bc_mask)
        N_rmse = calc_n_rmse(pred_states, states_img, bc_mask)

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
