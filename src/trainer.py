"""
Module defining a trainer for a LLM on a given dataset.
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt

from dataloader.simple_dataloader import MGNDataset
from utils import get_available_device, get_trainable_parameters
from metrics import calc_n_rmse
from losses import CombinedLoss, RMSELoss
from models.model import MultivariateTimeLLM
from dataloader.mesh_utils import plot_patches


def get_data_loader(config, mode="train"):
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

    dl = DataLoader(ds,
                    batch_size=config['batch_size'],
                    num_workers=config['num_workers'],
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



        bs, _, channel, px, py = target.shape

        # Reshape targets to images and downsample
        target = target.view(bs, -1, 60, channel, px, py)
        target = target.view(-1, 60, 3 * 16 * 16).transpose(-1, -2)

        targ_img = F.fold(target, output_size=(240, 64), kernel_size=(16, 16), stride=(16, 16))
        targ_img = targ_img.view(bs, -1, 3, 240, 64)
        targ_img_red = targ_img[:, :, :, ::2, ::2]

        # print(model_out.shape)
        # plot_vals = model_out[0, 0, 0]
        # plt.imshow(plot_vals.detach().cpu().numpy().T)
        # plt.show()
        #
        # plot_targs = targ_img_red[0, 0, 0]
        # plt.imshow(plot_targs.cpu().T)
        # plt.show()
        # exit(9)

        bc_mask = torch.zeros_like(targ_img_red).bool()
        loss, all_losses = self.loss_fn.forward(preds=model_out, target=targ_img_red, mask=bc_mask)

        # # Find predicted next state and true next state
        # if self.params['fit_diffs']:
        #     true_state = states + target
        #     preds = states + model_out
        # else:
        #     true_state = states
        #     preds = model_out
        #
        # # Calculate metrics
        # with torch.no_grad():
        #     N_rmse = self.calculate_metrics(preds, true_state, bc_mask)

        # Log metrics
        log_metrics = {"loss": loss}
        log_metrics.update(all_losses)
        log_metrics['N_RMSE'] = 0.  # N_rmse

        return loss, log_metrics

    def run_gen_train_step(self, batch):
        """ No teacher forcing. Model makes predictions for a sequence, then tries to predict diffs given generated sequence.
            No grad when making predictions.
        """

        states, diffs, bc_mask, position_ids = batch
        bs, tot_patch, channel, px, py = states.shape
        seq_len = tot_patch // self.N_patch

        # 1) Model makes prediction of the sequence as guide
        self.model.eval()
        with torch.no_grad():
            guide_states, _ = self.model.gen_seq(batch, self.N_patch, pred_steps=seq_len - 1)

        # 2) Model tries to predict diffs between generated sequence and next step to true sequence
        # Reshape to be easier to work with
        f_states = states.view(bs, seq_len, self.N_patch, channel, px, py)
        f_guide_states = guide_states.view(bs, seq_len, self.N_patch, channel, px, py)
        # Difference to predict
        f_guide_error = f_states[:, 1:] - f_guide_states[:, :-1]
        guide_error = f_guide_error.view(bs, -1, channel, px, py)
        # Last guide state has no diff to predict anymore. Delete last state
        guide_states = f_guide_states[:, :-1].view(bs, -1, channel, px, py)
        bc_mask = bc_mask[:, :-self.N_patch]
        position_ids = position_ids[:, :-self.N_patch]

        # Forward pass like normal
        self.model.train()
        guide_batch = (guide_states, guide_error, bc_mask, position_ids)
        loss, log_metrics = self.run_train_step(guide_batch)

        return loss, log_metrics

    @torch.no_grad()
    def run_gen_val_step(self, batch):
        """ Like above, but use model.eval()
        """
        self.model.eval()

        states, diffs, bc_mask, position_ids = batch
        bs, tot_patch, channel, px, py = states.shape
        seq_len = tot_patch // self.N_patch

        # 1) Model makes prediction of the sequence as guide
        guide_states, _ = self.model.gen_seq(batch, self.N_patch, pred_steps=seq_len - 1)

        # 2) Model tries to predict diffs between generated sequence and next step to true sequence
        # Reshape to be easier to work with
        f_states = states.view(bs, seq_len, self.N_patch, channel, px, py)
        f_guide_states = guide_states.view(bs, seq_len, self.N_patch, channel, px, py)
        # Difference to predict
        f_guide_error = f_states[:, 1:] - f_guide_states[:, :-1]
        guide_error = f_guide_error.view(bs, -1, channel, px, py)
        # Last guide state has no diff to predict anymore. Delete last state
        guide_states = f_guide_states[:, :-1].view(bs, -1, channel, px, py)
        bc_mask = bc_mask[:, :-self.N_patch]
        position_ids = position_ids[:, :-self.N_patch]

        # Forward pass like normal
        guide_batch = (guide_states, guide_error, bc_mask, position_ids)
        loss, log_metrics = self.run_train_step(guide_batch)

        # Rename losses
        log_metrics = {f'{k}': v for k, v in log_metrics.items()}
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
