"""
Module defining the losses used for training.
"""
import torch
import torch as t
import torch.nn as nn
from torch.nn import Parameter as P

class MAPELoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(MAPELoss, self).__init__()

        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param preds: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        # Invert mask so 1 is wanted pixel
        mask = ~mask

        # Apply mask to input and target
        input_masked = torch.masked_select(preds, mask)
        target_masked = torch.masked_select(target, mask)

        # Clamp loss to avoid division by zero
        target_abs = torch.abs(target_masked).clamp(min=self.eps)

        # Calculate MAPE
        loss = torch.abs((input_masked - target_masked) / target_abs)

        loss = loss.clamp(max=1.)
        return torch.mean(loss)


class SMAPELoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(SMAPELoss, self).__init__()

        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param preds: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        # Invert mask so 1 is wanted pixel
        mask = ~mask

        delta_y = t.abs((target - preds))
        scale = t.abs(target) + t.abs(preds) + self.eps

        smape = delta_y / scale
        smape = smape * mask
        smape = 2 * t.mean(smape)
        return smape


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        Mean Squared Error loss.

        :param preds: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        # Invert mask so 1 is wanted pixel
        mask = ~mask

        # Apply mask to input and target
        input_masked = torch.masked_select(preds, mask)
        target_masked = torch.masked_select(target, mask)

        loss = self.loss_fn(input_masked, target_masked)
        loss = loss.sum()

        non_zero_elements = mask.sum()
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val

    def __repr__(self):
        return "MSE"


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        Root Mean Squared Error loss.

        :param preds: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        # Invert mask so 1 is wanted pixel
        mask = ~mask

        loss = self.loss_fn(target * mask, preds * mask)

        # Calculate RMSE
        loss = torch.sqrt(loss)
        return loss

    def __repr__(self):
        return "RMSE"


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        MAE Loss

        Calculates Mean Absolute Error between
        y and y_hat. MAE measures the relative prediction
        accuracy of a forecasting method by calculating the
        deviation of the prediction and the true
        value at a given time and averages these devations
        over the length of the series.
        """
        # Invert mask so 1 is wanted pixel
        mask = ~mask
        # Apply mask to input and target
        input_masked = torch.masked_select(preds, mask)
        target_masked = torch.masked_select(target, mask)

        mae = t.abs(input_masked - target_masked)
        mae = t.sum(mae)

        non_zero_elements = mask.sum()
        mse_loss_val = mae / non_zero_elements
        return mse_loss_val

    def __repr__(self):
        return "MAE"


class CombinedLoss(nn.Module):
    def __init__(self, loss_fns, loss_weight: list, pressure_weight: float = 1.):
        super(CombinedLoss, self).__init__()
        self.loss_fns = nn.ModuleList([self.get_loss_fn(loss_fn) for loss_fn in loss_fns])
        self.loss_weight = loss_weight
        self.pressure_weight = pressure_weight

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        Combined loss function.

        :param preds: Forecast values. Shape:  (bs, seq_len, 3, tot_px, tot_py)
        :param target: Target values. Shape:  (bs, seq_len, 3, tot_px, tot_py)
        :param mask: 0/1 mask. Shape:  (bs, seq_len, 3, tot_px, tot_py)
        :return: Loss value
        """

        pressure_preds = preds[:, :, 2:, :]  # shape = (bs, seq_len*N_patch, 1, 16, 16)
        pressure_target = target[:, :, 2:, :]
        pressure_mask = mask[:, :, 0:, :]
        velocity_preds = preds[:, :, :2, :]  # shape = (bs, seq_len*N_patch, 2, 16, 16)
        velocity_target = target[:, :, :2, :]
        velocity_mask = mask[:, :, :2, :]

        tot_loss = 0
        all_losses = {}
        for loss_fn, weighting in zip(self.loss_fns, self.loss_weight):
            loss_pressure = loss_fn(pressure_preds, pressure_target, pressure_mask)
            loss_velocity = loss_fn(velocity_preds, velocity_target, velocity_mask)

            loss_val = loss_velocity + self.pressure_weight * loss_pressure
            tot_loss += loss_val * weighting

            # loss_val = loss_fn.forward(preds, target, mask)
            # print(loss_val)
            #
            # exit(5)

            all_losses[str(loss_fn)] = loss_val  # .item()

        return tot_loss, all_losses

    def get_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            return MSELoss()
        elif loss_fn == "rmse":
            return RMSELoss()
        elif loss_fn == "mae":
            return MAELoss()
        elif loss_fn == "mape":
            return MAPELoss()
        elif loss_fn == "smape":
            return SMAPELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
