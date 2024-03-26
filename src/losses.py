"""
Module defining the losses used for training.
"""
import torch
import torch as t
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

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

        loss = self.mse_loss(input_masked, target_masked)
        loss = loss.sum()

        non_zero_elements = mask.sum()
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

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

        # Apply mask to input and target
        input_masked = torch.masked_select(preds, mask)
        target_masked = torch.masked_select(target, mask)

        # Calculate RMSE
        loss = torch.sqrt(torch.mean((input_masked - target_masked) ** 2))
        return loss


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


class CombinedLoss(nn.Module):
    def __init__(self, loss_fns, weighting):
        super(CombinedLoss, self).__init__()
        self.loss_fns = nn.ModuleList([self.get_loss_fn(loss_fn) for loss_fn in loss_fns])
        self.weighting = weighting

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        Combined loss function.

        :param preds: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        loss = 0
        for loss_fn, weighting in zip(self.loss_fns, self.weighting):
            loss += loss_fn(preds, target, mask) * weighting

        return loss

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
