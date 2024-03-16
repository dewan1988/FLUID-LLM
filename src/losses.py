"""
Module defining the losses used for training.
"""
import torch
import torch as t
import torch.nn as nn


def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float("inf")] = 0.0
    return div


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
        loss = self.mse_loss(preds, target)
        loss = (loss * mask.float()).sum()

        non_zero_elements = mask.sum()
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val


class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param preds: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        # Ensure the mask is boolean for indexing
        mask = mask.bool()

        # Apply mask to input and target
        input_masked = torch.masked_select(preds, mask)
        target_masked = torch.masked_select(target, mask)

        # Avoid division by zero and ensure nonzero target values
        target_masked = torch.where(target_masked == 0, torch.tensor(1e-8, device=target_masked.device),
                                    target_masked)

        # Calculate MAPE
        loss = torch.abs((input_masked - target_masked) / target_masked)
        return torch.mean(loss)


class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param preds: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        if mask is None:
            mask = t.ones(preds.size())
        delta_y = t.abs((target - preds))
        scale = t.abs(target) + t.abs(preds)
        smape = _divide_no_nan(delta_y, scale)
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
        mae = t.abs(target - preds) * mask
        mae = t.mean(mae)
        return mae
