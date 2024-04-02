"""
Module defining the losses used for training.
"""
import torch
import torch as t
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        Mean Squared Error loss.

        :param preds: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        mask = ~mask
        loss = self.loss_fn(target * mask, preds * mask)
        return loss

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

    def __repr__(self):
        return "MAE"


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
        pressure_preds = preds[:, :, 2, :]
        pressure_target = target[:, :, 2, :]
        pressure_mask = mask[:, :, 2, :]
        velocity_preds = preds[:, :, 0, :]
        velocity_target = target[:, :, 0, :]
        velocity_mask = mask[:, :, 0, :]

        tot_loss = 0
        all_losses = {}
        for loss_fn, weighting in zip(self.loss_fns, self.weighting):
            loss_pressure = loss_fn(pressure_preds, pressure_target, pressure_mask)
            loss_velocity = loss_fn(velocity_preds, velocity_target, velocity_mask)

            # Eagle weights pressure by alpha 0.1
            eagle_alpha = 0.1 # 0.1
            loss_val = (loss_velocity * weighting) + (loss_pressure * weighting * eagle_alpha)

            tot_loss += loss_val

            all_losses[str(loss_fn)] = loss_val

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
