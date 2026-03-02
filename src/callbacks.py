"""Training callbacks for EnsemFormer.

This module defines a small callback system with lifecycle hooks for the
training loop and common utilities such as early stopping, checkpointing, and
metric aggregation.
"""

import logging
from typing import Literal
from abc import ABC
from torchmetrics.regression import (
    R2Score,
    MeanSquaredError,
    MeanAbsoluteError,
    PearsonCorrCoef,
)

from src.loggers import Logger


class BaseCallback(ABC):
    """Abstract base class for training callbacks."""

    def on_fit_start(self, *args, **kwargs):
        pass

    def on_fit_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_batch_start(self, *args, **kwargs):
        pass

    def on_validation_step(self, *args, **kwargs):
        pass

    def on_validation_end(self, *args, **kwargs):
        pass

    def on_checkpoint_load(self, *args, **kwargs):
        pass

    def on_checkpoint_save(self, *args, **kwargs):
        pass


class EarlyStoppingCallback(BaseCallback):
    """Simple early stopping on a scalar metric.

    Parameters
    ----------
    patience : int
        Number of validations without improvement to tolerate before stopping.
    delta : float
        Minimum change in the monitored metric to qualify as an improvement.
    direction : {"min", "max"}
        Whether a lower (``"min"``) or higher (``"max"``) value is considered
        better.
    """

    def __init__(
        self,
        patience: int,
        delta: float,
        direction: Literal["min", "max"],
    ):
        self.counter = 0
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.best_metric = None
        self.direction = direction

    def on_validation_end(self, epoch_idx, metric, **kwargs):
        """Update internal state and set ``early_stop`` when appropriate."""
        if self.direction == "min":
            if self.best_metric is None:
                self.best_metric = metric
            elif metric < self.best_metric - self.delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            if self.best_metric is None:
                self.best_metric = metric
            elif metric > self.best_metric + self.delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True


class AllMetricsCallback(BaseCallback):
    """Accumulate and log regression metrics (MAE, RMSE, R², Pearson r).

    Parameters
    ----------
    logger : Logger
        Logger instance used to record metrics at the end of validation.
    rescale_factor : float, default=1
        Post-multiplicative factor applied to MAE and RMSE to undo any
        normalization performed by a datamodule.
    prefix : {"valid", "test"}, default="valid"
        Prefix added to metric names when logging.
    """

    def __init__(
        self,
        logger: Logger,
        rescale_factor: float = 1,
        prefix: Literal["valid", "test"] = "valid",
    ):
        self.mae = MeanAbsoluteError()
        self.rmse = MeanSquaredError(squared=False)
        self.r2 = R2Score()
        self.pearson = PearsonCorrCoef()

        self.logger = logger
        self.rescale_factor = float(rescale_factor)
        self.prefix = prefix
        self.best_mae = float("inf")
        self.last_mae = None
        self.best_rmse = float("inf")
        self.last_rmse = None
        self.best_r2 = float("-inf")
        self.last_r2 = None
        self.best_pearson = float("-inf")
        self.last_pearson = None

    def on_validation_step(self, input, target, pred):
        """Update metric accumulators for the current batch."""
        pred_flat, target_flat = (
            pred.detach().view(-1).float().cpu(),
            target.detach().view(-1).float().cpu(),
        )
        self.mae(pred_flat, target_flat)
        self.rmse(pred_flat, target_flat)
        self.r2(pred_flat, target_flat)
        self.pearson(pred_flat, target_flat)

    def on_validation_end(self, epoch=None, **kwargs):
        """Compute epoch-level metrics and log them."""
        mae = float(self.mae.compute()) * self.rescale_factor
        rmse = float(self.rmse.compute()) * self.rescale_factor
        r2 = float(self.r2.compute())
        pearson = float(self.pearson.compute())

        logging.info(
            f"\n"
            f"            {self.prefix} MAE: {mae:.4f}\n"
            f"            {self.prefix} RMSE: {rmse:.4f}\n"
            f"            {self.prefix} R²: {r2:.4f}\n"
            f"            {self.prefix} Pearson r: {pearson:.4f}"
        )
        self.logger.log_metrics(
            {
                f"{self.prefix} MAE": mae,
                f"{self.prefix} RMSE": rmse,
                f"{self.prefix} R²": r2,
                f"{self.prefix} Pearson r": pearson,
            },
            epoch,
        )
        self.best_mae = min(self.best_mae, mae)
        self.last_mae = mae
        self.best_rmse = min(self.best_rmse, rmse)
        self.last_rmse = rmse
        self.best_r2 = max(self.best_r2, r2)
        self.last_r2 = r2
        self.best_pearson = max(self.best_pearson, pearson)
        self.last_pearson = pearson

        self.mae.reset()
        self.rmse.reset()
        self.r2.reset()
        self.pearson.reset()

    def on_fit_end(self):
        """Log best metrics observed over validation/test runs."""
        if self.best_mae != float("inf"):
            self.logger.log_metrics({f"{self.prefix} best MAE": self.best_mae})
            self.logger.log_metrics(
                {f"{self.prefix} final loss": self.last_mae / self.rescale_factor}
            )
        if self.best_rmse != float("inf"):
            self.logger.log_metrics({f"{self.prefix} best RMSE": self.best_rmse})
        if self.best_r2 != float("-inf"):
            self.logger.log_metrics({f"{self.prefix} best R²": self.best_r2})
        if self.best_pearson != float("-inf"):
            self.logger.log_metrics(
                {f"{self.prefix} best Pearson r": self.best_pearson}
            )
