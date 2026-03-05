"""Training utilities: checkpointing and Trainer loop for EnsemFormer."""

import logging
import pathlib
import types

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from src.module import Module
from src.callbacks import BaseCallback
from src.loggers import Logger
from src.mol_loader import MolLoader


def save_state(
    modelmodule: Module,
    epoch: int,
    path: pathlib.Path,
    callbacks: dict[str, BaseCallback],
):
    """Save model, optimizer, and epoch state to a checkpoint file.

    Only rank 0 performs the write in a distributed setting.
    """
    if modelmodule.local_rank == 0:
        # Unwrap DDP if necessary
        model = modelmodule.model
        if isinstance(model, DistributedDataParallel):
            state_dict = model.module.state_dict()
        elif isinstance(model, torch.nn.ModuleList):
            # CycloFormerModule stores sub-modules as a ModuleList
            state_dict = model.state_dict()
        else:
            state_dict = model.state_dict()

        checkpoint = {
            "state_dict": state_dict,
            "optimizer_state_dict": modelmodule.optimizer.state_dict(),
            "epoch": epoch,
        }
        for callback in callbacks.values():
            callback.on_checkpoint_save(checkpoint)

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(path))
        logging.info(f"Saved checkpoint to {path}")


def load_state(
    modelmodule: Module,
    path: pathlib.Path,
    callbacks: dict[str, BaseCallback],
) -> int:
    """Load model, optimizer, and epoch state from a checkpoint file.

    Returns
    -------
    int
        The epoch index stored in the checkpoint.
    """
    checkpoint = torch.load(
        str(path), map_location={"cuda:0": f"cuda:{modelmodule.local_rank}"}
    )
    model = modelmodule.model
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])
    modelmodule.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for callback in callbacks.values():
        callback.on_checkpoint_load(checkpoint)

    logging.info(f"Loaded checkpoint from {path}")
    return checkpoint.get("epoch", 0)


class Trainer:
    """Minimal training loop coordinator for EnsemFormer.

    Parameters
    ----------
    world_size : int
        Number of processes for distributed training (1 for single-GPU).
    log_dir : pathlib.Path
        Directory where checkpoints are saved.
    """

    def __init__(self, world_size: int, log_dir: pathlib.Path):
        self.world_size = world_size
        self.log_dir = pathlib.Path(log_dir)

    def fit(
        self,
        module: Module,
        datamodule: MolLoader,
        max_epochs: int,
        callbacks: dict[str, BaseCallback],
        logger: Logger,
        config: dict,
    ):
        """Run the training loop with periodic validation and checkpointing.

        Parameters
        ----------
        module : Module
            The model wrapper (e.g. CycloFormerModule).
        datamodule : MolLoader
        max_epochs : int
        callbacks : dict
            Must contain 'early_stopping' and 'all_metrics' keys.
        logger : Logger
        config : dict
            Full config dict (from YAML). config['training'] is used.
        """
        tc = config["training"]
        args = types.SimpleNamespace(**tc)

        train_dataloader = datamodule.train_dataloader()
        module.configure_optimizers(config)
        grad_scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

        # Resume from checkpoint if specified
        load_ckpt = config["paths"]["load_checkpoint"]
        epoch_start = 0
        if load_ckpt:
            epoch_start = load_state(module, pathlib.Path(load_ckpt), callbacks) + 1

        for epoch_idx in range(epoch_start, max_epochs):
            if isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch_idx)

            loss = module.train_one_epoch(
                train_dataloader,
                epoch_idx,
                grad_scaler,
                callbacks,
                config,
            )

            if module.lr_scheduler is not None:
                module.lr_scheduler.step()

            if dist.is_initialized():
                dist.all_reduce(loss)
                loss /= self.world_size

            loss_val = loss.item()
            logging.info(f"Epoch {epoch_idx + 1}: train loss = {loss_val:.6f}")
            logger.log_metrics({"train loss": loss_val}, epoch_idx)

            for callback in callbacks.values():
                callback.on_epoch_end(loss_val)

            if (
                tc.get("eval_interval", 1) > 0
                and (epoch_idx + 1) % tc.get("eval_interval", 1) == 0
            ) or epoch_idx + 1 == max_epochs:
                self.validate(module, datamodule, callbacks, config, epoch_idx)

                if callbacks["all_metrics"].last_r2 == callbacks["all_metrics"].best_r2:
                    save_state(
                        module,
                        epoch_idx,
                        self.log_dir / "best_epoch_ckpt.pth",
                        callbacks,
                    )

            if callbacks["early_stopping"].early_stop:
                logging.info("Early stopping triggered.")
                break

        save_state(
            module,
            epoch_idx,
            self.log_dir / "last_epoch_ckpt.pth",
            callbacks,
        )

        for callback in callbacks.values():
            callback.on_fit_end()

    def validate(
        self,
        module: Module,
        datamodule: MolLoader,
        callbacks: dict[str, BaseCallback],
        config: dict,
        epoch: int = -1,
    ) -> dict:
        """Validate the model and invoke validation-end callbacks."""
        val_dataloader = datamodule.val_dataloader()
        module.evaluate_one_epoch(val_dataloader, callbacks, config)

        callbacks["all_metrics"].on_validation_end(epoch)
        callbacks["early_stopping"].on_validation_end(epoch, callbacks["all_metrics"].last_r2)

        return {
            "mae": callbacks["all_metrics"].last_mae,
            "rmse": callbacks["all_metrics"].last_rmse,
            "r2": callbacks["all_metrics"].last_r2,
            "pearson": callbacks["all_metrics"].last_pearson,
        }

    def test(
        self,
        module: Module,
        datamodule: MolLoader,
        callbacks: dict[str, BaseCallback],
        config: dict,
        epoch: int = -1,
    ) -> dict:
        """Evaluate the model on the test dataloader."""
        test_dataloader = datamodule.test_dataloader()
        module.evaluate_one_epoch(test_dataloader, callbacks, config)
        callbacks["all_metrics"].on_validation_end(epoch)

        return {
            "mae": callbacks["all_metrics"].last_mae,
            "rmse": callbacks["all_metrics"].last_rmse,
            "r2": callbacks["all_metrics"].last_r2,
            "pearson": callbacks["all_metrics"].last_pearson,
        }
