"""Training utilities: experiment builder, checkpointing, and Trainer loop."""

import logging
import argparse
import pathlib
import torch
import torch.distributed as dist

from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from typing import Literal

from src.models import CPMPModule, EGNNModule, SE3TransformerModule, Module
from src.callbacks import BaseCallback
from src.loggers import Logger
from src.data_module import DataModule
from src.dataset import (
    SE3TransformerQM9DataModule,
    EGNNQM9DataModule,
    CPMPQM9DataModule,
    SE3TransformerCP3DDataModule,
    EGNNCP3DDataModule,
    CPMPCP3DDataModule,
)


REGISTRY = {
    "se3t": {
        "module_cls": SE3TransformerModule,
        "qm9": SE3TransformerQM9DataModule,
        "cp3d": SE3TransformerCP3DDataModule,
    },
    "egnn": {
        "module_cls": EGNNModule,
        "qm9": EGNNQM9DataModule,
        "cp3d": EGNNCP3DDataModule,
    },
    "cpmp": {
        "module_cls": CPMPModule,
        "qm9": CPMPQM9DataModule,
        "cp3d": CPMPCP3DDataModule,
    },
}


def build_experiment(
    model: Literal["se3t", "egnn", "cpmp"],
    dataset: str,
    device: torch.device,
    local_rank: int,
    args: argparse.ArgumentParser,
) -> tuple[Module, DataModule]:
    """Instantiate a model wrapper and matching DataModule.

    Parameters
    ----------
    model : {"se3t", "egnn", "cpmp"}
        Model architecture key.
    dataset : str
        Dataset key compatible with the chosen model.
    device : torch.device
        Compute device.
    local_rank : int
        Process local rank (0 for single GPU).
    args : argparse.Namespace
        Parsed CLI arguments used to configure both classes.

    Returns
    -------
    (Module, DataModule)
        Wrapper for training/eval and the associated DataModule.
    """
    if model not in {"se3t", "egnn", "cpmp"}:
        raise ValueError(
            f"Unknown model: {model}. Supported models are se3t, egnn, cpmp only."
        )
    arch_entry = REGISTRY[model]
    dm_cls = arch_entry[dataset]
    datamodule = dm_cls(**vars(args))
    if model == "se3t":
        modelmodule = arch_entry["module_cls"](
            datamodule.NODE_FEATURE_DIM,
            datamodule.EDGE_FEATURE_DIM,
            device,
            local_rank,
            args,
        )
    elif model == "cpmp":
        modelmodule = arch_entry["module_cls"](
            datamodule.d_atom, device, local_rank, args
        )
    else:
        train_dataloader = datamodule.train_dataloader()
        values = train_dataloader.dataset.data[args.task]
        mean = torch.mean(values)
        mad = torch.mean(torch.abs(values - mean))
        modelmodule = arch_entry["module_cls"](
            mean,
            mad,
            datamodule.max_charge,
            datamodule.num_species,
            device,
            local_rank,
            args,
        )

    return modelmodule, datamodule


def save_state(
    modelmodule: Module,
    epoch: int,
    path: pathlib.Path,
    callbacks: dict[str, BaseCallback],
):
    """Save model, optimizer, and epoch state to a checkpoint file.

    Only rank 0 performs the write in distributed setting.
    """
    if modelmodule.local_rank == 0:
        state_dict = (
            modelmodule.model.module.state_dict()
            if isinstance(modelmodule.model, DistributedDataParallel)
            else modelmodule.model.state_dict()
        )
        checkpoint = {
            "state_dict": state_dict,
            "optimizer_state_dict": modelmodule.optimizer.state_dict(),
            "epoch": epoch,
        }
        for _, callback in callbacks.items():
            callback.on_checkpoint_save(checkpoint)

        torch.save(checkpoint, str(path))
        logging.info(f"Saved checkpoint to {str(path)}")


def load_state(
    modelmodule: Module,
    path: pathlib.Path,
    callbacks: dict[str, BaseCallback],
):
    """Load model, optimizer, and epoch state from a checkpoint file."""
    checkpoint = torch.load(
        str(path), map_location={"cuda:0": f"cuda:{modelmodule.local_rank}"}
    )
    if isinstance(modelmodule.model, DistributedDataParallel):
        modelmodule.model.module.load_state_dict(checkpoint["state_dict"])
    else:
        modelmodule.model.load_state_dict(checkpoint["state_dict"])
    modelmodule.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for _, callback in callbacks.items():
        callback.on_checkpoint_load(checkpoint)

    logging.info(f"Loaded checkpoint from {str(path)}")
    return checkpoint["epoch"]


class Trainer:
    """Minimal training loop coordinator.

    Parameters
    ----------
    world_size : int
        Number of processes for distributed training (1 for single-GPU).
    """

    def __init__(self, world_size: int, log_dir: pathlib.Path):
        self.world_size = world_size
        self.log_dir = log_dir

    def fit(
        self,
        module: Module,
        datamodule: DataModule,
        max_epochs: int,
        callbacks: dict[str, BaseCallback],
        logger: Logger,
        args: argparse.ArgumentParser,
    ):
        """Run the training loop with periodic validation and checkpointing."""
        train_dataloader = datamodule.train_dataloader()
        module.model.train()
        module.configure_optimizers(args)
        grad_scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

        epoch_start = (
            load_state(module, args.load_ckpt_path, callbacks) + 1
            if args.load_ckpt_path
            else 0
        )

        for epoch_idx in range(epoch_start, max_epochs):
            if isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch_idx)
            loss = module.train_one_epoch(
                train_dataloader,
                epoch_idx,
                grad_scaler,
                callbacks,
                args,
            )

            if module.lr_scheduler is not None:
                module.lr_scheduler.step()

            if dist.is_initialized():
                torch.distributed.all_reduce(loss)
                loss /= self.world_size

            loss = loss.item()
            logging.info(f"Epoch {epoch_idx + 1}:")
            logging.info(f" train loss: {loss}")
            logger.log_metrics({"train loss": float(loss)}, epoch_idx)

            for _, callback in callbacks.items():
                callback.on_epoch_end(loss)

            if args.ckpt_interval > 0 and (epoch_idx + 1) % args.ckpt_interval == 0:
                save_state(
                    module,
                    epoch_idx,
                    self.log_dir / f"epoch_{epoch_idx}_ckpt.pth",
                    callbacks,
                )

            if (
                args.eval_interval > 0 and (epoch_idx + 1) % args.eval_interval == 0
            ) or epoch_idx + 1 == args.epochs:
                module.model.eval()
                self.validate(module, datamodule, callbacks, args, epoch_idx)

                if callbacks["all_metrics"].last_r2 == callbacks["all_metrics"].best_r2:
                    save_state(
                        module,
                        epoch_idx,
                        self.log_dir / "best_epoch_ckpt.pth",
                        callbacks,
                    )

                module.model.train()

            if callbacks["early_stopping"].early_stop:
                logging.info("Early stopping triggered")
                break

        save_state(
            module,
            epoch_idx,
            self.log_dir / "last_epoch_ckpt.pth",
            callbacks,
        )

        for _, callback in callbacks.items():
            callback.on_fit_end()

    def validate(
        self,
        module: Module,
        datamodule: DataModule,
        callbacks: dict[str, BaseCallback],
        args: argparse.ArgumentParser,
        epoch: int = -1,
    ):
        """Validate the model and invoke validation-end callbacks."""
        val_dataloader = datamodule.val_dataloader()
        module.evaluate_one_epoch(val_dataloader, callbacks, args)

        callbacks["all_metrics"].on_validation_end(epoch)

        metrics = {
            "mae": callbacks["all_metrics"].last_mae,
            "rmse": callbacks["all_metrics"].last_rmse,
            "r2": callbacks["all_metrics"].last_r2,
            "pearson": callbacks["all_metrics"].last_pearson,
        }

        callbacks["early_stopping"].on_validation_end(epoch, metrics["r2"])

        return metrics

    def test(
        self,
        module: Module,
        datamodule: DataModule,
        callbacks: dict[str, BaseCallback],
        args: argparse.ArgumentParser,
        epoch: int = -1,
    ):
        """Evaluate the model on the test dataloader."""
        test_dataloader = datamodule.test_dataloader()
        module.evaluate_one_epoch(test_dataloader, callbacks, args)

        callbacks["all_metrics"].on_validation_end(epoch)

        metrics = {
            "mae": callbacks["all_metrics"].last_mae,
            "rmse": callbacks["all_metrics"].last_rmse,
            "r2": callbacks["all_metrics"].last_r2,
            "pearson": callbacks["all_metrics"].last_pearson,
        }

        return metrics
