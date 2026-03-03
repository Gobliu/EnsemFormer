"""Abstract base class for EnsemFormer model modules."""

import torch
from abc import ABC, abstractmethod
from tqdm import tqdm


class Module(ABC):
    """Abstract wrapper providing a uniform training and inference interface."""

    def __init__(self, device, local_rank):
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.lr_scheduler = None
        self.device = device
        self.local_rank = local_rank

    @abstractmethod
    def configure_optimizers(self, args):
        """Instantiate optimizer and optional LR scheduler."""

    @abstractmethod
    def train_one_epoch(self, train_dataloader, epoch_idx, grad_scaler, callbacks, args):
        """Run a single training epoch and return average loss."""

    @abstractmethod
    def evaluate_one_epoch(self, val_dataloader, callbacks, args):
        """Run a single evaluation epoch and return validation metrics."""

    @abstractmethod
    @torch.inference_mode()
    def predict(self, batch):
        """Perform inference on a single batch and return raw predictions."""

    def save_checkpoint(self, path, epoch):
        """Standardized checkpointing including model state and metadata."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.lr_scheduler:
            state['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path):
        """Load weights and metadata."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0)

    def _get_tqdm(self, dataloader, desc, disable, leave=False):
        return tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            unit="batch",
            desc=desc,
            disable=disable,
            leave=leave,
        )
