"""Model wrappers exposing a minimal training/eval interface.

Each wrapper encapsulates a specific architecture (SE3-Transformer, CPMP, EGNN)
and provides three core methods used by the training loop:

- ``configure_optimizers``
- ``train_one_epoch``
- ``evaluate_one_epoch``
"""

import torch
from abc import ABC, abstractmethod
from tqdm import tqdm

from se3_transformer.se3_transformer.model.fiber import Fiber
from se3_transformer.se3_transformer.model.transformer import SE3TransformerPooled
from egnn.qm9.utils import preprocess_input, get_adj_matrix
from egnn.qm9.models import EGNN
from cpmp.model.transformer import CPMPGraphTransformer

from src.utils import to_device, using_tensor_cores


class Module(ABC):
    """Abstract wrapper providing a uniform training interface.

    This base class stores the model, loss function, and device metadata. Child
    classes implement optimizer setup and per-epoch train/eval logic.

    Parameters
    ----------
    device : torch.device
        Compute device for the model/batches.
    local_rank : int
        Rank of the current process in distributed training (0 for single-GPU).
    """

    def __init__(self, device, local_rank):
        self.model = None
        self.loss_fn = None
        self.device = device
        self.local_rank = local_rank

    @abstractmethod
    def configure_optimizers(self, args):
        """Instantiate optimizer and optional LR scheduler from parsed args."""

    @abstractmethod
    def train_one_epoch(self):
        """Run a single training epoch. Must return a scalar loss tensor."""

    @abstractmethod
    def evaluate_one_epoch(self):
        """Run a single evaluation epoch. Must return a scalar loss tensor."""


class SE3TransformerModule(Module):
    """Wrapper around :class:`SE3TransformerPooled` for regression."""

    def __init__(self, node_feature_dim, edge_feature_dim, device, local_rank, args):
        super().__init__(device, local_rank)
        self.model = SE3TransformerPooled(
            fiber_in=Fiber({0: node_feature_dim}),
            fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
            fiber_edge=Fiber({0: edge_feature_dim}),
            output_dim=1,
            tensor_cores=using_tensor_cores(args.amp),
            **vars(args),
        )
        self.loss_fn = torch.nn.L1Loss()

    def train_one_epoch(
        self,
        train_dataloader,
        epoch_idx,
        grad_scaler,
        callbacks,
        args,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        loss_acc = torch.zeros((1,), device=self.device)
        for i, batch in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            unit="batch",
            desc=f"Epoch {epoch_idx}",
            disable=(args.silent or self.local_rank != 0),
            leave=False,
        ):
            *inputs, target = to_device(batch, self.device)

            for _, callback in callbacks.items():
                callback.on_batch_start()

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = self.model(*inputs)
                loss = (
                    self.loss_fn(pred.flatten(), target.flatten())
                    / args.accumulate_grad_batches
                )

            loss_acc += loss.detach()
            grad_scaler.scale(loss).backward()

            if (i + 1) % 1 == 0 or (i + 1) == len(train_dataloader):
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.model.zero_grad(set_to_none=True)

        return loss_acc / len(train_dataloader)

    @torch.inference_mode()
    def evaluate_one_epoch(self, val_dataloader, callbacks, args):
        """Evaluate the model on the validation dataloader and update callbacks.

        Parameters
        ----------
        val_dataloader : DataLoader
        callbacks : dict
        args : argparse.Namespace
        """
        loss_acc = torch.zeros((1,), device=self.device)
        for _, batch in tqdm(
            enumerate(val_dataloader),
            total=len(val_dataloader),
            unit="batch",
            desc="Evaluation",
            leave=False,
            disable=(args.silent or self.local_rank != 0),
        ):
            *inputs, target = to_device(batch, self.device)

            for _, callback in callbacks.items():
                callback.on_batch_start()

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = self.model(*inputs)

                for _, callback in callbacks.items():
                    callback.on_validation_step(input, target, pred)

                loss = (
                    self.loss_fn(pred.flatten(), target.flatten())
                    / args.accumulate_grad_batches
                )

            loss_acc += loss.detach()

        return loss_acc / len(val_dataloader)

    def configure_optimizers(self, args):
        """Create optimizer and cosine warm restarts scheduler from args."""
        if args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.learning_rate,
                betas=(args.momentum, 0.999),
                weight_decay=args.weight_decay,
                fused=True,
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                fused=True,
            )
        min_lr = (
            args.min_learning_rate
            if args.min_learning_rate
            else args.learning_rate / 10.0
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, args.epochs, eta_min=min_lr, last_epoch=-1
        )


class CPMPModule(Module):
    """Wrapper for the CPMP graph transformer model."""

    def __init__(self, d_atom, device, local_rank, args):
        super().__init__(device, local_rank)
        self.model = CPMPGraphTransformer(d_atom, **vars(args))
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

    def train_one_epoch(
        self,
        train_dataloader,
        epoch_idx,
        grad_scaler,
        callbacks,
        args,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        loss_acc = torch.zeros((1,), device=self.device)
        for i, batch in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            unit="batch",
            desc=f"Epoch {epoch_idx}",
            disable=(args.silent or self.local_rank != 0),
            leave=False,
        ):
            adjacency_matrix, node_features, distance_matrix, target = batch

            for _, callback in callbacks.items():
                callback.on_batch_start()

            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

            adjacency_matrix = adjacency_matrix.to(self.device, non_blocking=True)
            node_features = node_features.to(self.device, non_blocking=True)
            distance_matrix = distance_matrix.to(self.device, non_blocking=True)
            batch_mask = batch_mask.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = self.model(
                    node_features, batch_mask, adjacency_matrix, distance_matrix, None
                )
                loss = (
                    self.loss_fn(pred.flatten(), target.flatten())
                    / args.accumulate_grad_batches
                )

            loss_acc += loss.detach()
            grad_scaler.scale(loss).backward()

            if (i + 1) % 1 == 0 or (i + 1) == len(train_dataloader):
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.model.zero_grad(set_to_none=True)

        return loss_acc / len(train_dataloader)

    @torch.inference_mode()
    def evaluate_one_epoch(self, val_dataloader, callbacks, args):
        """Evaluate CPMP model and update callbacks with predictions/targets."""
        loss_acc = torch.zeros((1,), device=self.device)
        for _, batch in tqdm(
            enumerate(val_dataloader),
            total=len(val_dataloader),
            unit="batch",
            desc="Evaluation",
            leave=False,
            disable=(args.silent or self.local_rank != 0),
        ):
            adjacency_matrix, node_features, distance_matrix, target = batch

            for _, callback in callbacks.items():
                callback.on_batch_start()

            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0

            adjacency_matrix = adjacency_matrix.to(self.device, non_blocking=True)
            node_features = node_features.to(self.device, non_blocking=True)
            distance_matrix = distance_matrix.to(self.device, non_blocking=True)
            batch_mask = batch_mask.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = self.model(
                    node_features, batch_mask, adjacency_matrix, distance_matrix, None
                )

                for _, callback in callbacks.items():
                    callback.on_validation_step(input, target, pred)

                loss = (
                    self.loss_fn(pred.flatten(), target.flatten())
                    / args.accumulate_grad_batches
                )

            loss_acc += loss.detach()

        return loss_acc / len(val_dataloader)

    def configure_optimizers(self, args):
        """Create an AdamW optimizer from args."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(args.momentum, 0.999),
            weight_decay=args.weight_decay,
            fused=True,
        )
        self.lr_scheduler = None


class EGNNModule(Module):
    """Wrapper for the EGNN model with target normalization metadata."""

    def __init__(self, mean, mad, max_charge, num_species, device, local_rank, args):
        super().__init__(device, local_rank)
        self.mean = mean
        self.mad = mad
        self.max_charge = max_charge
        self.num_species = num_species
        self.model = EGNN(**vars(args))
        self.loss_fn = torch.nn.L1Loss()

    def train_one_epoch(
        self,
        train_dataloader,
        epoch_idx,
        grad_scaler,
        callbacks,
        args,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        loss_acc = torch.zeros((1,), device=self.device)
        for i, batch in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            unit="batch",
            desc=f"Epoch {epoch_idx}",
            disable=(args.silent or self.local_rank != 0),
            leave=False,
        ):
            batch_size, n_nodes, _ = batch["positions"].size()

            for _, callback in callbacks.items():
                callback.on_batch_start()

            atom_positions = (
                batch["positions"]
                .view(batch_size * n_nodes, -1)
                .to(self.device, torch.float32)
            )
            atom_mask = (
                batch["atom_mask"]
                .view(batch_size * n_nodes, -1)
                .to(self.device, torch.float32)
            )
            edge_mask = batch["edge_mask"].to(
                self.device, torch.float32, non_blocking=True
            )
            one_hot = batch["one_hot"].to(self.device, torch.float32)
            charges = batch["charges"].to(self.device, torch.float32)
            nodes = preprocess_input(
                one_hot, charges, args.charge_power, self.max_charge, self.device
            )

            nodes = nodes.view(batch_size * n_nodes, -1)
            edges = get_adj_matrix(n_nodes, batch_size, self.device)
            target = batch[args.task].to(self.device, torch.float32)

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = self.model(
                    h0=nodes,
                    x=atom_positions,
                    edges=edges,
                    edge_attr=None,
                    node_mask=atom_mask,
                    edge_mask=edge_mask,
                    n_nodes=n_nodes,
                )
                loss = (
                    self.loss_fn(
                        pred.flatten(), (target.flatten() - self.mean) / self.mad
                    )
                    / args.accumulate_grad_batches
                )

            loss_acc += loss.detach()
            grad_scaler.scale(loss).backward()

            if (i + 1) % 1 == 0 or (i + 1) == len(train_dataloader):
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.model.zero_grad(set_to_none=True)

        return loss_acc / len(train_dataloader)

    @torch.inference_mode()
    def evaluate_one_epoch(self, val_dataloader, callbacks, args):
        """Evaluate EGNN and call callbacks with rescaled predictions."""
        loss_acc = torch.zeros((1,), device=self.device)
        for _, batch in tqdm(
            enumerate(val_dataloader),
            total=len(val_dataloader),
            unit="batch",
            desc="Evaluation",
            leave=False,
            disable=(args.silent or self.local_rank != 0),
        ):
            batch_size, n_nodes, _ = batch["positions"].size()

            for _, callback in callbacks.items():
                callback.on_batch_start()

            atom_positions = (
                batch["positions"]
                .view(batch_size * n_nodes, -1)
                .to(self.device, torch.float32)
            )
            atom_mask = (
                batch["atom_mask"]
                .view(batch_size * n_nodes, -1)
                .to(self.device, torch.float32)
            )
            edge_mask = batch["edge_mask"].to(
                self.device, torch.float32, non_blocking=True
            )
            one_hot = batch["one_hot"].to(self.device, torch.float32)
            charges = batch["charges"].to(self.device, torch.float32)
            nodes = preprocess_input(
                one_hot, charges, args.charge_power, self.max_charge, self.device
            )

            nodes = nodes.view(batch_size * n_nodes, -1)
            edges = get_adj_matrix(n_nodes, batch_size, self.device)
            target = batch[args.task].to(self.device, torch.float32)

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = self.model(
                    h0=nodes,
                    x=atom_positions,
                    edges=edges,
                    edge_attr=None,
                    node_mask=atom_mask,
                    edge_mask=edge_mask,
                    n_nodes=n_nodes,
                )

                pred = self.mad * pred + self.mean

                for _, callback in callbacks.items():
                    callback.on_validation_step(input, target, pred)

                loss = (
                    self.loss_fn(pred.flatten(), target.flatten())
                    / args.accumulate_grad_batches
                )

            loss_acc += loss.detach()

        return loss_acc / len(val_dataloader)

    def configure_optimizers(self, args):
        """Create an AdamW optimizer from args."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(args.momentum, 0.999),
            weight_decay=args.weight_decay,
            fused=True,
        )
        self.lr_scheduler = None
