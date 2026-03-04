"""Training-loop methods for CycloFormerModule (mixin).

Implements the abstract interface from ``src.module.Module``:
configure_optimizers, train_one_epoch, evaluate_one_epoch, predict.
"""

import types

import torch
import torch.nn as nn

from src.utils import to_device


class CycloFormerTrainingMixin:
    """Mixin providing training / evaluation / inference methods.

    Expects the host class to have: ``gnn_encoder``, ``conformer_encoder``,
    ``head``, ``cls_token``, ``proj``, ``loss_fn``, ``device``, ``local_rank``,
    ``forward()``, and ``_get_tqdm()``.
    """

    def configure_optimizers(self, config: dict):
        """Instantiate AdamW optimizer and cosine-annealing LR scheduler."""
        tc = config["training"]
        self.optimizer = torch.optim.AdamW(
            list(self.gnn_encoder.parameters())
            + list(self.conformer_encoder.parameters())
            + list(self.head.parameters())
            + [self.cls_token]
            + (list(self.proj.parameters()) if not isinstance(self.proj, nn.Identity) else []),
            lr=tc["learning_rate"],
            weight_decay=tc["weight_decay"],
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=tc["epochs"],
        )

    def train_one_epoch(
        self,
        train_dataloader,
        epoch_idx: int,
        grad_scaler,
        callbacks: dict,
        config: dict,
    ) -> torch.Tensor:
        args = types.SimpleNamespace(**config["training"])
        self.gnn_encoder.train()
        self.conformer_encoder.train()
        self.head.train()

        loss_acc = torch.zeros((1,), device=self.device)
        for i, batch in self._get_tqdm(
            train_dataloader,
            desc=f"Epoch {epoch_idx}",
            disable=(args.silent or self.local_rank != 0),
        ):
            batch = to_device(batch, self.device)

            for cb in callbacks.values():
                cb.on_batch_start()

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = self.forward(batch)
                target = batch["target"]
                loss = self.loss_fn(pred.flatten(), target.flatten()) / args.accumulate_grad_batches

            loss_acc += loss.detach()
            grad_scaler.scale(loss).backward()

            if (i + 1) % args.accumulate_grad_batches == 0 or (i + 1) == len(train_dataloader):
                if args.gradient_clip is not None:
                    grad_scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        list(self.gnn_encoder.parameters())
                        + list(self.conformer_encoder.parameters())
                        + list(self.head.parameters()),
                        args.gradient_clip,
                    )
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

        return loss_acc / len(train_dataloader)

    @torch.inference_mode()
    def evaluate_one_epoch(
        self,
        val_dataloader,
        callbacks: dict,
        config: dict,
    ) -> torch.Tensor:
        args = types.SimpleNamespace(**config["training"])
        self.gnn_encoder.eval()
        self.conformer_encoder.eval()
        self.head.eval()

        loss_acc = torch.zeros((1,), device=self.device)
        for _, batch in self._get_tqdm(
            val_dataloader,
            desc="Evaluation",
            disable=(args.silent or self.local_rank != 0),
        ):
            batch = to_device(batch, self.device)

            for cb in callbacks.values():
                cb.on_batch_start()

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = self.forward(batch)
                target = batch["target"]

                for cb in callbacks.values():
                    cb.on_validation_step(None, target, pred)

                loss = self.loss_fn(pred.flatten(), target.flatten()) / args.accumulate_grad_batches

            loss_acc += loss.detach()

        return loss_acc / len(val_dataloader)

    @torch.inference_mode()
    def predict(self, batch: dict) -> torch.Tensor:
        """Run inference and return raw predictions (B, 1)."""
        self.gnn_encoder.eval()
        self.conformer_encoder.eval()
        self.head.eval()
        batch = to_device(batch, self.device)
        return self.forward(batch)
