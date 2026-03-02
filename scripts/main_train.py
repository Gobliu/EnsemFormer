"""EnsemFormer training entrypoint.

Usage:
    python scripts/main_train.py                                # uses config/default.yaml
    python scripts/main_train.py --config config/custom.yaml
    python scripts/main_train.py --learning_rate 5e-4 --epochs 100
    torchrun --nproc_per_node=2 scripts/main_train.py
"""

import argparse
import logging
import pathlib
import warnings

import torch
import torch.distributed as dist
import yaml

# Anchor the repo root using __file__ — no relative ../ strings
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_config(config_path: str, cli_overrides: dict) -> dict:
    """Load YAML config and apply flat CLI overrides.

    Scans all top-level section dicts for matching keys and overwrites them.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    for key, value in cli_overrides.items():
        if key == "config" or value is None:
            continue
        for section in config.values():
            if isinstance(section, dict) and key in section:
                section[key] = value
                break

    return config


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EnsemFormer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "config" / "default.yaml"),
    )
    # Most-commonly overridden keys (all others live in the YAML)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_conformers", type=int)
    parser.add_argument("--gnn_type", type=str, choices=["egnn", "cpmp"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--version", type=int)
    parser.add_argument("--silent", action="store_true")
    return parser.parse_args()


def main():
    warnings.simplefilter("ignore", FutureWarning)
    torch.set_float32_matmul_precision("high")

    cli_args = get_args()
    config = load_config(cli_args.config, vars(cli_args))

    from src.utils import (
        init_distributed,
        get_local_rank,
        print_parameters_count,
        get_next_version,
    )
    from src.dataset import ConformerEnsembleDataModule
    from src.networks.cycloformer import CycloFormerModule
    from src.trainer import Trainer
    from src.callbacks import EarlyStoppingCallback, AllMetricsCallback
    from src.loggers import CSVLogger, TensorBoardLogger, LoggerCollection

    is_distributed = init_distributed()
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    torch.manual_seed(config["data"]["seed"])

    log_level = (
        logging.CRITICAL
        if (local_rank != 0 or config["training"].get("silent", False))
        else logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # ------------------------------------------------------------------
    # DataModule
    # ------------------------------------------------------------------
    paths_cfg = config["paths"]
    data_cfg = config["data"]

    datamodule = ConformerEnsembleDataModule(
        data_dir=_REPO_ROOT / paths_cfg["data_dir"],
        csv_path=data_cfg.get("csv_file", "pampa.csv"),
        conformer_source=data_cfg["conformer_source"],
        n_conformers=data_cfg["n_conformers"],
        pdb_dir=paths_cfg.get("pdb_dir"),
        split=data_cfg.get("split"),
        ff=data_cfg["ff"],
        add_dummy_node=data_cfg["add_dummy_node"],
        one_hot_formal_charge=data_cfg.get("one_hot_formal_charge", False),
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        seed=data_cfg["seed"],
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    gnn_cfg = config["gnn"]
    tf_cfg = config["conformer_transformer"]
    head_cfg = config["head"]

    d_gnn = gnn_cfg.get("hidden_nf") if gnn_cfg["type"] == "egnn" else gnn_cfg.get("d_model", 128)

    modelmodule = CycloFormerModule(
        gnn_type=gnn_cfg["type"],
        d_atom=datamodule.d_atom,
        d_gnn=d_gnn,
        d_model=tf_cfg["d_model"],
        n_tf_heads=tf_cfg["n_heads"],
        n_tf_layers=tf_cfg["n_layers"],
        d_ff=tf_cfg["d_ff"],
        dropout=tf_cfg["dropout"],
        pooling=tf_cfg["pooling"],
        max_conformers=tf_cfg["max_conformers"],
        device=device,
        local_rank=local_rank,
        # Pass remaining GNN-specific kwargs
        **{k: v for k, v in gnn_cfg.items() if k not in ("type", "hidden_nf", "d_model")},
    )

    # Move all sub-modules to device
    modelmodule.gnn_encoder.to(device)
    modelmodule.conformer_encoder.to(device)
    modelmodule.head.to(device)
    modelmodule.cls_token.data = modelmodule.cls_token.data.to(device)

    if local_rank == 0:
        print_parameters_count(modelmodule.gnn_encoder)

    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel
        modelmodule.gnn_encoder = DistributedDataParallel(
            modelmodule.gnn_encoder, device_ids=[local_rank], output_device=local_rank
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    tc = config["training"]
    log_base = _REPO_ROOT / paths_cfg["log_dir"] / gnn_cfg["type"]
    version = tc.get("version") or get_next_version(log_base)
    log_dir = log_base / f"run_{version}"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = LoggerCollection([
        CSVLogger(log_dir / "logs" / "csv"),
        TensorBoardLogger(log_dir / "logs" / "tsb"),
    ])
    logger.log_hyperparams(config)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = {
        "early_stopping": EarlyStoppingCallback(
            patience=tc["patience"],
            delta=tc["delta"],
            direction="max",
        ),
        "all_metrics": AllMetricsCallback(logger, rescale_factor=1, prefix="valid"),
    }

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(world_size, log_dir)
    trainer.fit(
        modelmodule,
        datamodule,
        max_epochs=tc["epochs"],
        callbacks=callbacks,
        logger=logger,
        config=config,
    )

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    test_callbacks = {
        "early_stopping": EarlyStoppingCallback(patience=9999, delta=0.0, direction="max"),
        "all_metrics": AllMetricsCallback(logger, rescale_factor=1, prefix="test"),
    }
    metrics = trainer.test(modelmodule, datamodule, test_callbacks, config)
    logging.info(f"Test results: {metrics}")


if __name__ == "__main__":
    main()
