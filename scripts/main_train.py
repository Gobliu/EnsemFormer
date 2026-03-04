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
import sys
import warnings

import torch
import torch.distributed as dist
import yaml

# Anchor the repo root using __file__ — no relative ../ strings
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def load_config(config_path: str, cli_overrides: dict) -> dict:
    """Load YAML config and apply flat CLI overrides.

    Scans all top-level section dicts for matching keys and overwrites them.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Map CLI arg names to (section, key) when they differ from the YAML key
    cli_renames = {
        "gnn_type": ("gnn", "type"),
    }

    for key, value in cli_overrides.items():
        if key == "config" or value is None:
            continue
        if key in cli_renames:
            section_name, yaml_key = cli_renames[key]
            config[section_name][yaml_key] = value
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
    parser.add_argument("--gnn_type", type=str, choices=["egnn", "cpmp", "se3t"])
    parser.add_argument("--mode", type=str, choices=["ensemble", "standalone"])
    parser.add_argument("--env", nargs="+", type=str, help="Environments: water hexane")
    parser.add_argument("--rep_frame_only", action="store_true")
    parser.add_argument("--use_bond_type", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--version", type=int)
    parser.add_argument("--silent", action="store_true")
    return parser.parse_args()


def main():
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", UserWarning)  # DGL graphbolt C++ lib missing (harmless)
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

    cache_file = paths_cfg.get("cache_file")
    if cache_file:
        cache_file = str(_REPO_ROOT / cache_file)

    cfg_env = data_cfg["env"]
    env = cfg_env if isinstance(cfg_env, list) else [cfg_env]

    datamodule = ConformerEnsembleDataModule(
        data_dir=_REPO_ROOT / paths_cfg["data_dir"],
        csv_path=paths_cfg["csv_file"],
        cache_file=cache_file,
        env=env,
        n_conformers=data_cfg["n_conformers"],
        rep_frame_only=data_cfg["rep_frame_only"],
        split=data_cfg["split"],
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

    gnn_type = gnn_cfg["type"]
    if gnn_type == "egnn":
        d_gnn = gnn_cfg["hidden_nf"]
    elif gnn_type == "se3t":
        d_gnn = gnn_cfg["num_degrees"] * gnn_cfg["num_channels"]
    else:
        d_gnn = gnn_cfg["d_model"]

    # Build SE3T-specific kwargs with renamed keys
    gnn_kwargs = {k: v for k, v in gnn_cfg.items()
                  if k not in ("type", "mode", "hidden_nf", "d_model", "use_bond_type")}
    # Rename se3t-prefixed keys for SE3TBackbone constructor
    if gnn_type == "se3t":
        rename_map = {
            "se3t_num_layers": "num_layers",
            "se3t_num_heads": "num_heads",
            "se3t_norm": "norm",
            "se3t_use_layer_norm": "use_layer_norm",
            "se3t_low_memory": "low_memory",
        }
        gnn_kwargs = {rename_map.get(k, k): v for k, v in gnn_kwargs.items()}

    modelmodule = CycloFormerModule(
        gnn_type=gnn_type,
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
        mode=gnn_cfg.get("mode", "ensemble"),
        use_bond_type=gnn_cfg.get("use_bond_type", False),
        **gnn_kwargs,
    )

    modelmodule.model.to(device)

    if local_rank == 0:
        print_parameters_count(modelmodule.model)

    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel
        modelmodule.model = DistributedDataParallel(
            modelmodule.model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
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
