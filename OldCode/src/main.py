"""Training entry point for CycPept3D.

Parses command-line arguments, initializes distributed training if requested,
builds the selected model and dataset, and runs training and evaluation.
"""

import logging
import warnings
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch_geometric import seed_everything
from src.arguments import PARSER, ARGS_DICT, COMMON_ARGS

from src.callbacks import EarlyStoppingCallback, AllMetricsCallback
from src.loggers import TensorBoardLogger, CSVLogger, LoggerCollection
from src.training import Trainer, build_experiment
from src.utils import (
    get_local_rank,
    init_distributed,
    print_parameters_count,
    get_next_version,
)


if __name__ == "__main__":
    warnings.simplefilter("ignore", FutureWarning)
    torch.set_float32_matmul_precision("high")
    is_distributed = init_distributed()
    args = PARSER.parse_args()
    seed_everything(int(args.seed))
    device = torch.cuda.current_device()
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    logging.getLogger().setLevel(
        logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO
    )
    modelmodule, datamodule = build_experiment(
        args.model, args.dataset, device, local_rank, args
    )

    logging.info("========== CycPept3D ==========")
    logging.info("|      Training procedure     |")
    logging.info("===============================")
    logging.info(f"Training model {args.model} on dataset {args.dataset}")

    print_parameters_count(modelmodule.model)

    modelmodule.model.to(device)

    if dist.is_initialized():
        modelmodule.model = DistributedDataParallel(
            modelmodule.model, device_ids=[local_rank], output_device=local_rank
        )
        modelmodule.model._set_static_graph()

    modelmodule.configure_optimizers(args)

    ARGS = list(COMMON_ARGS)
    ARGS.extend(ARGS_DICT[args.model])
    if args.model == "cpmp" and args.dataset == "cp3d":
        ARGS.extend(ARGS_DICT["cpmp_cp3d"])

    ARGS_DIR = "_".join(f"{arg}_{getattr(args, arg)}" for arg in ARGS)

    LOG_DIR = args.log_dir / args.model / args.dataset / ARGS_DIR
    version = args.version if args.version is not None else get_next_version(LOG_DIR)
    LOG_DIR = LOG_DIR / f"run_{version}"

    loggers = [
        CSVLogger(LOG_DIR / "logs" / "csv"),
        TensorBoardLogger(LOG_DIR / "logs" / "tsb"),
    ]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(vars(args))

    # Use datamodule.rescale_factor if available, otherwise default to 1
    rescale_factor = getattr(datamodule, "rescale_factor", 1)
    callbacks = {
        "early_stopping": EarlyStoppingCallback(args.patience, args.delta, "max"),
        "all_metrics": AllMetricsCallback(logger, rescale_factor),
    }

    trainer = Trainer(world_size, LOG_DIR)
    trainer.fit(modelmodule, datamodule, args.epochs, callbacks, logger, args)

    test_callbacks = {
        "all_metrics": AllMetricsCallback(logger, rescale_factor, prefix="test"),
    }
    trainer.test(modelmodule, datamodule, test_callbacks, args)
