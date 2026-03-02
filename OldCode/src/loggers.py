"""Logging backends for metrics and hyperparameters.

This module defines a minimal Logger interface with concrete implementations:

- ``CSVLogger``: writes metrics to a CSV file and hparams to a text file
- ``TensorBoardLogger``: writes metrics and hparams to TensorBoard event files
- ``LoggerCollection``: multiplex calls to multiple loggers

All loggers implement two methods: :meth:`log_hyperparams` and
:meth:`log_metrics`.
"""

import csv
import pathlib
import torch
import torch.distributed as dist
from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, Callable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from src.utils import rank_zero_only


class Logger(ABC):
    """Abstract base class for loggers.

    Subclasses must implement :meth:`log_hyperparams` and :meth:`log_metrics`.
    """

    @abstractmethod
    def log_hyperparams(self, params):
        """Log a dictionary of hyperparameters.

        Parameters
        ----------
        params : dict
            Set of hyperparameters, typically ``vars(args)``.
        """

        pass

    @abstractmethod
    def log_metrics(self, metrics, step=None):
        """Log a dictionary of scalar metrics.

        Parameters
        ----------
        metrics : dict[str, float | Tensor | dict]
            Metric name to value mapping. Nested dicts are supported by some
            backends (e.g., TensorBoard ``add_scalars``).
        step : int, optional
            Optional global step or epoch index.
        """

        pass

    @staticmethod
    def _sanitize_params(params):
        """Convert complex values into serializable forms.

        Parameters
        ----------
        params : dict
            Hyperparameter dictionary possibly containing callables, Enums,
            and Path objects.

        Returns
        -------
        dict
            A copy with values converted to strings, ints, floats, or other
            serializable types.
        """

        def _sanitize(val):
            if isinstance(val, Callable):
                try:
                    _val = val()
                    if isinstance(_val, Callable):
                        return val.__name__
                    return _val
                except Exception:
                    return getattr(val, "__name__", None)
            elif isinstance(val, pathlib.Path) or isinstance(val, Enum):
                return str(val)
            return val

        return {key: _sanitize(val) for key, val in params.items()}


class LoggerCollection(Logger):
    """Dispatch logging calls to multiple logger backends.

    Parameters
    ----------
    loggers : Sequence[Logger]
        Iterable of instantiated logger objects.
    """

    def __init__(self, loggers):
        super().__init__()
        self.loggers = loggers

    def __getitem__(self, index):
        return [logger for logger in self.loggers][index]

    def log_metrics(self, metrics, step=None):
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def log_hyperparams(self, params):
        for logger in self.loggers:
            logger.log_hyperparams(params)


class CSVLogger(Logger):
    """Persist metrics to CSV and hyperparameters to a text file.

    Directory layout
    ----------------
    save_dir/
        metrics.csv
        hparams.txt
    """

    NAME_METRICS_FILE = "metrics.csv"
    HPARAMS_FILE = "hparams.txt"

    def __init__(self, save_dir: pathlib.Path):
        """Create a CSV logger.

        Parameters
        ----------
        save_dir : pathlib.Path
            Base directory where versioned logs are stored.
        """
        super().__init__()
        self.log_dir = pathlib.Path(save_dir)
        self._metrics_fieldnames: list[str] | None = None
        self._metrics_file_path = self.log_dir / self.NAME_METRICS_FILE
        self._hparams_file_path = self.log_dir / self.HPARAMS_FILE
        self.experiment = None

        # Only rank 0 creates dirs and the writer
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if (
                self._metrics_file_path.exists()
                and self._metrics_file_path.stat().st_size > 0
            ):
                with self._metrics_file_path.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    self._metrics_fieldnames = list(reader.fieldnames or [])

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Write hyperparameters to a simple text file.

        Parameters
        ----------
        params : dict[str, Any]
            Hyperparameter mapping (keys are sorted for determinism).
        """
        params = self._sanitize_params(params)
        keys = sorted(params.keys())
        with self._hparams_file_path.open("w", encoding="utf-8") as f:
            for k in keys:
                v = params[k]
                f.write(f"{k}: {v}\n")

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Append a row to ``metrics.csv``.

        Parameters
        ----------
        metrics : dict[str, float]
            Mapping of metric names to values.
        step : int, optional
            Global step or epoch. If not provided, 0 is used.
        """
        row = metrics.copy()
        if step is not None:
            row["step"] = int(step)
        elif "step" not in row:
            row["step"] = 0
        if self._metrics_fieldnames is None:
            if (
                self._metrics_file_path.exists()
                and self._metrics_file_path.stat().st_size > 0
            ):
                with self._metrics_file_path.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    self._metrics_fieldnames = list(reader.fieldnames or [])
            else:
                fields = set(row.keys())
                ordered = ["step"] + sorted([k for k in fields if k != "step"])
                self._metrics_fieldnames = ordered
                with self._metrics_file_path.open(
                    "w", newline="", encoding="utf-8"
                ) as f:
                    writer = csv.DictWriter(f, fieldnames=self._metrics_fieldnames)
                    writer.writeheader()
        present = set(self._metrics_fieldnames)
        incoming = set(row.keys())
        new_fields = [k for k in incoming if k not in present]
        if new_fields:
            merged = ["step"] + sorted(
                [k for k in present.union(incoming) if k != "step"]
            )
            existing_rows = []
            if (
                self._metrics_file_path.exists()
                and self._metrics_file_path.stat().st_size > 0
            ):
                with self._metrics_file_path.open(
                    "r", newline="", encoding="utf-8"
                ) as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        existing_rows.append(r)
            with self._metrics_file_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=merged)
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow(r)
            self._metrics_fieldnames = merged
        with self._metrics_file_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._metrics_fieldnames, extrasaction="ignore"
            )
            writer.writerow(row)


class TensorBoardLogger(Logger):
    """Write metrics and hyperparameters to TensorBoard event files."""

    def __init__(self, save_dir: pathlib.Path):
        """Create a TensorBoard logger.

        Parameters
        ----------
        save_dir : pathlib.Path
            Base directory containing event files.
        """
        super().__init__()
        self.log_dir = pathlib.Path(save_dir)

        if not dist.is_initialized() or dist.get_rank() == 0:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.experiment = SummaryWriter(log_dir=self.log_dir)

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard HPARAMS plugin.

        Parameters
        ----------
        params : dict[str, Any]
            Mapping of hyperparameters to serializable values.
        """
        params = self._sanitize_params(params)
        metrics = {"hp_metric": -1}
        exp, ssi, sei = hparams(params, metrics)
        writer = self.experiment._get_file_writer()
        writer.add_summary(exp, None)
        writer.add_summary(ssi, None)
        writer.add_summary(sei, None)

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics as scalars or grouped scalars to TensorBoard.

        Parameters
        ----------
        metrics : dict
            Metric names to values (float, Tensor, or nested dict for
            ``add_scalars``)
        step : int, optional
            Global step.
        """
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, dict):
                self.experiment.add_scalars(k, v, step)
            else:
                try:
                    self.experiment.add_scalar(k, v, step)
                except Exception as ex:
                    raise ValueError(
                        f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
                    ) from ex
