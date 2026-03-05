# Training Loop (`scripts/main_train.py`)

## How to run

```bash
# Single GPU
python scripts/main_train.py

# With a custom config
python scripts/main_train.py --config config/custom.yaml

# Common CLI overrides (match keys in config/default.yaml sections)
python scripts/main_train.py --gnn_type cpmp --learning_rate 5e-4 --epochs 100

# Multi-GPU (DDP via torchrun)
torchrun --nproc_per_node=2 scripts/main_train.py
```

---

## Startup (`scripts/main_train.py::load_config`)

```
┌────────────────────────────────────────────────────────────┐
│  load_config(config/default.yaml, cli_overrides)           │
│                                                            │
│  1. Load YAML (paths / data / gnn / conformer_transformer  │
│               / head / training sections)                  │
│  2. Apply flat CLI --key value overrides:                  │
│     --gnn_type  → config['gnn']['type']  (special rename)  │
│     all others  → scan sections for matching key           │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│  main()                                                    │
│   torch.set_float32_matmul_precision("high")               │
└────────────────────────────────────────────────────────────┘
```

---

## Step 1 — Distributed + Device + Seed

```
init_distributed()   (src/utils.py)
  │  If torchrun (RANK env var): init NCCL process group → is_distributed=True
  │  Otherwise: no-op (single-GPU)
  ▼
local_rank = get_local_rank()
world_size = dist.get_world_size()  (or 1)
device = cuda:{local_rank}  (or cpu)
torch.manual_seed(config['data']['seed'])

logging: INFO on rank 0 only (or always silent if --silent)
```

---

## Step 2 — Data (`src/dataset.py`)

```
ConformerEnsembleDataModule(
    data_dir    = repo_root / paths.data_dir,
    csv_path    = paths.csv_file,
    cache_file  = paths.cache_file,       # .pt from preprocess_trajectories.py
    env         = data.env,               # [water], [hexane], [water, hexane]
    n_conformers = data.n_conformers,     # frames per env per molecule
    rep_frame_only = data.rep_frame_only,
    split       = data.split,             # 0-9 cross-val index
    batch_size  = data.batch_size,
    num_workers = data.num_workers,
    seed        = data.seed,
)
  │
  ├─ ConformerEnsembleDataset  (train)   ──┐
  ├─ ConformerEnsembleDataset  (val)       ├─ split from CSV
  └─ ConformerEnsembleDataset  (test)    ──┘
       each molecule → ConformerEnsembleMolecule
         .node_feat  : (N_atoms, F)
         .conformers : list[(dist, coords)]
         .y          : float  (regression target)

  conformer_collate_fn  pads atoms & conformers → batch dict
```

---

## Step 3 — Model (`src/networks/cycloformer.py`)

```
CycloFormerModule(
    gnn_type   = gnn.type,          # 'egnn' | 'cpmp' | 'se3t'
    d_atom     = datamodule.d_atom,
    d_gnn      = inferred from gnn config,
    d_model    = conformer_transformer.d_model,
    n_tf_heads = conformer_transformer.n_heads,
    n_tf_layers= conformer_transformer.n_layers,
    d_ff       = conformer_transformer.d_ff,
    dropout    = conformer_transformer.dropout,
    pooling    = conformer_transformer.pooling,    # 'cls' | 'mean'
    max_conformers = conformer_transformer.max_conformers,
    mode       = gnn.mode,          # 'ensemble' | 'standalone'
    **gnn_kwargs,                   # encoder-specific params
)
  │
  └─ .model (CycloFormerCore)
       ├─ .backbone          → EGNNBackbone | CPMPBackbone | SE3TBackbone
       ├─ .conformer_encoder → Transformer over conformer tokens  (ensemble mode)
       ├─ .head              → 2-layer MLP → scalar
       ├─ .proj              → optional Linear(d_gnn → d_model)
       └─ .cls_token         → learnable CLS prepended to conformer sequence

DDP wrapping: modelmodule.model (CycloFormerCore) is wrapped with
              DistributedDataParallel, covering all learnable parameters.
              (rank 0 logs parameter count)
```

---

## Step 4 — Logging (`src/loggers.py`)

```
log_dir = repo_root / paths.log_dir / gnn.type / run_{version}/
  │  version = training.version  or  auto-incremented from existing run_N dirs

LoggerCollection([
    CSVLogger(log_dir / 'logs' / 'csv'),       # metrics.csv + hparams.txt
    TensorBoardLogger(log_dir / 'logs' / 'tsb'),
])
logger.log_hyperparams(config)   # writes full config at startup
```

---

## Step 5 — Callbacks (`src/callbacks.py`)

```
EarlyStoppingCallback(
    patience  = training.patience,   # default: 20
    delta     = training.delta,      # default: 0.0001
    direction = 'max',               # maximise R²
)

AllMetricsCallback(logger, rescale_factor=1, prefix='valid')
    tracks per-epoch:  MAE, RMSE, R², Pearson r
    via torchmetrics (MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef)
```

---

## Step 6 — Training Loop (`src/trainer.py::Trainer.fit`)

```
module.configure_optimizers(config)
  └─ AdamW(lr=training.learning_rate, weight_decay=training.weight_decay)
     + optional LR scheduler (if configured)

grad_scaler = GradScaler(enabled=training.amp)

# Optional checkpoint resume
if paths.load_checkpoint:
    epoch_start = load_state(module, load_checkpoint, callbacks) + 1

for epoch_idx in range(epoch_start, training.epochs):
    │
    │  DDP sync: train_sampler.set_epoch(epoch_idx)
    │
    │  module.train_one_epoch(train_dataloader, epoch_idx, grad_scaler, callbacks, config)
    │       ┌─────────────────────────────────────────────────────────┐
    │       │  for batch in train_dataloader:                         │
    │       │    forward:   pred = model(batch)                       │
    │       │    loss:      MSE(pred, target)                         │
    │       │    backward:  grad_scaler.scale(loss).backward()        │
    │       │    clip:      gradient_clip (if set)                    │
    │       │    step:      grad_scaler.step(optimizer)               │
    │       │    accumulate: total_loss                               │
    │       │  return mean train loss                                 │
    │       └─────────────────────────────────────────────────────────┘
    │
    │  if lr_scheduler: lr_scheduler.step()     (once per epoch)
    │
    │  if DDP: dist.all_reduce(loss) / world_size
    │
    │  logger.log_metrics({'train loss': loss_val}, epoch_idx)
    │  callbacks.on_epoch_end(loss_val)
    │
    │  every training.eval_interval epochs (default: 1):
    │       ┌─────────────────────────────────────────────────────────┐
    │       │  module.evaluate_one_epoch(val_dataloader, ...)         │
    │       │    → AllMetricsCallback.on_validation_step per batch    │
    │       │                                                         │
    │       │  AllMetricsCallback.on_validation_end(epoch)            │
    │       │    → logs MAE / RMSE / R² / Pearson r                   │
    │       │                                                         │
    │       │  EarlyStoppingCallback.on_validation_end(epoch, R²)     │
    │       │                                                         │
    │       │  if R² == best R² so far:                               │
    │       │    save_state → log_dir/best_epoch_ckpt.pth  (rank 0)   │
    │       └─────────────────────────────────────────────────────────┘
    │
    │  if early_stopping.early_stop: break
    │
    ▼
save_state → log_dir/last_epoch_ckpt.pth   (rank 0)
callbacks.on_fit_end()   → logs best MAE / RMSE / R² / Pearson r
```

---

## Step 7 — Test (`src/trainer.py::Trainer.test`)

```
trainer.test(modelmodule, datamodule, test_callbacks, config)
  │
  ├─ module.evaluate_one_epoch(test_dataloader, ...)
  ├─ AllMetricsCallback.on_validation_end()
  └─ returns {mae, rmse, r2, pearson}

logging.info(f"Test results: {metrics}")
```

---

## File Layout (`experiments/`)

```
experiments/
  └─ {gnn_type}/            e.g. egnn/ | cpmp/ | se3t/
       └─ run_0/
                ├─ logs/
                │    ├─ csv/
                │    │    ├─ metrics.csv    all logged metrics (step-indexed)
                │    │    └─ hparams.txt    flat key: value config snapshot
                │    └─ tsb/
                │         └─ events.out.*   TensorBoard event file
                ├─ best_epoch_ckpt.pth      checkpoint at best validation R²
                └─ last_epoch_ckpt.pth      checkpoint at end of training
```

Each `.pth` contains:
```python
{
    'state_dict':           OrderedDict,  # model weights (DDP-unwrapped)
    'optimizer_state_dict': dict,
    'epoch':                int,
    # + any fields added by callbacks via on_checkpoint_save
}
```

---

## LR Schedule

Default: no scheduler (constant LR).
To enable, configure a scheduler inside `module.configure_optimizers`.
Scheduler steps once per epoch (after `train_one_epoch`, before validation).
