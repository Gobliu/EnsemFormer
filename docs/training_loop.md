# Training Loop (`scripts/main_train.py`)

## How to run

```bash
# Single GPU
python scripts/main_train.py

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 scripts/main_train.py

# With CLI overrides
python scripts/main_train.py --batch_size 8 --maxlr 0.0001
```

---

## Startup (`src/utils/config_loader.py`)

```
┌──────────────────────────────────────────────────────────┐
│  Config.from_yaml_and_cli(config/default.yaml)         │
│                                                          │
│  1. Load YAML defaults                                   │
│  2. Apply CLI --key value overrides                      │
│  3. Resolve data file paths (relative to repo root)      │
│  4. Compute derived fields:                              │
│     effective_loss_weights = loss_weights[-S:]           │
│  5. Resolve checkpoint: loadfile + run_dir               │
│     load_weight = 'best' → find latest run_N/best.pth    │
│     load_weight = 'last' → find latest run_N/last.pth    │
│     load_weight = 'none' → train from scratch            │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  main(cfg)                                               │
└──────────────────────────────────────────────────────────┘
```

---

## Step 1 — DDP + Device + Seed (`src/utils/distributed.py`, `src/utils/seed.py`)

```
ddp_setup()
  │  If RANK env var exists (torchrun): init NCCL process group
  │  Otherwise: no-op (single-GPU mode)
  ▼
device = cuda:{local_rank} or cpu
torch.set_default_dtype(float64)
set_global_seed(cfg.seed + rank)
  │  Seeds: Python hashseed, random, numpy, torch, cuda
```

---

## Step 2 — Data Loading (`src/datasets.py`)

```
build_dataloaders(cfg)
  │
  ├─ input_traj_range = cfg.input_steps  (= 8)
  ├─ label_range = (input_traj_range - 1) + window_sliding
  │
  ├─ TrajectoryDataset(train_file, ..., time_step=cfg.time_step)
  │    └─ Loads .pt, computes stride = round(time_step / native_time_gap)
  │    └─ If stride > 1: subsamples qp[:, :, ::stride, :, :]
  │    └─ Slices into input [2,T,N,3] / label [2,S,N,3]
  │    └─ Truncated to dpt_train samples if set
  │
  ├─ TrajectoryDataset(valid_file, ...)
  │    └─ Truncated to dpt_valid samples if set
  │
  ├─ DistributedSampler (if DDP, else None)
  │
  └─ DataLoader × 2 (train, val)
       num_workers=8, pin_memory=True (if CUDA)
       train: shuffle=True (or sampler)
       val:   shuffle=False
```

---

## Step 3 — Build Model (`src/build_model.py`)

```
build_model(cfg, device, box_size, raw_atom_id, mass)
  │
  ├─ Icosahedron(grid_radius=b[0], box_size)
  ├─ PhiGenerator(ngrids, raw_atom_id,
  │                rbf_n_per_bank, r_cut, rbf_learnable,
  │                mlp_layers=phi_mlp_layers)
  ├─ PsiGenerator(ngrids)
  ├─ HalfStepUpdate(cfg, tau_init=time_step)  × 2  (q_updater, p_updater)
  │
  ├─ LUFNetRollout(tau, mass, box_size, ...)
  │    └─ .to(device)
  │
  ├─ DDP wrap (if distributed)
  │
  └─ Returns: model, model_unwrapped, tau_params=[tau_q, tau_p]
```

See `docs/data_flow.md` for the full forward pass.

---

## Step 4 — Build Loss (`src/loss.py`)

```
build_loss(cfg, box_size)
  └─ Returns closure: loss_fn(q_pred_list, p_pred_list, qpl_label)
     Uses: effective_loss_weights, loss_type ('mse'/'l1'/'smooth_l1')
```

---

## Step 5 — Optimizer + Scheduler (`src/optim.py`)

```
build_optimizer(model, cfg)
  └─ Adam(lr=maxlr, weight_decay=weight_decay)  (or AdamW, SGD)

build_scheduler(optimizer, cfg)
  └─ CosineAnnealingLR(T_max=end_epoch)
```

---

## Step 6 — Checkpoint Resume (`src/checkpoint.py`)

```
CheckpointManager(model_unwrapped, optimizer, run_dir, scheduler)
  │
  ├─ load(loadfile)
  │    ├─ If loadfile exists: restore model, optimizer, scheduler state
  │    │   → returns (start_epoch, best_v_loss)
  │    ├─ If loadfile is None: train from scratch
  │    │   → returns (0, inf)
  │    └─ Safety: raises if run_dir has .pth files but loadfile is None
  │       (prevents accidental overwrite)
  │
  └─ save_config(cfg)
       └─ Dumps config.yaml snapshot into run_dir (rank 0 only)
```

---

## Step 7 — Logger Init (`src/utils/log_utils.py`)

```
TrainLogger(run_dir / 'train.log')
  │  Opens file in append mode (supports resume)
  │  Writes to both stdout and file; flushes after each line
  │  No-op on non-main DDP ranks
  │
  ├─ Fresh start (start_epoch == 0):
  │    log_start(logger, cfg, device, model, train_dataset, val_dataset)
  │      → device, param count, hyperparams, dataset info, training settings
  │
  └─ Resume (start_epoch > 0):
       log_resume(logger, start_epoch)
         → "--- Resumed: <timestamp>, starting from epoch N ---"
```

---

## Step 8 — Training Loop (`src/trainer.py`)

```
for epoch in range(start_epoch, end_epoch):
    │
    │  DDP sampler sync
    │    train_sampler.set_epoch(epoch)  — ensures different
    │    shuffling per epoch across ranks
    │
    │  Train one epoch → returns avg train_loss
    │       ┌──────────────────────────────────────────────────────┐
    │       │  for batch in train_loader:                          │
    │       │    cast to (device, float64)                         │
    │       │    forward:  q_preds, p_preds = model(qp)            │
    │       │    loss:     trajectory_loss(preds, label)           │
    │       │    backward: loss.backward()                         │
    │       │    clip:     clip_grad_norm_(grad_clip)              │
    │       │    step:     optimizer.step()                        │
    │       │    accumulate: total_loss, count                     │
    │       │  return total_loss / count                           │
    │       └──────────────────────────────────────────────────────┘
    │
    │  Step LR scheduler (cosine: every epoch, no val_loss)
    │
    │  Validate + log + checkpoint (every cfg.val_interval epochs)
    │       ┌──────────────────────────────────────────────────────┐
    │       │  val_loss = validate(model, val_loader)              │
    │       │    └─ same as train but no_grad, no step             │
    │       │                                                      │
    │       │  log_metrics(epoch, lr, train_loss, val_loss, tau)   │
    │       │    → writes to stdout + train.log                    │
    │       │                                                      │
    │       │  Save last.pth (all ranks check, rank 0 writes)      │
    │       │                                                      │
    │       │  if val_loss < best_v_loss:                          │
    │       │    Save best.pth  (rank 0 only)                      │
    │       └──────────────────────────────────────────────────────┘
    │
    ▼
```

---

## Step 9–10 — Cleanup

```
log_end(logger, best_v_loss, best_epoch)
  → "Best val_loss: X at epoch Y"
  → "Total time: Xh Ym Zs"
  → closes log file

ddp_cleanup()    →  destroy process group (if DDP)
```

---

## File Layout (experiments/)

```
experiments/
  └─ {model_name}/       e.g. "vanilla" or "maxlr=0.001-trans_dim=256"
       └─ run_0/
                ├─ config.yaml   snapshot of training config
                ├─ train.log     training log (appended on resume)
                ├─ last.pth      latest checkpoint (overwritten each val_interval)
                └─ best.pth      best validation loss
```

Each `.pth` contains:
```python
{
    'epoch': int,
    'val_loss': float,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,   # if scheduler exists
}
```

---

## LR Schedule

```
lr
 ▲
 │  maxlr ─┐
 │         │╲
 │         │  ╲
 │         │    ╲               Cosine annealing
 │         │      ╲             T_max = end_epoch
 │         │        ╲
 │         │          ──────╲
 │         │                  ╲─── → 0
 └─────────┴──────────────────────► epoch
           0                    end_epoch
```

Cosine scheduler steps once per epoch (after training, before validation).
