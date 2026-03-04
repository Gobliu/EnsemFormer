# Multi-GPU Training

This project supports Distributed Data Parallel (DDP) through `scripts/main_train.py`.

## Single Node, All Visible GPUs
```bash
python scripts/main_train.py
```

## Select Specific GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/main_train.py
```

## Notes
- The script initializes DDP automatically in `utils.distributed.ddp_setup`.
- Effective global batch size scales with GPU count (`batch_size` per process).
- Checkpoints are saved under `experiments/<model_name>/run_<n>/`.
