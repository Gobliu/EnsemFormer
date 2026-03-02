#!/usr/bin/env bash

BATCH_SIZE=${1:-1024}
AMP=${2:-false}
NUM_EPOCHS=${3:-25}
LEARNING_RATE=${4:-0.0021}
WEIGHT_DECAY=${5:-0.1}
SPLIT=${6:-0}

python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
    src.test \
    --model "egnn" \
    --task "homo" \
    --dataset qm9 \
    --data_dir "data/egnn/QM9" \
    --log_dir "results/egnn/qm9" \
    --amp "$AMP" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$NUM_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --min_lr 0.00001 \
    --weight_decay "$WEIGHT_DECAY" \
    --split "$SPLIT" \
    --save_ckpt_path "results/egnn/test_qm9_$SPLIT.pth"