#!/usr/bin/env bash

BATCH_SIZE=${1:-64}
AMP=${2:-true}
NUM_EPOCHS=${3:-10}
LEARNING_RATE=${4:-0.01}
WEIGHT_DECAY=${5:-0.1}
SPLIT=${6:-0}

python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
    src.test \
    --amp "$AMP" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$NUM_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --min_lr 0.00001 \
    --weight_decay "$WEIGHT_DECAY" \
    --use_layer_norm \
    --norm \
    --split "$SPLIT" \
    --save_ckpt_path "results/se3t/test_cp3d_$SPLIT.pth"