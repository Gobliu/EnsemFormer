#!/usr/bin/env bash

BATCH_SIZE=${1:-512}
AMP=${2:-false}
NUM_EPOCHS=${3:-25}
LEARNING_RATE=${4:-0.00001}
WEIGHT_DECAY=${5:-0.1}
SPLIT=${6:-0}

python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
    src.main \
    --model "cpmp" \
    --amp "$AMP" \
    --data_dir "data/cpmp/pkl" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$NUM_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --min_lr 0.00001 \
    --weight_decay "$WEIGHT_DECAY" \
    --split "$SPLIT"