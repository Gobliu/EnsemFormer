#!/usr/bin/env bash

BATCH_SIZE=${1:-1024}
AMP=${2:-false}
NUM_EPOCHS=${3:-25}
LEARNING_RATE=${4:-0.0021}
WEIGHT_DECAY=${5:-0.1}
SPLIT=${6:-0}

python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts 0 --module \
    src.main \
    --model "egnn" \
    --task "homo" \
    --dataset qm9 \
    --data_dir "data/egnn/QM9" \
    --amp "$AMP" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$NUM_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --min_lr 0.00001 \
    --weight_decay "$WEIGHT_DECAY" \
    --split "$SPLIT" \
    --load_ckpt_path "new_results/egnn/qm9/charge_power_2_in_node_nf_15_hidden_nf_128_n_layers_7_coords_weight_1.0_attention_False_node_attr_False/run_0/best_epoch_ckpt.pth" \
    --version 0