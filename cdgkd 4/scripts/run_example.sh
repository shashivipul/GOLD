#!/usr/bin/env bash
# GOLD end-to-end example:
#   Stage 1: GraphMAE pretrain a source teacher on Cora.
#   Stage 2: cross-domain distill into an MLP on Citeseer.
set -euo pipefail

SOURCE=cora
TARGET=citeseer
GNN=GCN
HIDDEN=128

CKPT="checkpoints/teacher_${SOURCE}_${GNN}.pth"

# Stage 1
python src/pretrain.py \
    --source "$SOURCE" \
    --gnn "$GNN" \
    --hidden_dim "$HIDDEN" \
    --num_layers 2 \
    --epochs 200 \
    --mask_rate 0.5 \
    --replace_rate 0.1 \
    --alpha_l 2.0 \
    --save_path "$CKPT"

# Stage 2
python src/train.py \
    --source "$SOURCE" \
    --target "$TARGET" \
    --gnn "$GNN" \
    --hidden_dim "$HIDDEN" \
    --teacher_ckpt "$CKPT" \
    --lambda_ot 0.5 \
    --lambda_de 0.1 \
    --lambda_kd 1.0 \
    --kd_num_anchors 512 \
    --kd_num_pos 5 \
    --kd_num_neg 15 \
    --kd_tau 0.5 \
    --epochs 200
