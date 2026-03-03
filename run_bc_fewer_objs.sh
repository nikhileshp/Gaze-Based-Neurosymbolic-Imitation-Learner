#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
# Incremental AGIL Baseline Sample Efficiency Sweep on train_bc_pt.py
# Trains 10 epochs incrementally per episode in the dataset.
# Evaluates on 5 episodes immediately after each training step.

set -e
LR=0.0001
DATASET="data/seaquest/full_data_16_episodes_10p0_sigma_win_10_obj_limit_1.pt"
EVAL_EPISODES=1
EPOCHS_PER_EPISODE=20

echo "=========================================="
echo "  AGIL (Incremental, train_bc_pt.py)"
echo "  Evaluating 1 namesake every epoch"
echo "  Training 20 epochs full dataset"
echo "=========================================="

conda run -n nesy-il python -u train_bc_pt.py \
    --dataset "$DATASET" \
    --gaze_method "None" \
    --send_email \
    --epochs "$EPOCHS_PER_EPISODE" \
    --eval_episodes 1\
    --lr $LR \
    --eval_episodes "$EVAL_EPISODES"


