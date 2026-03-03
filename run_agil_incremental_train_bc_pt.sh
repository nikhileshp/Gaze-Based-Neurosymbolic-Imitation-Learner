#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# Incremental AGIL Baseline Sample Efficiency Sweep on train_bc_pt.py
# Trains 10 epochs incrementally per episode in the dataset.
# Evaluates on 5 episodes immediately after each training step.

set -e
LR=0.0001
DATASET="data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt"
EVAL_EPISODES=5
EPOCHS_PER_EPISODE=10

echo "=========================================="
echo "  AGIL (Incremental, train_bc_pt.py)"
echo "  Evaluating 5 episodes per incrementally learned episode"
echo "  Training 10 epochs per increment"
echo "=========================================="

conda run -n nesy-il python -u train_bc_pt.py \
    --dataset "$DATASET" \
    --incremental \
    --gaze_method "AGIL" \
    --epochs "$EPOCHS_PER_EPISODE" \
    --lr $LR \
    --eval_episodes "$EVAL_EPISODES"
