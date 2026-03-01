#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
# Incremental BC Baseline Sample Efficiency Sweep: Plain BC vs AGIL
# Trains exactly 1 epoch sequentially per episode for all 28 episodes.
# Evaluates on 50 episodes immediately after each training step.
# Results saved to: bc_sample_efficiency_new_runs.csv
# Checkpoints saved to: models/bc/incremental_{none|agil}/{N}_ep/

set -e
LR=0.0001
DATASET="data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt"
EVAL_EPISODES=50


# ── PLAIN BC ──────────────────────────────────────────────────────────────────
echo "=========================================="
echo "  PLAIN BC (no gaze, INCREMENTAL)"
echo "=========================================="
conda run -n nesy-il python -u train_bc.py \
    --datapath "$DATASET" \
    --incremental \
    --gaze_method "BC"\
    --n_epochs 500 \
    --lr $LR \
    --eval_interval 500\
    --num_eval_episodes "$EVAL_EPISODES" \
    --send_email \
    --email_interval 30 \
    --result_csv "models/Seaquest/bc_results.csv"


# ── AGIL ──────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  AGIL (gaze-augmented BC, INCREMENTAL)"
echo "=========================================="
conda run -n nesy-il python -u train_bc.py \
    --datapath "$DATASET" \
    --incremental \
    --gaze_method "AGIL" \
    --n_epochs 500 \
    --stack 4 \
    --lr $LR \
    --eval_interval 500 \
    --num_eval_episodes "$EVAL_EPISODES" \
    --send_email \
    --email_interval 30 \
    --result_csv "models/Seaquest/bc_results.csv"

