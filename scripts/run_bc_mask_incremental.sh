#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
# Incremental BC+Mask on train_bc_pt.py
# Trains 20 epochs incrementally per episode in the full dataset.
# Evaluates on 50 episodes immediately after each training step.
# Early stopping and LR patience disabled (patience=0).

set -e
LR=0.0001
DATASET="data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt"
RUN_DIR="models/bc/bc_mask_incremental_50ep"
GAZE_MODEL_PATH="models/gaze_predictor/seaquest_gaze_predictor_2.pth"
EVAL_EPISODES=50
EPOCHS_PER_EPISODE=50

echo "=========================================="
echo "  BC+Mask (Incremental, Full Dataset, 20ep)"
echo "  Patience = 0 (No Early Stopping)"
echo "  LR Patience = 0 (No LR reduction)"
echo "  Eval Episodes = 50"
echo "=========================================="

conda run -n nesy-il python -u train_bc_pt.py \
    --dataset "$DATASET" \
    --gaze_method "Mask" \
    --send_email \
    --incremental \
    --epochs "$EPOCHS_PER_EPISODE" \
    --lr $LR \
    --gaze_model_path "$GAZE_MODEL_PATH" \
    --eval_episodes "$EVAL_EPISODES" \
    --patience 0 \
    --lr_patience 0 \
    --use_gazemap \
    --run_dir "$RUN_DIR" 


