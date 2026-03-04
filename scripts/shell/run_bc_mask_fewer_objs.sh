#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# Incremental Mask Baseline Sample Efficiency Sweep on train_bc_pt.py
# Trains 20 epochs incrementally per episode in the dataset.
# Evaluates on 1 episode immediately after each training step.

set -e
LR=0.0001
DATASET="data/seaquest/full_data_16_episodes_10p0_sigma_win_10_obj_limit_2.pt"
RUN_DIR="trained_models/bc/bc_mask_fewer_objs_2_max_epoch_50"
GAZE_MODEL_PATH="trained_models/gaze_predictor/seaquest_visual_gaze_predictor_limit_2.pth"
EVAL_EPISODES=1
EPOCHS_PER_EPISODE=50

echo "=========================================="
echo "  Mask (Incremental, train_bc_pt.py)"
echo "  Evaluating 1 episode every phase"
echo "  Training 20 epochs per episode"
echo "=========================================="

conda run -n nesy-il python -u train_bc_pt.py \
    --dataset "$DATASET" \
    --gaze_method "Mask" \
    --send_email \
    --epochs "$EPOCHS_PER_EPISODE" \
    --lr $LR \
    --gaze_model_path "$GAZE_MODEL_PATH" \
    --eval_interval 3 \
    --eval_episodes "$EVAL_EPISODES" \
    --run_dir "$RUN_DIR" \
    