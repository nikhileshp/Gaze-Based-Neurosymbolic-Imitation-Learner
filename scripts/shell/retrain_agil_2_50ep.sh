#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# Retrain AGIL with 2-object limit for 50 epochs on GPU 3
set -e
LR=0.0001
DATASET="data/seaquest/full_data_16_episodes_10p0_sigma_win_10_obj_limit_2.pt"
RUN_DIR="trained_models/bc/agil_fewer_objs_2_50_ep_new"
GAZE_MODEL_PATH="trained_models/gaze_predictor/seaquest_visual_gaze_predictor_limit_2.pth"
EVAL_EPISODES=1
EPOCHS=50

echo "=========================================="
echo "  Retraining AGIL (2-obj) for 50 epochs on GPU 3"
echo "  Run Dir: $RUN_DIR"
echo "=========================================="

conda run -n nesy-il python -u train_bc_pt.py \
    --dataset "$DATASET" \
    --gaze_method "AGIL" \
    --send_email \
    --epochs "$EPOCHS" \
    --eval_episodes "$EVAL_EPISODES" \
    --lr $LR \
    --gaze_model_path "$GAZE_MODEL_PATH" \
    --run_dir "$RUN_DIR"
