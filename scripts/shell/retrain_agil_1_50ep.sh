#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
# Retrain AGIL with 1-object limit for 50 epochs on GPU 2
set -e
LR=0.0001
DATASET="data/seaquest/full_data_16_episodes_10p0_sigma_win_10_obj_limit_1.pt"
RUN_DIR="models/bc/agil_fewer_objs_1_50_ep_new"
GAZE_MODEL_PATH="models/gaze_predictor/seaquest_gaze_predictor_limit_1.pth"
EVAL_EPISODES=1
EPOCHS=50

echo "=========================================="
echo "  Retraining AGIL (1-obj) for 50 epochs on GPU 2"
echo "  Run Dir: $RUN_DIR"
echo "=========================================="

conda run -n nesy-il python -u train_bc_pt.py \
    --dataset "$DATASET" \
    --gaze_method "AGIL" \
    --send_email \
    --epochs "$EPOCHS" \
    --eval_episodes "$EVAL_EPISODES" \
    --lr $LR \
    --eval_interval 10 \
    --gaze_model_path "$GAZE_MODEL_PATH" \
    --run_dir "$RUN_DIR"
