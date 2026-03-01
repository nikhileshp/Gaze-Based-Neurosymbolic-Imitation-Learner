#!/bin/bash
# Sample Efficiency Sweep: Gaze vs No-Gaze
# After each training run, evaluates best.pth for 30 episodes.
# Episode-wise rewards -> {run_dir}/run_experiment_{gaze_tag}_{N}_ep.log
# Mean reward per run  -> sample_efficiency_runs.csv
export CUDA_VISIBLE_DEVICES=2
set -e

DATASET="data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt"
RULES="new"
LR=0.01
EPOCHS=20
BATCH=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEED=42
EVAL_EPISODES=50
EPISODES=(1 2 4 8 16 28)
SUMMARY_CSV="sample_efficiency_runs_gaze_128_batch.csv"

# Write CSV header if file doesn't exist yet
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "num_episodes,gaze,mean_reward,std_reward,model_path" > "$SUMMARY_CSV"
fi

# ── Helper: evaluate best.pth and log results ─────────────────────────────────
run_eval() {
    local N=$1
    local GAZE_TAG=$2       # "gaze" or "no_gaze"
    local GAZE_FLAGS=$3     # "--use_gazemap" or ""
    local RUN_DIR="models/nsfr/seaquest/${GAZE_TAG}/${N}_ep"
    local BEST_MODEL="${RUN_DIR}/best.pth"
    local LOG_FILE="${RUN_DIR}/run_experiment_${GAZE_TAG}_${N}_ep.log"

    if [ ! -f "$BEST_MODEL" ]; then
        echo "WARNING: best.pth not found at $BEST_MODEL — skipping eval."
        return
    fi

    echo "--- Evaluating best.pth: ${GAZE_TAG} | N=${N} ---"
    conda run -n nesy-il python evaluate_model.py \
        --model_path "$BEST_MODEL" \
        --episodes "$EVAL_EPISODES" \
        --seed "$SEED" \
        --rules "$RULES" \
        $GAZE_FLAGS \
        --send_email \
        
        2>&1 | tee "$LOG_FILE"

    # Parse mean and std from the log
    MEAN=$(grep "Mean Reward:" "$LOG_FILE" | tail -1 | awk '{print $3}')
    STD=$(grep "Std Deviation:" "$LOG_FILE" | tail -1 | awk '{print $3}')

    echo "${N},${GAZE_TAG},${MEAN},${STD},${BEST_MODEL}" >> "$SUMMARY_CSV"
    echo "Logged: N=${N} ${GAZE_TAG} mean=${MEAN} std=${STD}"
}

# ── GAZE EXPERIMENTS ──────────────────────────────────────────────────────────
echo "=========================================="
echo "  GAZE EXPERIMENTS"
echo "=========================================="
for N in "${EPISODES[@]}"; do
    echo ""
    echo "--- Training: Gaze | N=${N} episodes ---"
    conda run -n nesy-il python -u train_il.py \
        --dataset "$DATASET" \
        --num_episodes "$N" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH" \
        --lr "$LR" \
        --seed "$SEED" \
        --rules "$RULES" \
        --eval_interval 5 \
        --valuation_path "models/nsfr/seaquest/gaze/valuation.pt" \
        --send_email \
        --email_interval 30 \
        --eval_max_steps 3000 \
        --use_gazemap

    run_eval "$N" "gaze" "--use_gazemap"
done

# ── NO-GAZE EXPERIMENTS ───────────────────────────────────────────────────────
# echo ""
# echo "=========================================="
# echo "  NO-GAZE EXPERIMENTS"
# echo "=========================================="
# for N in "${EPISODES[@]}"; do
#     echo ""
#     echo "--- Training: No-Gaze | N=${N} episodes ---"
#     conda run -n nesy-il python -u train_il.py \
#         --dataset "$DATASET" \
#         --num_episodes "$N" \
#         --epochs "$EPOCHS" \
#         --batch_size "$BATCH" \
#         --lr "$LR" \
#         --seed "$SEED" \
#         --rules "$RULES" \
#         --eval_interval 5 \
#         --valuation_path "models/nsfr/seaquest/_no_gaze/valuation.pt" \
#         --eval_max_steps 3000 \
#         --send_email \
#         --email_interval 30

#     run_eval "$N" "no_gaze" ""
# done

echo ""
echo "=========================================="
echo "  SWEEP COMPLETE"
echo "  Summary: $SUMMARY_CSV"
echo "  Models:  models/nsfr/seaquest/{gaze|no_gaze}/{N}_ep/"
echo "=========================================="
