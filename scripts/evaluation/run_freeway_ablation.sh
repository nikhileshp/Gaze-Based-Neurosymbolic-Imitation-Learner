#!/usr/bin/env bash
# Freeway ruleset ablation (GRAIL-faithful max+nll) + one linear+ce appendix run.
# REBOOT-RESILIENT: the box has a watchdog that reboots on Wi-Fi loss, so:
#   - log to a persistent path in $HOME (NOT /tmp, which is wiped on reboot)
#   - skip any (rules,loss) whose final-epoch checkpoint already exists (resume)
#   - checkpoints persist under trained_models/, so re-running continues where it died
#   - tqdm disabled to keep the log small
# Safe to re-run after a reboot; it picks up the remaining work.
set -u
cd /home/nick/Projects/Gaze-Based-Neurosymbolic-Imitation-Learner
PY=~/miniconda3/envs/grail/bin/python
export HSA_OVERRIDE_GFX_VERSION=11.0.0 SDL_VIDEODRIVER=dummy WANDB_MODE=disabled TQDM_DISABLE=1
DATA=data/freeway/Freeway_logicfmt5_obj12.pt
EPOCHS=6
COMMON="--env freeway --dataset $DATA --device cuda --epochs $EPOCHS --batch_size 512 --lr 0.05 \
  --target_diagonal 0.5 --num_eval_episodes 1 --eval_interval $EPOCHS --eval_max_steps 1500 --num_workers 0"
LOG=$HOME/freeway_ablation.log
echo "=== ablation start $(date) ===" >> "$LOG"

done_already () {  # $1=rules $2=loss -> 0 if a final-epoch checkpoint exists
  ls trained_models/freeway/nsfr/${1}_rules_*_lr_${2}_td_0.5/full_ep/*/epoch_${EPOCHS}.pth >/dev/null 2>&1
}

train () {  # $1=rules $2=loss $3=head
  if done_already "$1" "$2"; then
    echo "SKIP (done): rules=$1 loss=$2 head=$3" | tee -a "$LOG"; return
  fi
  echo "===== TRAIN rules=$1 loss=$2 head=$3 $(date) =====" | tee -a "$LOG"
  $PY scripts/training/train_il.py --rules "$1" --loss "$2" --action_head "$3" $COMMON >> "$LOG" 2>&1
  echo "  done rules=$1 rc=$? $(date)" | tee -a "$LOG"
}

train default              nll max
train new                  nll max
train conditional_xclose   nll max
train conditional_approach nll max
train conditional_xclose   ce  linear

echo "===== EVAL (20 full episodes each) $(date) =====" | tee -a "$LOG"
$PY scripts/evaluation/eval_ablation.py \
  default,nll,max new,nll,max conditional_xclose,nll,max conditional_approach,nll,max conditional_xclose,ce,linear >> "$LOG" 2>&1
echo "ABLATION DONE $(date)" | tee -a "$LOG"
