"""
train_gaze.py
==============
Train a human gaze heatmap predictor (HumanGazeNet or FactGazeNet) and save
the checkpoint to:

    trained_models/gaze_predictor/<env>/<model_name>.pth

Wraps the Human_Gaze_Predictor class from scripts/gaze/gaze_predictor.py and
the data loaders from core/utils/utils.py.

Supported input modes:
  --dataset         .pt dataset file (frames + gaze heatmaps)
  --use_facts       Train FactGazeNet from pre-computed NSFR atom vectors
                    (requires --facts_dataset and --dataset for gaze targets)

Usage (visual model):
  python scripts/training/train_gaze.py \\
      --env seaquest \\
      --dataset data/seaquest/dataset.pt \\
      --epochs 30 --batch_size 64 --lr 1.0

Usage (fact-based model):
  python scripts/training/train_gaze.py \\
      --env seaquest \\
      --dataset data/seaquest/dataset.pt \\
      --use_facts \\
      --facts_dataset trained_models/nsfr/seaquest/no_gaze/full_ep/valuation.pt \\
      --epochs 30
"""

import os
import argparse
import torch

from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
from core.utils.utils import (
    load_gaze_predictor_data,
    load_fact_gaze_predictor_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Train a gaze heatmap predictor.")

    # Dataset
    parser.add_argument("--env", type=str, required=True,
                        help="Game / environment name (e.g. seaquest). Used for the save path.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to .pt dataset file.")
    parser.add_argument("--use_facts", action="store_true",
                        help="Train FactGazeNet from NSFR atom vectors instead of raw frames.")
    parser.add_argument("--facts_dataset", type=str, default=None,
                        help="Path to .pkl / .pt atom probabilities (required with --use_facts).")
    parser.add_argument("--frame_stack", type=int, default=4,
                        help="Number of frames stacked as one input sample (default: 4).")

    # Model
    parser.add_argument("--model_weights", type=str, default=None,
                        help="Path to a .pth checkpoint to resume training from.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Custom filename for the saved checkpoint (without .pth). "
                             "Defaults to '<env>_gaze_predictor' or '<env>_fact_gaze_predictor'.")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # Logging
    parser.add_argument("--log_csv", type=str, default=None,
                        help="CSV path to log (epoch, avg_loss) per epoch.")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    if args.use_facts and not args.facts_dataset:
        raise ValueError("--facts_dataset is required when using --use_facts.")

    # ── Determine output path ─────────────────────────────────────────────────
    model_type = "fact_gaze_predictor" if args.use_facts else "gaze_predictor"
    model_name = args.model_name or f"{args.env}_{model_type}"
    out_dir = os.path.join("trained_models", "gaze_predictor", args.env)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}.pth")

    print(f"{'='*60}")
    print(f"  Gaze predictor training")
    print(f"  Env      : {args.env}")
    print(f"  Model    : {model_type}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Epochs   : {args.epochs}  |  Batch size: {args.batch_size}  |  LR: {args.lr}")
    print(f"  Save to  : {out_path}")
    print(f"{'='*60}\n")

    # ── Initialise predictor ──────────────────────────────────────────────────
    gp = Human_Gaze_Predictor(args.env)
    gp.init_model(
        gaze_model_file=args.model_weights,
        lr=args.lr,
        rho=args.rho,
        weight_decay=args.weight_decay,
        use_facts=args.use_facts,
    )
    gp.log_csv = args.log_csv

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.use_facts:
        print(f"Loading fact-based gaze data from {args.facts_dataset} + {args.dataset} ...")
        imgs, gaze_masks, valid_indices = load_fact_gaze_predictor_data(
            args.facts_dataset, args.dataset, frame_stack=args.frame_stack, device="cpu"
        )
    else:
        print(f"Loading visual gaze data from {args.dataset} ...")
        imgs, gaze_masks, valid_indices = load_gaze_predictor_data(
            args.dataset, frame_stack=args.frame_stack, device="cpu"
        )

    print(f"  Frames   : {len(imgs)}  |  Gaze masks: {len(gaze_masks)}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    gp.train_model(imgs, gaze_masks, valid_indices,
                   epochs=args.epochs, batch_size=args.batch_size)

    # ── Save to canonical path ─────────────────────────────────────────────────
    torch.save(gp.model.state_dict(), out_path)
    print(f"\nCheckpoint saved to {out_path}")


if __name__ == "__main__":
    main()
