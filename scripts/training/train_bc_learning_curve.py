"""
train_bc_learning_curve.py
==========================
Learning-curve baseline built on top of the GABRIL CNN architecture.
Loads from an existing .pt dataset file.

For each percentage step (10%, 20%, ..., 100%), the script:
  1. Takes the FIRST N% of samples (in dataset order, not shuffled).
  2. Trains for --epochs epochs from scratch.
  3. Evaluates for --eval_episodes episodes.
  4. Saves the model and results in  <run_dir>/<Xp>/

Supported gaze methods (--gaze_method):
  None : Plain BC — no gaze information
  AGIL : Dual CNN: averages encoder(frame) + encoder_agil(frame × gaze)
  Mask : Multiplies pixels by gaze mask before encoding

Usage:
  python scripts/training/train_bc_learning_curve.py \\
    --dataset data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt \\
    --env seaquest --rules new --epochs 20 --eval_episodes 50 \\
    --gaze_method None --seed 42
"""

import os
import sys
import argparse
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import deque
from tqdm import tqdm
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from core.utils.utils import set_seed_everywhere, load_pt_dataset
from core.utils.linear_models import Encoder, weight_init
from nsfr.env import NSFRBaseEnv

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

PERCENTAGES = list(range(10, 110, 10))   # [10, 20, 30, ..., 100]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_obs(obs_batch, device):
    """uint8 (B, 1, H, W) → float32 (B, 1, H, W) normalised to [0,1]."""
    x = obs_batch.float().to(device) / 255.0
    if x.ndim == 3:       # (B, H, W) — add channel dim
        x = x.unsqueeze(1)
    elif x.ndim == 5:     # (B, stack, 1, H, W) — squeeze extra dim
        x = x.squeeze(2)
    return x


def build_model(args, action_dim, device):
    """Construct fresh encoder, pre_actor, actor (and optional encoder_agil) and return them."""
    encoder_out_dim = 8 * 8 * args.embedding_dim   # 4096 for defaults

    encoder = Encoder(args.stack, args.embedding_dim, args.num_hiddens,
                      args.num_residual_layers, args.num_residual_hiddens).to(device)

    pre_actor = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(encoder_out_dim, args.z_dim),
        nn.ReLU()
    )
    pre_actor.apply(weight_init)
    pre_actor.to(device)

    actor = nn.Sequential(
        nn.Linear(args.z_dim, args.z_dim), nn.ReLU(),
        nn.Linear(args.z_dim, action_dim)
    )
    actor.apply(weight_init)
    actor.to(device)

    encoder_agil = None
    if args.gaze_method == "AGIL":
        encoder_agil = Encoder(args.stack, args.embedding_dim, args.num_hiddens,
                               args.num_residual_layers, args.num_residual_hiddens).to(device)

    return encoder, pre_actor, actor, encoder_agil


def evaluate_bc(encoder, pre_actor, actor, env, num_episodes, seed, device,
                gaze_method='None', encoder_agil=None, gaze_predictor=None, id2action=None,
                stack=1):
    """Run the policy for num_episodes, return list of total rewards."""
    dev = torch.device(device)
    encoder.to(dev).eval()
    pre_actor.to(dev).eval()
    actor.to(dev).eval()
    if encoder_agil is not None:
        encoder_agil.to(dev).eval()

    rewards = []
    for i in range(num_episodes):
        try:
            state = env.reset(seed=seed + i)
        except TypeError:
            state = env.reset()

        done, total_r = False, 0.0

        # Encoder frame-stack buffer (stack>1 = higher-order Markov input); padded with frame 0
        enc_buf = deque(maxlen=stack)
        f0_raw = env.get_rgb_frame() if hasattr(env, 'get_rgb_frame') else env.render()
        f0 = cv2.cvtColor(f0_raw, cv2.COLOR_RGB2GRAY)
        f0 = cv2.resize(f0, (84, 84), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        for _ in range(stack):
            enc_buf.append(f0)

        # Initialise gaze temporal buffer
        frame_buffer = None
        if gaze_predictor is not None:
            frame_buffer = deque(maxlen=4)
            raw_frame = env.get_rgb_frame() if hasattr(env, 'get_rgb_frame') else env.render()
            gray_init = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
            gray_init = cv2.resize(gray_init, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
            for _ in range(4):
                frame_buffer.append(gray_init)

        while not done:
            raw_frame = env.get_rgb_frame()          # (H, W, 3) RGB uint8
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            enc_buf.append(gray.astype(np.float32) / 255.0)
            xx = torch.tensor(np.stack(list(enc_buf), axis=0),
                              dtype=torch.float32, device=dev).unsqueeze(0)  # (1, stack, 84, 84)

            gg = torch.zeros(1, 1, 84, 84, device=dev)
            if gaze_predictor is not None:
                img_stack = np.stack(frame_buffer, axis=-1)   # (84, 84, 4)
                inp = torch.tensor(img_stack, dtype=torch.float32,
                                   device=gaze_predictor.device).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    gg = gaze_predictor.predict_normalized(inp).to(dev)

            with torch.no_grad():
                xx_in = xx * gg if gaze_method == 'Mask' else xx
                z = encoder(xx_in)
                if gaze_method == 'AGIL' and encoder_agil is not None:
                    z = (z + encoder_agil(xx * gg)) / 2
                logits = actor(pre_actor(z))
                action_idx = logits.argmax(dim=1).item()

            action_str = id2action[action_idx]
            state, reward, done = env.step(action_str)
            total_r += reward

            if gaze_predictor is not None and not done:
                next_raw = env.get_rgb_frame() if hasattr(env, 'get_rgb_frame') else env.render()
                next_gray = cv2.cvtColor(next_raw, cv2.COLOR_RGB2GRAY)
                next_gray = cv2.resize(next_gray, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
                frame_buffer.append(next_gray)

        rewards.append(total_r)
        print(f"    Episode {i+1}/{num_episodes}: {total_r:.0f}")

    return rewards


# ═══════════════════════════════════════════════════════════════════════════════
# Args
# ═══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description="BC Learning-Curve Baseline (GABRIL CNN) — supports None / AGIL / Mask"
    )
    # Dataset
    p.add_argument("--dataset",      type=str, required=True,
                   help="Path to the .pt dataset file (used for ALL percentage steps).")
    p.add_argument("--env",          type=str, default="seaquest")
    p.add_argument("--rules",        type=str, default="new")
    # Training
    p.add_argument("--epochs",       type=int, default=20)
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--num_workers",  type=int, default=0)
    # Model
    p.add_argument("--embedding_dim",          type=int, default=64)
    p.add_argument("--num_hiddens",            type=int, default=128)
    p.add_argument("--num_residual_layers",    type=int, default=2)
    p.add_argument("--num_residual_hiddens",   type=int, default=32)
    p.add_argument("--z_dim",                  type=int, default=256)
    p.add_argument("--stack",                  type=int, default=1,
                   help="Stacked frames as encoder input channels (4 = higher-order Markov for OOD speed).")
    # Gaze method
    p.add_argument("--gaze_method",  type=str, default="None",
                   choices=["None", "AGIL", "Mask"])
    p.add_argument("--use_gaze",     action="store_true",
                   help="Pipe live gaze predictions into the agent during evaluation.")
    p.add_argument("--gaze_model_path", type=str,
                   default="gaze_models/seaquest/seaquest_gaze_predictor_2.pth")
    # Evaluation
    p.add_argument("--eval_episodes", type=int, default=50,
                   help="Number of episodes to evaluate after each training run.")
    # Output
    p.add_argument("--run_dir",      type=str, default=None,
                   help="Base output directory.")
    p.add_argument("--device",       type=str, default="cuda")
    # Optional: restrict which percentages to run
    p.add_argument("--percentages",  type=int, nargs="+", default=PERCENTAGES,
                   help="Which percentages to train on, e.g. --percentages 10 20 50 100")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    set_seed_everywhere(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Gaze method: {args.gaze_method}")

    use_gaze_data = args.use_gaze or args.gaze_method in ["AGIL", "Mask"]

    # ── Optional live gaze predictor for evaluation ───────────────────────────
    gaze_predictor = None
    if use_gaze_data and args.gaze_method != "None":
        try:
            from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
            print(f"Initializing Gaze Predictor from {args.gaze_model_path} ...")
            gaze_predictor = Human_Gaze_Predictor(args.env)
            gaze_predictor.init_model(args.gaze_model_path)
            gaze_predictor.model.eval()
        except ImportError:
            print("Warning: Could not import Human_Gaze_Predictor. Gaze will be zero.")

    # ── Output directory ──────────────────────────────────────────────────────
    gaze_tag = args.gaze_method.lower()
    model_name = gaze_tag if gaze_tag != 'none' else 'bc'
    if args.run_dir:
        base_run_dir = args.run_dir
    else:
        stack_tag = f"_stack{args.stack}" if args.stack > 1 else ""
        base_run_dir = (
            f"trained_models/{args.env}/{model_name}{stack_tag}_learning_curve"
            f"_epoch_{args.epochs}_seed_{args.seed}_lr_{args.lr}"
        )
    os.makedirs(base_run_dir, exist_ok=True)

    # ── Environment (shared across all runs) ──────────────────────────────────
    env = NSFRBaseEnv.from_name(args.env, mode='logic')
    
    # Deriving action mapping from environment
    id2action = {v: k for k, v in env.pred2action.items()}
    print(f"Action mapping derived from Env: {id2action}")

    # ── Load FULL dataset once to determine total number of samples ───────────
    print("\nLoading full dataset to determine total sample count ...")
    obs_full, actions_full, gaze_full, _, _ = load_pt_dataset(
        args.dataset, num_episodes=None, use_gaze=use_gaze_data, stack=args.stack
    )
    total_samples = len(obs_full)
    action_dim = int(actions_full.max() + 1)
    print(f"Total samples in dataset: {total_samples} | Action Dim: {action_dim}")

    # Check if we have string mappings for all actions predicted by the model
    for a_idx in range(action_dim):
        if a_idx not in id2action:
            print(f"Warning: Action index {a_idx} from dataset not found in {args.env} pred2action!")
            id2action[a_idx] = 'noop' # Fallback

    # ── Results accumulator ───────────────────────────────────────────────────
    results_log = []

    # ── Learning-curve loop ───────────────────────────────────────────────────
    for pct in args.percentages:
        n_samples = max(1, int(total_samples * pct / 100))
        folder_name = f"{pct}p"
        run_dir = os.path.join(base_run_dir, folder_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {pct}% of dataset  →  {n_samples} / {total_samples} samples")
        print(f"  Output: {run_dir}")
        print(f"{'='*60}")

        # ── Slice the first N samples (in order, no shuffle) ──────────────────
        obs     = obs_full[:n_samples]
        actions = actions_full[:n_samples]
        gaze    = gaze_full[:n_samples]   # zeros when use_gaze_data is False

        # ── 95/5 train/val split (shuffled within the slice) ──────────────────
        idx   = list(range(n_samples))
        random.shuffle(idx)
        split = max(1, int(0.95 * n_samples))
        tr_idx, va_idx = idx[:split], idx[split:]

        tr_ds = TensorDataset(obs[tr_idx], actions[tr_idx], gaze[tr_idx])
        va_ds = TensorDataset(obs[va_idx], actions[va_idx], gaze[va_idx])
        tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

        # ── Fresh model for each percentage ───────────────────────────────────
        set_seed_everywhere(args.seed)   # reset RNG so each run is comparable
        encoder, pre_actor, actor, encoder_agil = build_model(args, action_dim, device)

        params = (list(encoder.parameters()) + list(pre_actor.parameters()) +
                  list(actor.parameters()))
        if encoder_agil is not None:
            params += list(encoder_agil.parameters())

        optimizer = torch.optim.Adam(params, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc = -float('inf')
        epoch_log = []   # per-epoch metrics -> training_log.csv (loss/acc curves)

        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in range(args.epochs):
            encoder.train(); pre_actor.train(); actor.train()
            if encoder_agil is not None: encoder_agil.train()

            total_loss, total_correct, total_n = 0.0, 0, 0
            pbar = tqdm(tr_dl, desc=f"[{pct}%] Epoch {epoch+1}/{args.epochs}", leave=False)
            for xx_raw, aa, gg in pbar:
                xx = preprocess_obs(xx_raw, device)
                aa = aa.to(device)
                gg = gg.to(device)

                optimizer.zero_grad()

                if args.gaze_method == "Mask":
                    xx_in = xx * gg
                else:
                    xx_in = xx

                z = encoder(xx_in)
                if args.gaze_method == "AGIL" and encoder_agil is not None:
                    z = (z + encoder_agil(xx * gg)) / 2

                logits = actor(pre_actor(z))
                loss   = criterion(logits, aa)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                total_loss    += loss.item() * aa.size(0)
                total_correct += (logits.argmax(1) == aa).sum().item()
                total_n       += aa.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Validation
            encoder.eval(); pre_actor.eval(); actor.eval()
            if encoder_agil is not None: encoder_agil.eval()
            val_correct, val_n, val_loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for xx_raw, aa, gg in va_dl:
                    xx = preprocess_obs(xx_raw, device)
                    aa, gg = aa.to(device), gg.to(device)
                    xx_in = xx * gg if args.gaze_method == "Mask" else xx
                    z = encoder(xx_in)
                    if args.gaze_method == "AGIL" and encoder_agil is not None:
                        z = (z + encoder_agil(xx * gg)) / 2
                    logits = actor(pre_actor(z))
                    val_loss_sum += criterion(logits, aa).item() * aa.size(0)
                    val_correct += (logits.argmax(1) == aa).sum().item()
                    val_n += aa.size(0)

            val_acc   = val_correct / val_n if val_n > 0 else 0.0
            val_loss  = val_loss_sum / val_n if val_n > 0 else 0.0
            avg_loss  = total_loss  / total_n if total_n > 0 else 0.0
            train_acc = total_correct / total_n if total_n > 0 else 0.0

            scheduler.step(val_acc)
            print(f"  [{pct}%] Epoch {epoch+1}/{args.epochs} | "
                  f"Loss {avg_loss:.4f} | TrainAcc {train_acc:.3f} | "
                  f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.3f}")

            # Per-epoch curve (loss + acc), matches the NSFR/grail training_log format
            epoch_log.append({
                'percentage': pct,
                'n_samples':  n_samples,
                'epoch':      epoch + 1,
                'train_loss': avg_loss,
                'train_acc':  train_acc,
                'val_loss':   val_loss,
                'val_acc':    val_acc,
            })

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(encoder.state_dict(),   f"{run_dir}/best_encoder.pth")
                torch.save(pre_actor.state_dict(), f"{run_dir}/best_pre_actor.pth")
                torch.save(actor.state_dict(),     f"{run_dir}/best_actor.pth")
                if encoder_agil is not None:
                    torch.save(encoder_agil.state_dict(), f"{run_dir}/best_encoder_agil.pth")
                print(f"    *** New best model saved (ValAcc: {val_acc:.3f})")

        # Save per-epoch training curve (train/val loss + acc) for this percentage
        pd.DataFrame(epoch_log).to_csv(f"{run_dir}/training_log.csv", index=False)

        # Save final epoch checkpoint
        torch.save(encoder.state_dict(),   f"{run_dir}/final_encoder.pth")
        torch.save(pre_actor.state_dict(), f"{run_dir}/final_pre_actor.pth")
        torch.save(actor.state_dict(),     f"{run_dir}/final_actor.pth")
        if encoder_agil is not None:
            torch.save(encoder_agil.state_dict(), f"{run_dir}/final_encoder_agil.pth")

        # ── Evaluate for eval_episodes episodes using BEST model ──────────────
        print(f"\n  Evaluating best model for {args.eval_episodes} episodes ...")
        # Load best weights back
        encoder.load_state_dict(torch.load(f"{run_dir}/best_encoder.pth", map_location=device))
        pre_actor.load_state_dict(torch.load(f"{run_dir}/best_pre_actor.pth", map_location=device))
        actor.load_state_dict(torch.load(f"{run_dir}/best_actor.pth", map_location=device))
        if encoder_agil is not None:
            agil_ckpt = f"{run_dir}/best_encoder_agil.pth"
            if os.path.exists(agil_ckpt):
                encoder_agil.load_state_dict(torch.load(agil_ckpt, map_location=device))

        rewards   = evaluate_bc(encoder, pre_actor, actor, env,
                                num_episodes=args.eval_episodes,
                                seed=args.seed, device=str(device),
                                gaze_method=args.gaze_method,
                                encoder_agil=encoder_agil,
                                gaze_predictor=gaze_predictor,
                                id2action=id2action, stack=args.stack)
        mean_r    = np.mean(rewards)
        std_r     = np.std(rewards)
        median_r  = float(np.median(rewards))
        print(f"\n  [{pct}%] Eval → Mean: {mean_r:.2f} ± {std_r:.2f}  |  Median: {median_r:.2f}")

        results_log.append({
            'percentage':    pct,
            'n_samples':     n_samples,
            'best_val_acc':  best_val_acc,
            'mean_reward':   mean_r,
            'std_reward':    std_r,
            'median_reward': median_r,
            'all_rewards':   rewards,
        })

        # Save per-percentage results immediately (so partial runs are preserved)
        per_pct_csv = os.path.join(run_dir, "eval_results.csv")
        pd.DataFrame([{k: v for k, v in results_log[-1].items() if k != 'all_rewards'}]).to_csv(
            per_pct_csv, index=False
        )

        # Free GPU memory before next run
        del encoder, pre_actor, actor, params, optimizer, scheduler
        if encoder_agil is not None:
            del encoder_agil
        del tr_ds, va_ds, tr_dl, va_dl, gaze
        torch.cuda.empty_cache()
        gc.collect()

    # ── Save aggregate results CSV ────────────────────────────────────────────
    summary_rows = [{k: v for k, v in r.items() if k != 'all_rewards'} for r in results_log]
    df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(base_run_dir, "learning_curve_summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"\n{'='*60}")
    print("Learning Curve Summary:")
    print(df.to_string(index=False))
    print(f"\nSummary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
