"""
train_bc_pt.py
==============
Behavior Cloning (BC) baselines using GABRIL's CNN architecture, adapted to
load from the existing .pt dataset files.

Supported methods (--gaze_method):
  None   : Plain BC — no gaze information
  AGIL   : Dual CNN: averages encoder(frame) + encoder_agil(frame × gaze)
  Mask   : Multiplies pixels by gaze mask before encoding

Results are saved to:
  models/bc/{gaze_method}/{N}_ep/  (checkpoints + results CSV)

Usage:
  python train_bc_pt.py \\
    --dataset data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt \\
    --num_episodes 4 --epochs 20 --gaze_method None --seed 42
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    from scripts.email_me import send_email
except ImportError:
    send_email = None

# ── GABRIL CNN models (copied to baselines/) ──────────────────────────────────
from baselines.models.linear_models import Encoder, weight_init

# ── NUDGE environment for evaluation (same as NSFR) ──────────────────────────
from nudge.env import NudgeBaseEnv
from nudge.utils import make_deterministic


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

PRIMITIVE_ACTIONS = {0: 'noop', 1: 'fire', 2: 'up', 3: 'right', 4: 'left', 5: 'down'}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    make_deterministic(seed)


def load_pt_dataset(pt_path, num_episodes=None, use_gaze=False):
    """Load .pt file and return tensors suitable for BC training."""
    print(f"Loading dataset from {pt_path} ...")
    data = torch.load(pt_path, map_location='cpu', weights_only=False)

    obs      = data['observations']            # (N, H, W) uint8
    actions  = data['actions']                 # (N,)
    ep_nums  = data.get('episode_number', None)
    gaze     = data.get('gaze_image', None)    # (N, 84, 84) float32

    if not isinstance(obs, torch.Tensor):      obs = torch.from_numpy(obs)
    if not isinstance(actions, torch.Tensor):  actions = torch.from_numpy(actions)
    if ep_nums is not None and not isinstance(ep_nums, torch.Tensor):
        ep_nums = torch.from_numpy(ep_nums)
    if gaze is not None and not isinstance(gaze, torch.Tensor):
        gaze = torch.from_numpy(gaze)

    actions = actions.long()
    obs     = obs.byte()   # keep as uint8 to save RAM; cast to float in loop

    # Filter to supported actions (0-5)
    mask = (actions <= 5)
    obs, actions = obs[mask], actions[mask]
    if ep_nums is not None: ep_nums = ep_nums[mask]
    if gaze is not None:    gaze    = gaze[mask]

    # Select first num_episodes episodes
    if num_episodes is not None and ep_nums is not None:
        unique_eps = torch.unique(ep_nums)[:num_episodes]
        ep_mask = torch.isin(ep_nums, unique_eps)
        obs, actions = obs[ep_mask], actions[ep_mask]
        if gaze is not None: gaze = gaze[ep_mask]
        print(f"  Using {num_episodes} episodes → {len(actions)} samples")
    else:
        print(f"  Total samples: {len(actions)}")

    if not use_gaze or gaze is None:
        gaze = torch.zeros(len(obs), 1, 84, 84)   # dummy
    else:
        gaze = gaze.float().unsqueeze(1)           # (N, 1, H, W)

    return obs, actions, gaze


def preprocess_obs(obs_batch, device):
    """uint8 (B, H, W) → float32 (B, 1, H, W) normalised to [0, 1]."""
    return obs_batch.float().unsqueeze(1).to(device) / 255.0


def evaluate_bc(encoder, pre_actor, actor, env, num_episodes=10, seed=42,
                gaze_method='None', encoder_agil=None, device='cuda', gaze_predictor=None):
    """Run policy in env for num_episodes, return list of total rewards."""
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
        done, total_r, step = False, 0.0, 0
        
        # Initialize Gaze temporal buffer
        frame_buffer = None
        if gaze_predictor is not None:
            frame_buffer = deque(maxlen=4)
            raw_frame = env.get_rgb_frame() if hasattr(env, 'get_rgb_frame') else (env.render() if hasattr(env, 'render') else state)
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
            for _ in range(4): frame_buffer.append(gray)

        while not done:
            # NudgeBaseEnv returns (logic_state, neural_state) in 'logic' mode
            # We want the raw frame for pixel-based BC.
            raw_frame = env.get_rgb_frame()  # (H, W, 3) RGB uint8

            # Grayscale + resize to 84x84 + normalise
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            xx = torch.tensor(gray, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(0) / 255.0
            
            gg = torch.zeros(1, 1, 84, 84, device=dev)
            if gaze_predictor is not None:
                img_stack = np.stack(frame_buffer, axis=-1) # (84, 84, 4)
                input_tensor = torch.tensor(img_stack, dtype=torch.float32, device=gaze_predictor.device).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    gaze_pred = gaze_predictor.model(input_tensor)
                gg = gaze_pred.to(dev)

            with torch.no_grad():
                if gaze_method == 'Mask':
                    xx_in = xx * gg
                else:
                    xx_in = xx
                z = encoder(xx_in)
                if gaze_method == 'AGIL' and encoder_agil is not None:
                    z = (z + encoder_agil(xx * gg)) / 2
                logits = actor(pre_actor(z))
                action_idx = logits.argmax(dim=1).item()

            action_str = PRIMITIVE_ACTIONS[action_idx]
            state, reward, done = env.step(action_str)
            total_r += reward
            step += 1
            
            # Update frame buffer for gaze with the new environment state
            if gaze_predictor is not None and not done:
                next_raw = env.get_rgb_frame() if hasattr(env, 'get_rgb_frame') else (env.render() if hasattr(env, 'render') else state)
                next_gray = cv2.cvtColor(next_raw, cv2.COLOR_RGB2GRAY)
                next_gray = cv2.resize(next_gray, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
                frame_buffer.append(next_gray)

        rewards.append(total_r)
        print(f"  Episode {i+1}: {total_r:.0f}")

    return rewards


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser(description="BC Baseline (GABRIL CNN) on .pt dataset")
    # Dataset
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--env", type=str, default="seaquest")
    parser.add_argument("--rules", type=str, default="new")
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    # Model
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--num_residual_layers", type=int, default=2)
    parser.add_argument("--num_residual_hiddens", type=int, default=32)
    parser.add_argument("--z_dim", type=int, default=256) #FOR MLP
    # Gaze method
    parser.add_argument("--gaze_method", type=str, default="None",
                        choices=["None", "AGIL", "Mask"])
    # Incremental mode
    parser.add_argument("--incremental", action="store_true", help="Train 1 epoch per episode iteratively for all episodes in dataset.")
    # Evaluation
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_gazemap", action="store_true", help="Pipe live 84x84 gaze predictions into logic agent during testing")
    parser.add_argument("--gaze_model_path", type=str, default="seaquest_gaze_predictor_2.pth")
    parser.add_argument("--send_email", action="store_true", help="Send email with results after evaluation")
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Gaze method: {args.gaze_method}")

    use_gaze = args.gaze_method in ["AGIL", "Mask"]
    num_ep   = args.num_episodes

    gaze_predictor = None
    if (args.use_gazemap or args.gaze_method in ['ViSaRL', 'Mask', 'AGIL']) and args.gaze_method != "None":
        try:
            from scripts.gaze_predictor import Human_Gaze_Predictor
            print(f"Initializing Test-Time Gaze Predictor from {args.gaze_model_path}...")
            gaze_predictor = Human_Gaze_Predictor(args.env)
            gaze_predictor.init_model(args.gaze_model_path)
            gaze_predictor.model.eval()
        except ImportError:
            print("Warning: Could not import Human_Gaze_Predictor! Gaze will drop to 0.0.")

    if not args.incremental:
        # Shuffle + 95/5 train/val split is done inside the loop for conventional mode now
        pass

    # ── Dataset Loading ───────────────────────────────────────────────────────
    # Peek at dataset to get action dimension dynamically
    data_peek = torch.load(args.dataset, map_location='cpu', weights_only=False)
    actions = data_peek['actions']
    if not isinstance(actions, torch.Tensor): actions = torch.from_numpy(actions)
    action_dim = int(actions.max().item()) + 1
    print(f"Inferred action_dim from dataset: {action_dim}")
    del data_peek # free memory

    encoder_out_dim  = 8 * 8 * args.embedding_dim  # → 4096 for default settings
    
    encoder = Encoder(1, args.embedding_dim, args.num_hiddens,
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
        encoder_agil = Encoder(1, args.embedding_dim, args.num_hiddens,
                               args.num_residual_layers, args.num_residual_hiddens).to(device)

    params = list(encoder.parameters()) + list(pre_actor.parameters()) + list(actor.parameters())
    if encoder_agil is not None:
        params += list(encoder_agil.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # ── Output dirs ───────────────────────────────────────────────────────────
    gaze_tag  = args.gaze_method.lower()
    base_run_dir = f"models/bc/incremental_{gaze_tag}" if args.incremental else f"models/bc/{gaze_tag}"
    os.makedirs(base_run_dir, exist_ok=True)

    results_log = []
    # ── Environment for evaluation ────────────────────────────────────────────
    env = NudgeBaseEnv.from_name(args.env, mode='logic')

    # Determine episodes to loop over
    if args.incremental:
        # We need to know max episodes. We can peek at the dataset.
        data_peek = torch.load(args.dataset, map_location='cpu', weights_only=False)
        episodes = data_peek.get('episode_number', None)
        if episodes is not None:
            if not isinstance(episodes, torch.Tensor): episodes = torch.from_numpy(episodes)
            max_ep = len(torch.unique(episodes))
        else:
            max_ep = 1
        episodes_to_train = list(range(1, max_ep + 1))
        epochs_per_episode = 1
        print(f"Incremental Mode: Training sequentially on {max_ep} episodes (1 epoch each)")
        del data_peek
        import gc
        gc.collect()
    else:
        episodes_to_train = [args.num_episodes] if args.num_episodes else ["all"]
        epochs_per_episode = args.epochs

    best_mean_global = -float('inf')

    for current_ep in episodes_to_train:
        # Load dataset up to current_ep if incremental, or up to num_episodes
        target_ep_load = current_ep if args.incremental else num_ep

        print(f"\n=======================================================")
        if args.incremental:
            print(f"  Training on episode: {current_ep} (Total cumulative episodes shown to model: {current_ep})")
        else:
            print(f"  Training conventional mode for {epochs_per_episode} epochs")
        print(f"=======================================================")

        obs, actions, gaze = load_pt_dataset(args.dataset, num_episodes=target_ep_load if target_ep_load != "all" else None, use_gaze=use_gaze)

        # Shuffle + 95/5 train/val split for the current accumulative dataset
        idx = list(range(len(obs)))
        random.shuffle(idx)
        split = int(0.95 * len(idx))
        tr_idx, va_idx = idx[:split], idx[split:]

        tr_ds = TensorDataset(obs[tr_idx], actions[tr_idx], gaze[tr_idx])
        va_ds = TensorDataset(obs[va_idx], actions[va_idx], gaze[va_idx])
        tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        run_dir = f"{base_run_dir}/{current_ep}_ep"
        os.makedirs(run_dir, exist_ok=True)
        
        best_mean  = -float('inf')

        # ── Training loop ─────────────────────────────────────────────────────────
        for epoch in range(epochs_per_episode):
            encoder.train(); pre_actor.train(); actor.train()
            if encoder_agil is not None: encoder_agil.train()

            total_loss, total_correct, total_n = 0.0, 0, 0
            pbar = tqdm(tr_dl, desc=f"Ep {current_ep} Epoch {epoch+1}/{epochs_per_episode}", leave=False)
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

            scheduler.step()
            avg_loss = total_loss / total_n if total_n > 0 else 0
            train_acc = total_correct / total_n if total_n > 0 else 0

            # Validation accuracy
            encoder.eval(); pre_actor.eval(); actor.eval()
            if encoder_agil is not None: encoder_agil.eval()
            val_correct, val_n = 0, 0
            with torch.no_grad():
                for xx_raw, aa, gg in va_dl:
                    xx = preprocess_obs(xx_raw, device)
                    aa, gg = aa.to(device), gg.to(device)
                    xx_in = xx * gg if args.gaze_method == "Mask" else xx
                    z = encoder(xx_in)
                    if args.gaze_method == "AGIL" and encoder_agil is not None:
                        z = (z + encoder_agil(xx * gg)) / 2
                    logits = actor(pre_actor(z))
                    val_correct += (logits.argmax(1) == aa).sum().item()
                    val_n += aa.size(0)
            val_acc = val_correct / val_n if val_n > 0 else 0

            # Save checkpoint for this specific epoch
            epoch_id = epoch + 1 if not args.incremental else current_ep
            torch.save(encoder.state_dict(),   f"{run_dir}/epoch_{epoch_id}_encoder.pth")
            torch.save(pre_actor.state_dict(), f"{run_dir}/epoch_{epoch_id}_pre_actor.pth")
            torch.save(actor.state_dict(),     f"{run_dir}/epoch_{epoch_id}_actor.pth")

        # Environment evaluation after training phases for current_ep complete
        rewards = evaluate_bc(encoder, pre_actor, actor, env,
                               num_episodes=args.eval_episodes, seed=args.seed,
                               gaze_method=args.gaze_method, encoder_agil=encoder_agil,
                               device=str(device), gaze_predictor=gaze_predictor)
        mean_r, std_r = np.mean(rewards), np.std(rewards)
        print(f"\nFinal Eval for N={current_ep} | TrainLoss {avg_loss:.4f} | TrainAcc {train_acc:.3f} | ValAcc {val_acc:.3f} | MeanR {mean_r:.2f} | StdR {std_r:.2f}")

        results_log.append({
            'num_episodes': current_ep if args.incremental else num_ep,
            'gaze_method': args.gaze_method,
            'train_loss': avg_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'mean_reward': mean_r,
            'std_reward': std_r,
        })

        if mean_r > best_mean_global:
            best_mean_global = mean_r

        # Always explicitly save best as the final in incremental
        torch.save(encoder.state_dict(),   f"{run_dir}/best_encoder.pth")
        torch.save(pre_actor.state_dict(), f"{run_dir}/best_pre_actor.pth")
        torch.save(actor.state_dict(),     f"{run_dir}/best_actor.pth")
        if encoder_agil is not None:
            torch.save(encoder_agil.state_dict(), f"{run_dir}/best_encoder_agil.pth")
            
        print(f"  *** Checkpoints saved to {run_dir}")

    # ── Save results CSV ──────────────────────────────────────────────────────
    df = pd.DataFrame(results_log)
    csv_path = f"{base_run_dir}/results_incremental_lr_{args.lr}.csv" if args.incremental else f"{base_run_dir}/{num_ep}_ep/results_lr_{args.lr}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Best global eval score: {best_mean_global:.2f}")

    if args.send_email and send_email is not None:
        subject = f"BC sweep complete: {args.gaze_method}"
        body = f"""
Sweep complete!
Gaze method: {args.gaze_method}
Best global eval score: {best_mean_global:.2f}
Results saved to: {csv_path}
"""
        try:
            send_email(subject, body.strip())
            print("Email notification sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")


if __name__ == "__main__":
    main()
