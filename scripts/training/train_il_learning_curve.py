"""
train_il_learning_curve.py
==========================
Learning-curve baseline for the NSFR Imitation Learning agent (ImitationAgent).
Mirrors train_il.py logic, adapted to sweep over increasing data fractions.

For each percentage step (10%, 20%, ..., 100%), the script:
  1. Takes the FIRST N% of samples (in dataset order, no shuffle).
  2. Trains a fresh ImitationAgent for --epochs epochs from scratch.
  3. Evaluates for --eval_episodes episodes.
  4. Saves model + results under <run_dir>/<Xp>/

The pre-computed valuation file (--valuation_path or auto-discovered) is
shared across all percentage steps — only the training rows that fall in the
first N% slice are used.

Usage:
  python scripts/training/train_il_learning_curve.py \\
    --dataset data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt \\
    --env seaquest --rules new \\
    --epochs 20 --eval_episodes 50 --seed 42
"""

import os
os.environ['ALE_PY_QUIET'] = '1'

import argparse
import datetime
import gc
import signal
import logging
import time

logging.getLogger("ale_py").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, random_split
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from core.utils.utils import (
    get_primitive_action_map, PtDataset,
    set_seed_everywhere, send_run_update, load_pt_dataset
)
from nsfr.agents.imitation_agent import ImitationAgent
from nsfr.utils import make_deterministic
from nsfr.env import NSFRBaseEnv
from scripts.evaluation.evaluate_model import evaluate, evaluate_parallel

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

PERCENTAGES = list(range(10, 110, 10))   # [10, 20, 30, ..., 100]


# ═══════════════════════════════════════════════════════════════════════════════
# Args
# ═══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description="NSFR IL Learning-Curve: train on 10%..100% of data, evaluate after each run."
    )
    # Dataset
    p.add_argument("--dataset",         type=str, required=True,
                   help="Path to the .pt dataset file.")
    p.add_argument("--env",             type=str, default=None)
    p.add_argument("--rules",           type=str, default="new")
    # Training
    p.add_argument("--epochs",          type=int, default=20)
    p.add_argument("--batch_size",      type=int, default=32)
    p.add_argument("--lr",              type=float, default=0.01)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--val_split",       type=float, default=0.05,
                   help="Fraction of each slice held out for validation.")
    p.add_argument("--lr_patience",     type=int, default=3)
    p.add_argument("--loss",            type=str, default="nll",
                   choices=["nll", "bce"])
    p.add_argument("--aggregation",     type=str, default="max",
                   choices=["softor", "max"])
    p.add_argument("--target_diagonal", type=float, default=0.99,
                   help="Initial confidence for clauses in block diagonal initialization.")
    # Gaze
    p.add_argument("--use_gaze",        action="store_true")
    p.add_argument("--gaze_threshold",  type=float, default=50.0)
    p.add_argument("--gaze_model_path", type=str,
                   default="gaze_models/seaquest/seaquest_gaze_predictor_2.pth")
    p.add_argument("--unnormalized",    action="store_true",
                   help="Use unnormalized gaze probabilities.")
    p.add_argument("--visible_preds_only", action="store_true")
    p.add_argument("--alpha",           type=float, default=None)
    # Valuations
    p.add_argument("--valuation_path",  type=str, default=None,
                   help="Path to pre-computed {rules}_valuation.pt (shared for all slices).")
    # Evaluation
    p.add_argument("--eval_episodes",   type=int, default=50)
    p.add_argument("--eval_max_steps",  type=int, default=10000)
    # Output
    p.add_argument("--run_dir",         type=str, default=None,
                   help="Base output directory.")
    p.add_argument("--device",          type=str, default="cpu")
    # Optional: restrict which percentages to run
    p.add_argument("--percentages",     type=int, nargs="+", default=PERCENTAGES,
                   help="Percentages to sweep, e.g. --percentages 10 50 100")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_valuations(v_path, agent, device):
    """Load pre-computed valuations dict from a .pt file (same format as train_il.py)."""
    if not os.path.exists(v_path):
        return None
    print(f"Loading pre-computed valuations from {v_path} ...")
    valuations_raw = torch.load(v_path, map_location=device, weights_only=False)

    if (isinstance(valuations_raw, dict)
            and 'data' in valuations_raw
            and isinstance(valuations_raw['data'], list)):
        print("  Reformatting valuations from list-of-dicts to episode-based dict ...")
        valuations_indexed = {}
        for item in valuations_raw['data']:
            frame_id = item['frame_id']
            try:
                parts    = frame_id.split('_')
                ep_id    = int(parts[1])
                step_idx = int(parts[3])
                if ep_id not in valuations_indexed:
                    valuations_indexed[ep_id] = {}
                atoms = item['atoms']
                if not isinstance(atoms, torch.Tensor):
                    atoms = torch.tensor(atoms, dtype=torch.float32)
                valuations_indexed[ep_id][step_idx] = atoms.to(device)
            except (IndexError, ValueError):
                continue

        valuations = {}
        for ep_id, steps in valuations_indexed.items():
            max_step = max(steps.keys())
            v_list   = [torch.zeros(len(agent.unwrapped_model.atoms)).to(device)] * (max_step + 1)
            for s_idx, v in steps.items():
                v_list[s_idx] = v
            valuations[ep_id] = v_list
        print(f"  Loaded valuations for {len(valuations)} episodes.")
    else:
        valuations = valuations_raw

    return valuations


def build_vT_batch(ep_nums, step_idxs, valuations, agent, device):
    """Assemble a vT tensor from the pre-computed valuation dict for one batch."""
    if valuations is None:
        return None
    vT_list = []
    for ep_id, s_idx in zip(ep_nums.tolist(), step_idxs.tolist()):
        if ep_id in valuations and s_idx < len(valuations[ep_id]):
            vT_list.append(valuations[ep_id][s_idx])
        else:
            vT_list.append(torch.zeros(len(agent.unwrapped_model.atoms), device=device))
    return torch.stack(vT_list).to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    if args.env is None:
        raise ValueError("--env must be specified")

    # ── Gaze flags (mirrors train_il.py) ─────────────────────────────────────
    unnormalized       = args.unnormalized
    visible_preds_only = args.visible_preds_only
    alpha              = args.alpha
    use_gaze           = args.use_gaze
    target_diagonal    = args.target_diagonal
    if unnormalized or visible_preds_only:
        use_gaze = True
    if use_gaze and alpha is None and not unnormalized:
        alpha = 0.1

    # ── Device ───────────────────────────────────────────────────────────────
    make_deterministic(args.seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "cpu":
        device_name = args.device
    device = torch.device(device_name)
    print(f"Device: {device} | Gaze: {use_gaze} | Loss: {args.loss}")

    # ── Optional gaze predictor (for evaluation) ─────────────────────────────
    gaze_predictor = None
    if use_gaze:
        try:
            from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
            print(f"Initializing Gaze Predictor from {args.gaze_model_path} ...")
            gaze_predictor = Human_Gaze_Predictor(args.env)
            gaze_predictor.init_model(args.gaze_model_path)
            gaze_predictor.model.eval()
        except ImportError:
            print("Warning: Could not import Human_Gaze_Predictor. Gaze will be zero.")

    # ── Output directory ──────────────────────────────────────────────────────
    vis_tag   = "_vis_only" if visible_preds_only else ""
    alpha_tag = f"_a{alpha}" if alpha is not None else ""
    if args.run_dir:
        base_run_dir = args.run_dir
    else:
        if use_gaze and unnormalized:
            tag = f"grail_unnormalized{vis_tag}_td_{args.target_diagonal}"
        elif use_gaze:
            tag = f"grail_normalized{vis_tag}{alpha_tag}_td_{args.target_diagonal}"
        else:
            tag = f"nsfr_td_{args.target_diagonal}"
        base_run_dir = (
            f"trained_models/{args.env}/{tag}_learning_curve"
            f"_{args.rules}_rules_{args.lr}_lr_{args.loss}"
            f"_ep_{args.epochs}_seed_{args.seed}"
        )
    os.makedirs(base_run_dir, exist_ok=True)
    print(f"Base output dir: {base_run_dir}")

    # ── Probe agent to get the correct max_action for this environment ────────
    # (mirrors train_il.py which does: max_action=agent.num_actions - 1)
    # Must be done before loading PtDataset so actions outside the valid range
    # are filtered out immediately — otherwise NLLLoss raises an out-of-range
    # CUDA assertion (e.g. Asterix has 5 actions 0-4, not 6 like Seaquest).
    print("\nProbing ImitationAgent to determine action count ...")
    _probe_agent = ImitationAgent(
        args.env, args.rules, device,
        gaze_threshold=(args.gaze_threshold if use_gaze else None),
        unnormalized=unnormalized,
        visible_preds_only=visible_preds_only,
        alpha=alpha,
        aggregation_method=args.aggregation,
        target_diagonal=target_diagonal,
    )
    max_action = _probe_agent.num_actions - 1
    print(f"  num_actions={_probe_agent.num_actions}  →  max_action={max_action}")
    del _probe_agent
    gc.collect()

    # ── Load full dataset once to know total sample count ────────────────────
    print("\nLoading full dataset ...")
    full_dataset = PtDataset(
        args.dataset,
        use_gaze=use_gaze,
        num_episodes=None,
        sort_by=None,
        max_action=max_action,   # env-specific, e.g. 4 for Asterix, 5 for Seaquest
    )
    total_samples = len(full_dataset)
    print(f"Total samples in dataset: {total_samples}")

    # ── Resolve valuation path ────────────────────────────────────────────────
    if args.valuation_path:
        v_path = args.valuation_path
    else:
        if args.use_gaze:
            tag = "grail_normalized"
            if unnormalized:
                tag = "grail_unnormalized"
            if visible_preds_only:
                tag = "grail_normalized_vis_only"
        else:
            tag = "nsfr"
        config_dir = os.path.join("trained_models", args.env, tag,
                                  f"{args.rules}_rules_{args.lr}_lr_{args.loss}")
        v_path = os.path.join(config_dir, f"valuations_{args.rules}.pt")
    print(f"Valuation path: {v_path}")

    # ── Precompute valuations if not found ────────────────────────────────────
    if not os.path.exists(v_path):
        print(f"\n{'='*50}")
        print("Valuation file not found — precomputing from full dataset ...")
        print(f"{'='*50}")

        # Build a temporary agent for access to fc / atoms
        _tmp_agent = ImitationAgent(
            args.env, args.rules, device,
            gaze_threshold=(args.gaze_threshold if use_gaze else None),
            unnormalized=unnormalized,
            visible_preds_only=visible_preds_only,
            alpha=alpha,
            aggregation_method=args.aggregation,
            target_diagonal=target_diagonal,
        )

        # Initialise gaze predictor if needed for precompute
        _pre_gaze_predictor = None
        if use_gaze:
            try:
                from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
                print(f"Initializing Gaze Predictor for precomputation from {args.gaze_model_path} ...")
                _pre_gaze_predictor = Human_Gaze_Predictor(args.env)
                _pre_gaze_predictor.init_model(args.gaze_model_path)
                _pre_gaze_predictor.model.eval()
                _pre_gaze_predictor.model.to(device)
            except ImportError:
                print("Warning: Could not import Human_Gaze_Predictor for precomputation.")
        elif use_gaze:
            _pre_gaze_predictor = gaze_predictor

        _data = torch.load(args.dataset, map_location='cpu', weights_only=False)
        obs_t     = _data['observations']
        if not isinstance(obs_t, torch.Tensor):     obs_t     = torch.tensor(obs_t)
        logic_t   = _data['logic_state']
        if not isinstance(logic_t, torch.Tensor):   logic_t   = torch.tensor(logic_t)
        actions_t = _data['actions']
        if not isinstance(actions_t, torch.Tensor): actions_t = torch.tensor(actions_t)
        ep_nums_t = _data.get('episode_number', torch.zeros(len(obs_t), dtype=torch.long))
        if not isinstance(ep_nums_t, torch.Tensor): ep_nums_t = torch.tensor(ep_nums_t)

        valuations_precomp = {}
        with torch.no_grad():
            unique_eps_pre = torch.unique(ep_nums_t).numpy()
            for ep in tqdm(unique_eps_pre, desc="Precomputing valuations"):
                mask      = (ep_nums_t == ep)
                ep_obs    = obs_t[mask]
                ep_logic  = logic_t[mask]
                ep_act    = actions_t[mask]
                T_ep      = len(ep_obs)
                if T_ep == 0:
                    continue

                # 1. Gaze stacking (only if using gaze)
                ep_gazes_t = None
                if use_gaze and _pre_gaze_predictor is not None:
                    pad = ep_obs[0:1].expand(3, -1, -1)
                    padded_obs = torch.cat([pad, ep_obs], dim=0)
                    stacks = torch.stack([padded_obs[i:i+4] for i in range(T_ep)], dim=0)

                    batch_sz_g = 256
                    ep_gazes = []
                    for i in range(0, T_ep, batch_sz_g):
                        bs = stacks[i:i+batch_sz_g].to(device, dtype=torch.float32) / 255.0
                        gp = _pre_gaze_predictor.predict_normalized(bs).squeeze(1)
                        ep_gazes.append(gp)
                    ep_gazes_t = torch.cat(ep_gazes, dim=0)

                # 2. Filter valid actions
                valid_mask = (ep_act <= 5)
                valid_logic = ep_logic[valid_mask]
                K_ep = len(valid_logic)
                if K_ep == 0:
                    continue

                # 3. Compute v0 (batched)
                ep_v0 = []
                batch_sz_v = 256
                for i in range(0, K_ep, batch_sz_v):
                    b_logic = valid_logic[i:i+batch_sz_v].to(device, dtype=torch.float32)
                    b_gaze  = None
                    if use_gaze and ep_gazes_t is not None:
                        valid_gazes = ep_gazes_t[valid_mask]
                        b_gaze = valid_gazes[i:i+batch_sz_v]
                    v0 = _tmp_agent.unwrapped_model.fc(
                        b_logic,
                        _tmp_agent.unwrapped_model.atoms,
                        _tmp_agent.unwrapped_model.bk,
                        gaze=b_gaze,
                    )
                    ep_v0.append(v0.cpu())
                ep_v0_t = torch.cat(ep_v0, dim=0)
                valuations_precomp[int(ep)] = [ep_v0_t[i] for i in range(K_ep)]

        del _data, obs_t, logic_t, actions_t, ep_nums_t, _tmp_agent
        gc.collect()

        os.makedirs(os.path.dirname(v_path), exist_ok=True)
        torch.save(valuations_precomp, v_path)
        print(f"Valuations saved to {v_path}\n")
        del valuations_precomp

    # ── Results accumulator ───────────────────────────────────────────────────
    results_log = []

    # ── Graceful interrupt handler ────────────────────────────────────────────
    def _emergency_save(signum, frame):
        print(f"\n\n[INTERRUPTED] Signal {signum} received. Saving learning curve summary...")
        try:
            if results_log:
                df = pd.DataFrame(results_log)
                summary_csv = os.path.join(base_run_dir, "learning_curve_summary.csv")
                df.to_csv(summary_csv, index=False)
                print(f"  Saved partial summary to {summary_csv}")
                print("\nNSFR Learning Curve Summary (Partial):")
                print(df.to_string(index=False))
            else:
                print("  No completed percentage subsets to save.")
        except Exception as e:
            print(f"  WARNING: Could not save summary CSV: {e}")
        print("[INTERRUPTED] Exiting.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT,  _emergency_save)
    signal.signal(signal.SIGTERM, _emergency_save)

    # ── Learning-curve loop ───────────────────────────────────────────────────
    for pct in args.percentages:
        n_samples = max(1, int(total_samples * pct / 100))
        folder_name = f"{pct}p"
        run_dir = os.path.join(base_run_dir, folder_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"  {pct}% of dataset  →  {n_samples} / {total_samples} samples")
        print(f"  Output: {run_dir}")
        print(f"{'='*65}")

        # ── Slice first N% (in order) via Subset ─────────────────────────────
        slice_indices = list(range(n_samples))
        slice_dataset = Subset(full_dataset, slice_indices)

        # ── 95/5 train/val split ──────────────────────────────────────────────
        if args.val_split > 0:
            val_n   = max(1, int(n_samples * args.val_split))
            train_n = n_samples - val_n
            train_dataset, val_dataset = random_split(
                slice_dataset, [train_n, val_n],
                generator=torch.Generator().manual_seed(args.seed),
            )
            print(f"  Train: {train_n} | Val: {val_n}")
        else:
            train_dataset = slice_dataset
            val_dataset   = None
            print(f"  Train: {n_samples} (no val split)")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0),
        ) if val_dataset else None

        # ── Fresh ImitationAgent for each percentage ──────────────────────────
        make_deterministic(args.seed)
        agent_gaze_threshold = args.gaze_threshold if use_gaze else None
        agent = ImitationAgent(
            args.env, args.rules, device,
            gaze_threshold=agent_gaze_threshold,
            unnormalized=unnormalized,
            visible_preds_only=visible_preds_only,
            alpha=alpha,
            aggregation_method=args.aggregation,
            target_diagonal=target_diagonal,
        )

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"  DataParallel across {num_gpus} GPUs.")
            agent.model = nn.DataParallel(agent.model)
        else:
            print(f"  Single GPU / CPU — DataParallel not applied.")

        # ── Load pre-computed valuations (guaranteed by precompute block above) ─
        valuations = load_valuations(v_path, agent, device)
        if valuations is None:
            print(f"  WARNING: Valuations still not found at {v_path}. "
                  f"Will use zero vT tensors — check precompute step.")

        # ── Optimizer / Scheduler ─────────────────────────────────────────────
        optimizer = torch.optim.Adam(agent.unwrapped_model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=args.lr_patience
        )

        best_val_loss  = float('inf')
        best_val_acc   = -float('inf')
        pct_results    = []

        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in range(args.epochs):
            agent.model.train()
            total_loss, n_batches        = 0.0, 0
            train_correct, train_samples = 0, 0

            pbar = tqdm(train_loader, desc=f"[{pct}%] Epoch {epoch+1}/{args.epochs}", leave=False)
            for states, actions, gazes, ep_nums, step_idxs in pbar:
                states  = states.to(device)
                actions = actions.to(device)
                gazes   = gazes.to(device)
                batch_sz = states.size(0)

                vT_batch = build_vT_batch(ep_nums, step_idxs, valuations, agent, device)

                loss, probs, action_scores = agent.update(
                    states, actions,
                    gazes if use_gaze else None,
                    vT=vT_batch,
                    loss_type=args.loss,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.unwrapped_model.parameters(), 1.0)
                optimizer.step()

                loss_val        = loss.item() / max(num_gpus, 1)
                train_correct  += (action_scores.argmax(dim=1) == actions).sum().item()
                train_samples  += batch_sz
                total_loss     += loss_val
                n_batches      += 1
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            avg_loss      = total_loss / max(n_batches, 1)
            avg_train_acc = train_correct / max(train_samples, 1)
            print(f"  [{pct}%] Epoch {epoch+1}/{args.epochs} | "
                  f"Loss {avg_loss:.4f} | TrainAcc {avg_train_acc:.4f}")

            # ── Validation ────────────────────────────────────────────────────
            avg_val_acc  = 0.0
            avg_val_loss = float('nan')
            eps_nll      = 1e-10

            if val_loader:
                agent.model.eval()
                val_loss_sum, val_n          = 0.0, 0
                val_correct,  val_samples    = 0, 0

                with torch.no_grad():
                    for states, actions, gazes, ep_nums, step_idxs in val_loader:
                        states  = states.to(device)
                        actions = actions.to(device)
                        gazes   = gazes.to(device)
                        B       = states.size(0)

                        vT_batch = build_vT_batch(ep_nums, step_idxs, valuations, agent, device)

                        probs, action_scores = agent.predict(
                            states,
                            gazes if use_gaze else None,
                            vT=vT_batch,
                        )
                        val_correct  += (action_scores.argmax(dim=1) == actions).sum().item()
                        val_samples  += B
                        log_scores    = torch.log(action_scores.clamp(min=eps_nll))
                        val_loss_sum += nn.NLLLoss()(log_scores, actions).item()
                        val_n        += 1

                avg_val_acc  = val_correct  / max(val_samples, 1)
                avg_val_loss = val_loss_sum / max(val_n, 1)
                print(f"  [{pct}%] Epoch {epoch+1}/{args.epochs} | "
                      f"ValLoss {avg_val_loss:.4f} | ValAcc {avg_val_acc:.4f}")

            scheduler.step(avg_val_acc if val_loader else -avg_loss)

            # ── Save checkpoint per epoch ─────────────────────────────────────
            save_path = os.path.join(run_dir, f"epoch_{epoch+1}.pth")
            agent.save(save_path)

            # ── Track best model ──────────────────────────────────────────────
            improved = False
            if val_loader:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    improved = True
            else:
                # No val split: use negative train loss as proxy
                if -avg_loss > -best_val_loss:
                    best_val_loss = avg_loss
                    improved = True

            if improved:
                best_path = os.path.join(run_dir, "best_model.pth")
                agent.save(best_path)
                print(f"    *** New best model saved (ValLoss: {avg_val_loss:.4f})")

            pct_results.append({
                'percentage':  pct,
                'n_samples':   n_samples,
                'epoch':       epoch + 1,
                'train_loss':  avg_loss,
                'train_acc':   avg_train_acc,
                'val_loss':    avg_val_loss,
                'val_acc':     avg_val_acc,
            })

        # ── Evaluate best model for eval_episodes ─────────────────────────────
        print(f"\n  Evaluating best model for {args.eval_episodes} episodes ...")
        best_ckpt = os.path.join(run_dir, "best_model.pth")
        if os.path.exists(best_ckpt):
            agent.load(best_ckpt)
            print(f"  Loaded best checkpoint: {best_ckpt}")
        else:
            print(f"  No best checkpoint found — evaluating last epoch model.")

        rewards = evaluate_parallel(
            agent,
            env_name=args.env,
            num_episodes=args.eval_episodes,
            seed=args.seed,
            max_steps=args.eval_max_steps,
            gaze_model_path=(args.gaze_model_path if (use_gaze or unnormalized) else None),
            use_gaze=use_gaze,
            num_workers=(16 if (use_gaze or unnormalized) else None),
            gabril_compat=False,
            train_run=True,
        )
        mean_r   = float(np.mean(rewards))
        std_r    = float(np.std(rewards))
        median_r = float(np.median(rewards))
        print(f"\n  [{pct}%] Eval → Mean: {mean_r:.2f} ± {std_r:.2f}  |  Median: {median_r:.2f}")

        # ── Save per-percentage results (written immediately for partial runs) ─
        per_pct_csv = os.path.join(run_dir, "training_log.csv")
        pd.DataFrame(pct_results).to_csv(per_pct_csv, index=False)

        eval_row = {
            'percentage':    pct,
            'n_samples':     n_samples,
            'best_val_loss': best_val_loss,
            'mean_reward':   mean_r,
            'std_reward':    std_r,
            'median_reward': median_r,
        }
        pd.DataFrame([eval_row]).to_csv(
            os.path.join(run_dir, "eval_results.csv"), index=False
        )
        results_log.append(eval_row)

        # ── Free resources before next run ────────────────────────────────────
        del agent, optimizer, scheduler
        del train_dataset, val_dataset, train_loader, val_loader, slice_dataset
        if valuations is not None:
            del valuations
        torch.cuda.empty_cache()
        gc.collect()

    # ── Aggregate summary ─────────────────────────────────────────────────────
    df = pd.DataFrame(results_log)
    summary_csv = os.path.join(base_run_dir, "learning_curve_summary.csv")
    df.to_csv(summary_csv, index=False)

    print(f"\n{'='*65}")
    print("NSFR Learning Curve Summary:")
    print(df.to_string(index=False))
    print(f"\nSummary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
