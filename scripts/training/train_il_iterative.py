"""
train_il_iterative.py
=====================
Iterative imitation-learning training with progressive data loading.

For each data fraction f in [10%, 20%, ..., 100%]:
  1. Subset the full dataset to the first f% of *unique episodes*.
  2. Train for --epochs_per_iter epochs (default 10).
  3. Save a checkpoint  <run_dir>/iter_{f:03d}pct_final.pth
  4. Evaluate for --num_eval_episodes episodes (default 50).
  5. Append results to a running CSV.

All other mechanics (optimizer, scheduler, gaze, valuations, multi-GPU,
graceful interrupt) mirror the original train_il.py.

Usage example
-------------
python -m scripts.training.train_il_iterative \
    --env seaquest \
    --rules improved \
    --dataset /data/seaquest/seaquest_25obj.pt \
    --device cuda \
    --epochs_per_iter 10 \
    --num_eval_episodes 50 \
    --lr 0.01
"""

import os
os.environ['ALE_PY_QUIET'] = '1'
import argparse
import datetime
import logging
import signal
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from core.utils.utils import get_primitive_action_map, PtDataset, set_seed_everywhere
from nsfr.agents.imitation_agent import ImitationAgent
from nsfr.utils import make_deterministic
from nsfr.env import NSFRBaseEnv
from scripts.evaluation.evaluate_model import evaluate_parallel

logging.getLogger("ale_py").setLevel(logging.ERROR)

import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_fraction_subset(full_dataset: PtDataset, fraction: float) -> Subset:
    """
    Return a Subset containing the first `fraction` of unique episodes.

    Episodes are taken in the order they appear in the dataset (by episode id).
    This mirrors the trajectory-based sampling used in the original training
    script rather than a pure random sample-level slice.
    """
    unique_eps = torch.unique(full_dataset.ep_nums).tolist()
    n_keep = max(1, int(round(len(unique_eps) * fraction)))
    kept_eps = set(unique_eps[:n_keep])

    indices = [
        i for i, ep in enumerate(full_dataset.ep_nums.tolist())
        if ep in kept_eps
    ]
    return Subset(full_dataset, indices)


def make_weighted_loader(subset: Subset, batch_size: int, num_workers: int,
                         pin_memory: bool) -> DataLoader:
    """DataLoader with inverse-frequency class balancing."""
    if isinstance(subset, Subset):
        train_actions = subset.dataset.actions[subset.indices]
    else:
        train_actions = subset.actions

    num_classes   = 6
    class_counts  = torch.bincount(train_actions, minlength=num_classes).float().clamp(min=1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_actions]

    # Use a plain shuffle loader (weighted sampler commented-out in original)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Iterative IL training: load 10→100% data in steps of 10%."
    )
    # ── Core args (same as train_il.py) ──────────────────────────────────────
    parser.add_argument("--env",              type=str, default="seaquest")
    parser.add_argument("--rules",            type=str, default="improved")
    parser.add_argument("--dataset",          type=str, required=True)
    parser.add_argument("--epochs_per_iter",  type=int, default=10,
                        help="Training epochs per data-fraction iteration")
    parser.add_argument("--loss",             type=str, default="nll",
                        choices=["nll", "bce"])
    parser.add_argument("--num_eval_episodes",type=int, default=50)
    parser.add_argument("--batch_size",       type=int, default=32)
    parser.add_argument("--lr",               type=float, default=0.01)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--device",           type=str, default="cpu")
    parser.add_argument("--num_workers",      type=int, default=4)
    parser.add_argument("--val_split",        type=float, default=0.05)
    parser.add_argument("--lr_patience",      type=int, default=3)
    parser.add_argument("--gaze_threshold",   type=float, default=50.0)
    parser.add_argument("--use_gaze",         action="store_true")
    parser.add_argument("--gaze_model_path",  type=str,
                        default="gaze_models/seaquest/seaquest_gaze_predictor_2.pth")
    parser.add_argument("--sort_by",          type=str, default=None,
                        choices=['length', 'reward_per_step'])
    parser.add_argument("--valuation_path",   type=str, default=None)
    parser.add_argument("--eval_max_steps",   type=int, default=10_000)
    parser.add_argument("--send_email",       action="store_true")
    parser.add_argument("--email_interval",   type=int, default=30)
    parser.add_argument("--unnormalized",     action="store_true")
    parser.add_argument("--visible_preds_only", action="store_true")
    parser.add_argument("--alpha",            type=float, default=None)
    parser.add_argument("--aggregation",      type=str, default="max",
                        choices=["softor", "max"], help="Action aggregation method")
    # ── Iterative-specific ────────────────────────────────────────────────────
    parser.add_argument("--step_pct",         type=int, default=10,
                        help="Increment step (%%): 10 means 10,20,...,100")
    parser.add_argument("--resume_from",      type=str, default=None,
                        help="Path to checkpoint to resume agent weights from")
    parser.add_argument("--fresh_optimizer",  action="store_true",
                        help="Reset optimizer at each new data fraction")

    args = parser.parse_args()

    # ── Derived flags ─────────────────────────────────────────────────────────
    unnormalized       = args.unnormalized
    visible_preds_only = args.visible_preds_only
    alpha              = args.alpha
    use_gaze           = args.use_gaze
    if unnormalized or visible_preds_only:
        use_gaze = True
    if use_gaze and alpha is None and not unnormalized:
        alpha = 0.1

    now_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # ── Gaze predictor ────────────────────────────────────────────────────────
    gaze_predictor = None
    if use_gaze:
        from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
        print(f"Initializing Gaze Predictor from {args.gaze_model_path}...")
        gaze_predictor = Human_Gaze_Predictor(args.env)
        gaze_predictor.init_model(args.gaze_model_path)
        gaze_predictor.model.eval()

    # ── Device / seed / env ───────────────────────────────────────────────────
    make_deterministic(args.seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "cpu":
        device_name = args.device
    device = torch.device(device_name)
    print(f"Using device: {device}")

    env = NSFRBaseEnv.from_name(args.env, mode='logic')

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent_gaze_threshold = args.gaze_threshold if use_gaze else None
    agent = ImitationAgent(
        args.env, args.rules, device,
        gaze_threshold=agent_gaze_threshold,
        unnormalized=unnormalized,
        visible_preds_only=visible_preds_only,
        alpha=alpha,
        aggregation_method=args.aggregation,
    )

    if args.resume_from:
        print(f"Resuming agent weights from {args.resume_from}")
        agent.load(args.resume_from)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs.")
        agent.model = nn.DataParallel(agent.model)
    else:
        print("Single GPU / CPU — DataParallel not applied.")

    # ── Load full dataset once ────────────────────────────────────────────────
    full_dataset = PtDataset(
        args.dataset,
        use_gaze=use_gaze,
        sort_by=args.sort_by,
    )
    all_unique_eps = torch.unique(full_dataset.ep_nums)
    total_episodes = len(all_unique_eps)
    print(f"Full dataset: {len(full_dataset)} samples | {total_episodes} episodes")

    # ── Output directory ──────────────────────────────────────────────────────
    vis_tag   = "_vis_only" if visible_preds_only else ""
    alpha_tag = f"_a{alpha}" if alpha is not None else ""
    if use_gaze and unnormalized:
        run_dir = (f"trained_models/{args.env}/grail_iterative/"
                   f"{args.rules}_rules_{args.lr}_lr_{args.loss}_unnormalized{vis_tag}/{now_time}")
    elif use_gaze:
        run_dir = (f"trained_models/{args.env}/grail_iterative/"
                   f"{args.rules}_rules_{args.lr}_lr_{args.loss}_normalized{vis_tag}{alpha_tag}/{now_time}")
    else:
        run_dir = (f"trained_models/{args.env}/nsfr_iterative/"
                   f"{args.rules}_rules_{args.lr}_lr_{args.loss}/{now_time}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ── Pre-compute valuations ────────────────────────────────────────────────
    config_dir = os.path.dirname(os.path.dirname(run_dir))
    valuations = None
    v_path = args.valuation_path
    if v_path is None:
        v_path = os.path.join(config_dir, f"valuations_{args.rules}.pt")

    if os.path.exists(v_path):
        print(f"Loading pre-computed valuations from {v_path}...")
        valuations_raw = torch.load(v_path, map_location=device, weights_only=False)
        if (isinstance(valuations_raw, dict)
                and 'data' in valuations_raw
                and isinstance(valuations_raw['data'], list)):
            print("  Reformatting valuations from list-of-dicts...")
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
    else:
        print(f"No pre-computed valuations at {v_path}. Will precompute from logic states.")

    if valuations is None:
        print("\n" + "=" * 50)
        print("Precomputing valuations...")
        print("=" * 50)
        if use_gaze and gaze_predictor is None:
            from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
            gaze_predictor = Human_Gaze_Predictor(args.env)
            gaze_predictor.init_model(args.gaze_model_path)
            gaze_predictor.model.eval()
            gaze_predictor.model.to(device)

        import gc
        data     = torch.load(args.dataset, map_location='cpu', weights_only=False)
        obs_t    = data['observations']
        if not isinstance(obs_t, torch.Tensor):    obs_t    = torch.tensor(obs_t)
        logic_t  = data['logic_state']
        if not isinstance(logic_t, torch.Tensor):  logic_t  = torch.tensor(logic_t)
        actions_t = data['actions']
        if not isinstance(actions_t, torch.Tensor): actions_t = torch.tensor(actions_t)
        ep_nums_t = data.get('episode_number', torch.zeros(len(obs_t), dtype=torch.long))
        if not isinstance(ep_nums_t, torch.Tensor): ep_nums_t = torch.tensor(ep_nums_t)

        valuations = {}
        with torch.no_grad():
            for ep in tqdm(torch.unique(ep_nums_t).numpy(), desc="Precomputing valuations"):
                mask     = (ep_nums_t == ep)
                ep_obs   = obs_t[mask]
                ep_logic = logic_t[mask]
                ep_actions = actions_t[mask]
                T_ep = len(ep_obs)
                if T_ep == 0:
                    continue

                if use_gaze:
                    pad    = ep_obs[0:1].expand(3, -1, -1)
                    padded = torch.cat([pad, ep_obs], dim=0)
                    stacks = torch.stack([padded[i:i+4] for i in range(T_ep)])
                    ep_gazes = []
                    for i in range(0, T_ep, 256):
                        b = stacks[i:i+256].to(device, dtype=torch.float32) / 255.0
                        ep_gazes.append(gaze_predictor.predict_normalized(b).squeeze(1))
                    ep_gazes_t = torch.cat(ep_gazes, dim=0)

                valid_mask  = (ep_actions <= 5)
                valid_logic = ep_logic[valid_mask]
                K_ep = len(valid_logic)
                if K_ep == 0:
                    continue

                ep_v0 = []
                for i in range(0, K_ep, 256):
                    b_logic = valid_logic[i:i+256].to(device, dtype=torch.float32)
                    b_gaze  = None
                    if use_gaze:
                        b_gaze = ep_gazes_t[valid_mask][i:i+256]
                    v0 = agent.unwrapped_model.fc(
                        b_logic, agent.unwrapped_model.atoms,
                        agent.unwrapped_model.bk, gaze=b_gaze
                    )
                    ep_v0.append(v0.cpu())
                valuations[int(ep)] = [torch.cat(ep_v0, dim=0)[i] for i in range(K_ep)]

        del data, obs_t, logic_t, actions_t, ep_nums_t
        gc.collect()
        v_save_path = os.path.join(config_dir, f"valuations_{args.rules}.pt")
        print(f"Saving precomputed valuations to {v_save_path}")
        torch.save(valuations, v_save_path)

    # ── Shared results log & graceful interrupt ───────────────────────────────
    results_log     = []
    last_email_time = time.time()

    def _emergency_save(signum, frame):
        print(f"\n\n[INTERRUPTED] Signal {signum}. Saving checkpoint...")
        try:
            agent.save(os.path.join(run_dir, "interrupted.pth"))
        except Exception as e:
            print(f"  WARNING: Could not save model: {e}")
        try:
            if results_log:
                csv_path = os.path.join(run_dir, "results_interrupted.csv")
                pd.DataFrame(results_log).to_csv(csv_path, index=False)
                print(f"  Results saved to {csv_path}")
        except Exception as e:
            print(f"  WARNING: Could not save CSV: {e}")
        raise SystemExit(0)

    signal.signal(signal.SIGINT,  _emergency_save)
    signal.signal(signal.SIGTERM, _emergency_save)

    # ── Build fraction steps  [10, 20, ..., 100] ─────────────────────────────
    fractions = list(range(args.step_pct, 101, args.step_pct))
    print(f"\nIterative training plan: {fractions}% data steps, "
          f"{args.epochs_per_iter} epochs/step, "
          f"{args.num_eval_episodes} eval episodes/step\n")

    # ── Outer loop: data fractions ────────────────────────────────────────────
    for pct in fractions:
        fraction = pct / 100.0
        n_keep_eps = max(1, int(round(total_episodes * fraction)))

        print("\n" + "=" * 70)
        print(f"  DATA FRACTION: {pct}%  ({n_keep_eps}/{total_episodes} episodes)")
        print("=" * 70)

        # ── Build fraction subset ─────────────────────────────────────────────
        frac_subset = build_fraction_subset(full_dataset, fraction)

        # Optional val split
        if args.val_split > 0:
            val_n   = max(1, int(len(frac_subset) * args.val_split))
            train_n = len(frac_subset) - val_n
            train_subset, val_subset = random_split(
                frac_subset, [train_n, val_n],
                generator=torch.Generator().manual_seed(args.seed),
            )
            print(f"  Train: {train_n} samples | Val: {val_n} samples")
        else:
            train_subset = frac_subset
            val_subset   = None
            print(f"  Train: {len(train_subset)} samples (no val split)")

        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0),
        ) if val_subset else None

        # ── Optimizer (optionally reset each iteration) ───────────────────────
        if pct == fractions[0] or args.fresh_optimizer:
            optimizer = torch.optim.Adam(
                agent.unwrapped_model.parameters(), lr=args.lr
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=args.lr_patience,
            )
            if pct != fractions[0]:
                print("  [fresh_optimizer] Reset optimizer & scheduler.")

        best_val_loss    = float('inf')
        patience_counter = 0
        patience         = 12

        # ── Inner loop: epochs per fraction ──────────────────────────────────
        for epoch in range(args.epochs_per_iter):
            epoch_display = epoch + 1
            print(f"\n  --- Iter {pct}% | Epoch {epoch_display}/{args.epochs_per_iter} ---")

            train_loader = DataLoader(
                train_subset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda'),
                persistent_workers=(args.num_workers > 0),
            )

            agent.model.train()
            total_loss, n_batches        = 0.0, 0
            train_correct, train_samples = 0, 0

            pbar = tqdm(train_loader, desc=f"Iter {pct}% Epoch {epoch_display}")
            for states, actions, gazes, ep_nums, step_idxs in pbar:
                states  = states.to(device)
                actions = actions.to(device)
                gazes   = gazes.to(device)

                # Build vT batch from pre-computed valuations
                vT_batch = None
                if valuations is not None:
                    vT_list = []
                    for ep_id, s_idx in zip(ep_nums.tolist(), step_idxs.tolist()):
                        if ep_id in valuations and s_idx < len(valuations[ep_id]):
                            vT_list.append(valuations[ep_id][s_idx])
                        else:
                            vT_list.append(
                                torch.zeros(len(agent.unwrapped_model.atoms), device=device)
                            )
                    vT_batch = torch.stack(vT_list).to(device)

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
                train_samples  += states.size(0)
                total_loss     += loss_val
                n_batches      += 1
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            avg_loss      = total_loss / max(n_batches, 1)
            avg_train_acc = train_correct / max(train_samples, 1)
            print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {avg_train_acc:.4f}")

            # ── Validation ────────────────────────────────────────────────────
            avg_val_acc  = 0.0
            avg_val_loss = float('nan')

            if val_loader:
                agent.model.eval()
                val_loss, val_n          = 0.0, 0
                val_correct, val_samples = 0, 0
                eps_nll                  = 1e-10

                with torch.no_grad():
                    for states, actions, gazes, ep_nums, step_idxs in val_loader:
                        states  = states.to(device)
                        actions = actions.to(device)
                        gazes   = gazes.to(device)

                        vT_batch = None
                        if valuations is not None:
                            vT_list = [
                                valuations[ep_id.item()][s_idx.item()]
                                if (ep_id.item() in valuations
                                    and s_idx.item() < len(valuations[ep_id.item()]))
                                else torch.zeros(len(agent.unwrapped_model.atoms), device=device)
                                for ep_id, s_idx in zip(ep_nums, step_idxs)
                            ]
                            vT_batch = torch.stack(vT_list).to(device)

                        probs, action_scores = agent.predict(
                            states,
                            gazes if use_gaze else None,
                            vT=vT_batch,
                        )
                        val_correct += (action_scores.argmax(dim=1) == actions).sum().item()
                        val_samples += states.size(0)
                        log_scores   = torch.log(action_scores.clamp(min=eps_nll))
                        val_loss    += nn.NLLLoss()(log_scores, actions).item()
                        val_n       += 1

                avg_val_acc  = val_correct / max(val_samples, 1)
                avg_val_loss = val_loss / max(val_n, 1)
                print(f"  Val   Loss: {avg_val_loss:.4f} | Val   Acc: {avg_val_acc:.4f}")

            scheduler.step(avg_val_acc if val_loader else -avg_loss)

            # Early stopping within this fraction's training
            if avg_val_loss < best_val_loss:
                best_val_loss    = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [Early stop] No val improvement for {patience} epochs.")
                    break

        # ── After all epochs for this fraction: save checkpoint ───────────────
        ckpt_path = os.path.join(run_dir, f"iter_{pct:03d}pct_final.pth")
        agent.save(ckpt_path)
        print(f"\n  [Checkpoint] Saved → {ckpt_path}")

        # ── Evaluate for 50 episodes ──────────────────────────────────────────
        print(f"  [Eval] Running {args.num_eval_episodes} episodes...")
        rewards = evaluate_parallel(
            agent,
            env_name=args.env,
            num_episodes=args.num_eval_episodes,
            seed=args.seed,
            max_steps=args.eval_max_steps,
            gaze_model_path=(args.gaze_model_path if use_gaze or unnormalized else None),
            use_gaze=use_gaze,
            num_workers=(16 if use_gaze or unnormalized else None),
            train_run=True,
        )
        mean_reward = np.mean(rewards)
        std_reward  = np.std(rewards)
        print(f"  [Eval] Fraction {pct}%: Mean={mean_reward:.2f}  Std={std_reward:.2f}")

        # ── Log results ───────────────────────────────────────────────────────
        results_log.append({
            'data_pct':        pct,
            'num_episodes_used': n_keep_eps,
            'epochs_trained':  args.epochs_per_iter,
            'mean_reward':     mean_reward,
            'std_reward':      std_reward,
            'train_loss':      avg_loss,
            'train_acc':       avg_train_acc,
            'val_acc':         avg_val_acc,
            'val_loss':        avg_val_loss,
            'gaze':            use_gaze,
            'checkpoint':      ckpt_path,
        })

        csv_path = os.path.join(run_dir, "results_iterative.csv")
        pd.DataFrame(results_log).to_csv(csv_path, index=False)
        print(f"  [CSV] Updated → {csv_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ITERATIVE TRAINING COMPLETE — LEARNING CURVE BY DATA FRACTION")
    print("=" * 70)
    print(f"{'Data%':>6}  {'Episodes':>8}  {'Train Loss':>10}  {'T-Acc':>6}  {'V-Acc':>6}  {'Score':>20}")
    print("-" * 70)
    for res in results_log:
        print(
            f"  {res['data_pct']:>4}%  "
            f"{res['num_episodes_used']:>8}  "
            f"{res['train_loss']:>10.4f}  "
            f"{res['train_acc']:>6.4f}  "
            f"{res['val_acc']:>6.4f}  "
            f"{res['mean_reward']:>8.2f} ± {res['std_reward']:.2f}"
        )
    print("=" * 70)

    final_csv = os.path.join(run_dir, "results_iterative_final.csv")
    pd.DataFrame(results_log).to_csv(final_csv, index=False)
    print(f"\nFinal results saved to {final_csv}")


if __name__ == "__main__":
    main()
