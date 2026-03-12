

import os
os.environ['ALE_PY_QUIET'] = '1'
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
from tqdm import tqdm
from core.utils.utils import get_primitive_action_map, get_gaze_model_path
from core.utils.utils import PtDataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from nsfr.agents.imitation_agent import ImitationAgent
from nsfr.utils import make_deterministic
from nsfr.env import NSFRBaseEnv
from core.utils.utils import (
    set_seed_everywhere, format_results_table,
    send_run_update, load_pt_dataset
)
import gc
from scripts.evaluation.evaluate_model import evaluate, evaluate_parallel
import time
import signal
import logging
logging.getLogger("ale_py").setLevel(logging.ERROR)
import torch.multiprocessing as mp
try:
     mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Free unused memory
torch.cuda.empty_cache()
gc.collect()

def run_diagnostics(agent, epoch_loader, device, args, warmup_batches=30, measure_batches=50):
    """
    Profiles real training throughput after warming up DataParallel replication.

    Cold single-batch timing is misleading with DataParallel — the first batch
    pays a one-time model replication cost to each GPU. This function runs
    warmup_batches first (discarded), then measures over measure_batches to
    get accurate steady-state throughput.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC MODE — profiling real steady-state throughput")
    print(f"  Warm-up:  {warmup_batches} batches (discarded)")
    print(f"  Measured: {measure_batches} batches (averaged)")
    print("=" * 60)

    loader_iter = iter(epoch_loader)
    optimizer = torch.optim.Adam(agent.unwrapped_model.parameters(), lr=args.lr)

    # Warm up: let DataParallel replicate model, CUDA JIT compile kernels
    print(f"  Running {warmup_batches} warm-up batches...")
    agent.model.train()
    for i in range(warmup_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(epoch_loader)
            batch = next(loader_iter)
        states, actions, gazes, ep_nums, step_idxs = batch
        states, actions, gazes = states.to(device), actions.to(device), gazes.to(device)
        loss, _, _ = agent.update(states, actions, gazes if use_gaze else None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure steady-state throughput over multiple batches
    print(f"  Measuring {measure_batches} batches...")
    t_loads, t_forwards, t_backwards = [], [], []
    total_samples = 0

    agent.model.train()
    wall_start = time.perf_counter()

    for i in range(measure_batches):
        try:
            t0 = time.perf_counter()
            batch = next(loader_iter)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_loads.append(time.perf_counter() - t0)

            states, actions, gazes, ep_nums, step_idxs = batch
            states, actions, gazes = states.to(device), actions.to(device), gazes.to(device)
            total_samples += states.size(0)

            t0 = time.perf_counter()
            loss, _, _ = agent.update(states, actions, gazes if use_gaze else None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_forwards.append(time.perf_counter() - t0)

            optimizer.zero_grad()
            t0 = time.perf_counter()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            optimizer.step()
            t_backwards.append(time.perf_counter() - t0)

        except StopIteration:
            break

    wall_elapsed = time.perf_counter() - wall_start
    actual_its   = len(t_loads)
    actual_its_s = actual_its / wall_elapsed
    samples_s    = total_samples / wall_elapsed

    def ms(vals): return sum(vals) / len(vals) * 1000

    avg_load = ms(t_loads)
    avg_fwd  = ms(t_forwards)
    avg_bwd  = ms(t_backwards)
    avg_total = avg_load + avg_fwd + avg_bwd

    print(f"\n  Batch size:             {args.batch_size}")
    print(f"  GPUs:                   {torch.cuda.device_count()}")
    print(f"  Per-GPU batch:          {args.batch_size // max(torch.cuda.device_count(), 1)}")
    print(f"\n  Avg data loading:       {avg_load:.1f} ms")
    print(f"  Avg forward pass:       {avg_fwd:.1f} ms")
    print(f"  Avg backward pass:      {avg_bwd:.1f} ms")
    print(f"  Avg total per batch:    {avg_total:.1f} ms")
    print(f"\n  ── Real throughput ──────────────────────────────────────")
    print(f"  Wall-clock it/s:        {actual_its_s:.2f}")
    print(f"  Samples/sec:            {samples_s:.0f}")
    print(f"  Est. epoch time (200k): {200_000 / samples_s / 60:.1f} min")

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved  = torch.cuda.memory_reserved(device)  / 1e9
        print(f"\n  GPU memory allocated:   {allocated:.2f} GB")
        print(f"  GPU memory reserved:    {reserved:.2f} GB")

    if avg_load > avg_fwd + avg_bwd:
        print(f"\n  >> BOTTLENECK: Data loading ({avg_load:.0f}ms). Increase --num_workers.")
    elif avg_fwd > avg_bwd * 2:
        print(f"\n  >> BOTTLENECK: Forward pass ({avg_fwd:.0f}ms). Symbolic ops dominate.")
    else:
        print(f"\n  >> BOTTLENECK: Backward pass ({avg_bwd:.0f}ms).")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE — remove --diagnose flag to train normally")
    print("=" * 60 + "\n")
    quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="seaquest", help="Environment name")
    parser.add_argument("--rules", type=str, default="improved", help="Ruleset name")
    parser.add_argument("--dataset", type=str, default=None, help="Path to .pt dataset file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--loss", type=str, default="nll", choices=["nll", "bce"],
                        help="Loss function: 'nll' or 'bce'")
    parser.add_argument("--num_eval_episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation fraction")
    parser.add_argument("--lr_patience", type=int, default=3, help="LR reduction patience")
    parser.add_argument("--gaze_threshold", type=float, default=50.0, help="Gaze valuation threshold")
    parser.add_argument("--use_gaze", action="store_true", help="Use full gaze map")
    parser.add_argument("--gaze_model_path", type=str,
                        default=None, help="Path to gaze predictor weights")
    parser.add_argument("--num_episodes", type=int, default=None, help="Episodes to load from .pt dataset")
    parser.add_argument("--sort_by", type=str, default=None, choices=['length', 'reward_per_step'])
    parser.add_argument("--valuation_path", type=str, default=None, help="Path to pre-computed valuation.pt")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--eval_max_steps", type=int, default=10000, help="Max steps per eval episode")
    parser.add_argument("--send_email", action="store_true", help="Enable periodic email updates")
    parser.add_argument("--email_interval", type=int, default=30, help="Minutes between email updates")
    parser.add_argument("--diagnose", action="store_true", help="Profile bottlenecks and exit")
    parser.add_argument("--unnormalized", action="store_true", help="Use unnormalized gaze probabilities and multiply valuations")
    parser.add_argument("--visible_preds_only", action="store_true", help="Only scale visible_{object} predicates with gaze")
    parser.add_argument("--alpha", type=float, default=None, help="Laplacian smoothing parameter for gaze normalization")
    parser.add_argument("--aggregation", type=str, default="max", choices=["softor", "max"], help="Aggregation method for action scores")
    parser.add_argument("--target_diagonal", type=float, default=0.99, help="Initial confidence for clauses in block diagonal initialization.")
    parser.add_argument("--random_init", action="store_true", help="Initialize weights randomly instead of target_diagonal.")
    args = parser.parse_args()
    
    # -- Resolve gaze model path if not provided
    if args.gaze_model_path is None:
        args.gaze_model_path = get_gaze_model_path(args.env)
    
    unnormalized = args.unnormalized
    visible_preds_only = args.visible_preds_only
    alpha = args.alpha
    use_gaze = args.use_gaze
    target_diagonal = args.target_diagonal
    # Auto-enable use_gaze if unnormalized is set
    if unnormalized or visible_preds_only:
        use_gaze = True

    if use_gaze and alpha is None and not unnormalized:
        alpha = 0.1

    now_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # ── Gaze setup ───────────────────────────────────────────────────────────
    gaze_predictor = None
    if use_gaze:
        from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
        print(f"Initializing Gaze Predictor from {args.gaze_model_path}...")
        gaze_predictor = Human_Gaze_Predictor(args.env)
        gaze_predictor.init_model(args.gaze_model_path)
        gaze_predictor.model.eval()

    if not args.dataset:
        raise ValueError("--dataset is required.")

    # ── Device ───────────────────────────────────────────────────────────────
    PRIMITIVE_ACTION_MAP = get_primitive_action_map(args.env)
    make_deterministic(args.seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "cpu":
        device_name = args.device
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # ── Environment & Agent ──────────────────────────────────────────────────
    env = NSFRBaseEnv.from_name(args.env, mode='logic')

    print(f"Initializing ImitationAgent for {args.env} with rules {args.rules}...")
    agent_gaze_threshold = args.gaze_threshold if use_gaze else None
    agent = ImitationAgent(args.env, args.rules, device, gaze_threshold=agent_gaze_threshold, unnormalized=unnormalized, visible_preds_only=visible_preds_only, alpha=alpha, aggregation_method=args.aggregation, target_diagonal=target_diagonal, random_init=args.random_init)

        # Multi-GPU: DataParallel splits the batch evenly across all visible GPUs.
    # Each GPU processes batch/N samples independently — no cross-GPU communication
    # during forward. Gradients are summed on GPU 0 after backward.
    # Requires nn.ModuleList (not plain lists) in all submodules — see infer.py.
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs.")
        print(f"Effective per-GPU batch size: {args.batch_size // num_gpus}")
        agent.model = nn.DataParallel(agent.model)
    else:
        print(f"Single GPU detected — DataParallel not applied.")


    # ── Dataset ──────────────────────────────────────────────────────────────
    full_dataset = PtDataset(
        args.dataset,
        use_gaze=use_gaze,
        num_episodes=args.num_episodes,
        sort_by=args.sort_by,
        max_action=agent.num_actions - 1,
        env_name=args.env,
    )
    if args.limit:
        full_dataset.logic   = full_dataset.logic[:args.limit]
        full_dataset.actions = full_dataset.actions[:args.limit]
        full_dataset.gaze    = full_dataset.gaze[:args.limit]

    if args.val_split > 0:
        val_n   = max(1, int(len(full_dataset) * args.val_split))
        train_n = len(full_dataset) - val_n
        train_dataset, val_dataset = random_split(
            full_dataset, [train_n, val_n],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"Train: {train_n} samples | Val: {val_n} samples")
    else:
        train_dataset = full_dataset
        val_dataset   = None
        print(f"Train: {len(train_dataset)} samples (no validation split)")

    if isinstance(train_dataset, torch.utils.data.Subset):
        ep_nums_all = train_dataset.dataset.ep_nums[train_dataset.indices]
    else:
        ep_nums_all = train_dataset.ep_nums
    unique_eps = torch.unique(ep_nums_all)
    print(f"Found {len(unique_eps)} unique episodes in training set.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    ) if val_dataset else None

    # ── Weighted sampler — balance action classes ─────────────────────────────
    # Action 0 (noop) is ~21% of data without balancing the model learns to
    # predict noop too often. WeightedRandomSampler gives rare actions higher
    # sample weights so each class is drawn at roughly equal frequency.
    if isinstance(train_dataset, torch.utils.data.Subset):
        train_actions = train_dataset.dataset.actions[train_dataset.indices]
    else:
        train_actions = train_dataset.actions

    num_classes    = agent.num_actions
    class_counts   = torch.bincount(train_actions, minlength=num_classes).float().clamp(min=1)
    class_weights  = 1.0 / class_counts
    sample_weights = class_weights[train_actions]

    action_names = ['noop', 'fire', 'up', 'right', 'left', 'down']
    # print("  Weighted sampler class weights:")
    # for i, (cnt, w) in enumerate(zip(class_counts.tolist(), class_weights.tolist())):
    #     print(f"    {i} ({action_names[i]:5s}): {int(cnt):6d} samples  weight={w:.5f}")

    # weighted_sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True,
    # )

    # ── Paths ─────────────────────────────────────────────────────────────────
    num_iters = len(unique_eps) if (args.num_episodes is None) and (args.limit is not None) else "full"
    vis_tag = "_vis_only" if visible_preds_only else ""
    alpha_tag = f"_a{alpha}" if alpha is not None else ""
    init_tag = "random_init" if args.random_init else f"td_{args.target_diagonal}"
    
    if use_gaze and unnormalized:
        run_dir   = (f"trained_models/{args.env}/grail/{args.rules}_rules_{args.lr}_lr_{args.loss}_unnormalized{vis_tag}_{init_tag}/{num_iters}_ep/{now_time}")
    elif use_gaze:
        run_dir   = (f"trained_models/{args.env}/grail/{args.rules}_rules_{args.lr}_lr_{args.loss}_normalized{vis_tag}_{init_tag}/{num_iters}_ep/{now_time}")
    else:
        run_dir   = (f"trained_models/{args.env}/nsfr/{args.rules}_rules_{args.lr}_lr_{args.loss}_{init_tag}/{num_iters}_ep/{now_time}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("out/imitation", exist_ok=True)

    # ── Load pre-computed valuations ─────────────────────────────────────────
    # First, try to load from the specific run_dir's parent (the config folder)
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
            print("  Reformatting valuations from list-of-dicts to episode-based lists...")
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
        print("\n" + "="*50)
        print("Precomputing valuations...")
        print("="*50)
        
        # We need gaze_predictor only if we are using gaze
        if use_gaze and gaze_predictor is None:
            from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
            print(f"Initializing Gaze Predictor for precomputation from {args.gaze_model_path}...")
            gaze_predictor = Human_Gaze_Predictor(args.env)
            gaze_predictor.init_model(args.gaze_model_path)
            gaze_predictor.model.eval()
            gaze_predictor.model.to(device)
            
        import gc
        data = torch.load(args.dataset, map_location='cpu', weights_only=False)
        obs_t = data['observations']
        if not isinstance(obs_t, torch.Tensor): obs_t = torch.tensor(obs_t)
        logic_t = data['logic_state']
        if not isinstance(logic_t, torch.Tensor): logic_t = torch.tensor(logic_t)
        actions_t = data['actions']
        if not isinstance(actions_t, torch.Tensor): actions_t = torch.tensor(actions_t)
        ep_nums_t = data.get('episode_number', torch.zeros(len(obs_t), dtype=torch.long))
        if not isinstance(ep_nums_t, torch.Tensor): ep_nums_t = torch.tensor(ep_nums_t)
        
        valuations = {}
        with torch.no_grad():
            unique_eps_pre = torch.unique(ep_nums_t).numpy()
            for ep in tqdm(unique_eps_pre, desc="Precomputing valuations"):
                mask = (ep_nums_t == ep)
                ep_obs = obs_t[mask]
                ep_logic = logic_t[mask]
                ep_actions = actions_t[mask]
                
                T_ep = len(ep_obs)
                if T_ep == 0: continue
                
                # 1. Stacking (only needed for gaze)
                if use_gaze:
                    pad = ep_obs[0:1].expand(3, -1, -1)
                    padded_obs = torch.cat([pad, ep_obs], dim=0) # (T_ep + 3, H, W)
                    
                    stacks = []
                    for i in range(T_ep):
                        stacks.append(padded_obs[i:i+4])
                    stacks = torch.stack(stacks, dim=0) # (T_ep, 4, H, W)
                    
                    # 2. Predict Gazes
                    batch_size_gaze = 256
                    ep_gazes = []
                    for i in range(0, T_ep, batch_size_gaze):
                        batch_stacks = stacks[i:i+batch_size_gaze].to(device, dtype=torch.float32) / 255.0
                        gaze_pred = gaze_predictor.predict_normalized(batch_stacks).squeeze(1) # (B, 84, 84)
                        ep_gazes.append(gaze_pred)
                    ep_gazes_t = torch.cat(ep_gazes, dim=0) # (T_ep, 84, 84)
                
                # 3. Filter actions <= 5
                valid_mask = (ep_actions <= 5)
                valid_logic = ep_logic[valid_mask]
                
                K_ep = len(valid_logic)
                if K_ep == 0: continue
                
                # 4. Compute V_0
                ep_v0 = []
                batch_size_gaze = 256
                for i in range(0, K_ep, batch_size_gaze):
                    b_logic = valid_logic[i:i+batch_size_gaze].to(device, dtype=torch.float32)
                    b_gaze = None
                    if use_gaze:
                        valid_gazes = ep_gazes_t[valid_mask]
                        b_gaze = valid_gazes[i:i+batch_size_gaze]
                    v0 = agent.unwrapped_model.fc(b_logic, agent.unwrapped_model.atoms, agent.unwrapped_model.bk, gaze=b_gaze)
                    ep_v0.append(v0.cpu())
                ep_v0_t = torch.cat(ep_v0, dim=0) # (K_ep, num_atoms)
                
                # 5. Store in list
                valuations[int(ep)] = [ep_v0_t[i] for i in range(K_ep)]
                
        # cleanup memory
        del data, obs_t, logic_t, actions_t, ep_nums_t
        gc.collect()
        
        v_save_path = os.path.join(config_dir, f"valuations_{args.rules}.pt")
        print(f"Saving precomputed valuations to {v_save_path}")
        torch.save(valuations, v_save_path)
        print("Valuation precomputation complete.\n")

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(agent.unwrapped_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.lr_patience,
    )

    best_mean_reward = -float('inf')
    best_loss        = float('inf')
    patience         = 12
    patience_counter = 0
    results_log      = []
    last_email_time  = time.time()

    # ── Graceful interrupt handler ────────────────────────────────────────────
    # On Ctrl+C or SIGTERM: save current model + CSV, then exit cleanly.
    def _emergency_save(signum, frame):
        print(f"\n\n[INTERRUPTED] Signal {signum} received. Saving checkpoint and results...")
        try:
            interrupt_path = os.path.join(run_dir, "interrupted.pth")
            agent.save(interrupt_path)
            print(f"  Model saved to {interrupt_path}")
        except Exception as e:
            print(f"  WARNING: Could not save model: {e}")
        try:
            if results_log:
                csv_path = os.path.join(run_dir, f"results_{args.epochs}_eval_{args.num_eval_episodes}_interrupted.csv")
                pd.DataFrame(results_log).to_csv(csv_path, index=False)
                print(f"  Results CSV saved to {csv_path}")
                # Also print a quick summary to stdout
                print("  Learning curve so far:")
                for res in results_log:
                    print(f"    Epoch {res['epoch']:3d} | Loss {res.get('train_loss', float('nan')):.4f} "
                          f"| T-Acc {res.get('train_acc', 0.0):.4f} "
                          f"| Score {res['mean_reward']:.2f} ± {res['std_reward']:.2f}")
        except Exception as e:
            print(f"  WARNING: Could not save CSV: {e}")
        print("[INTERRUPTED] Done. Exiting.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT,  _emergency_save)
    signal.signal(signal.SIGTERM, _emergency_save)

    print("Starting epoch-based training over full dataset...")

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} (all {len(unique_eps)} episodes) ---")

        epoch_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0),
        )

        # ── DIAGNOSTIC: run with --diagnose flag to profile then exit ────────
        # Usage: python train_il.py --diagnose [your other args]
        if args.diagnose and epoch == 0:
            run_diagnostics(agent, epoch_loader, device, args)

        agent.model.train()
        total_loss, n_batches        = 0.0, 0
        train_correct, train_samples = 0, 0

        pbar = tqdm(epoch_loader, desc=f"Epoch {epoch+1}")
        for states, actions, gazes, ep_nums, step_idxs in pbar:
            states  = states.to(device)
            actions = actions.to(device)
            gazes   = gazes.to(device)
            batch_size = states.size(0)

            # ── Build vT batch from pre-computed valuations ──────────────────
            vT_batch = None
            if valuations is not None:
                vT_list = []
                for ep_id, s_idx in zip(ep_nums.tolist(), step_idxs.tolist()):
                    if ep_id in valuations and s_idx < len(valuations[ep_id]):
                        vT_list.append(valuations[ep_id][s_idx])
                    else:
                        vT_list.append(torch.zeros(len(agent.unwrapped_model.atoms), device=device))
                vT_batch = torch.stack(vT_list).to(device)

            # Single forward pass — update() returns probs & action_scores
            loss, probs, action_scores = agent.update(
                states, actions,
                gazes if use_gaze else None,
                vT=vT_batch,
                loss_type=args.loss,
            )
            # # DataParallel returns sum of losses across GPUs — normalize to per-sample
            # if isinstance(agent.model, nn.DataParallel):
            #     loss = loss / torch.cuda.device_count()

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(agent.unwrapped_model.parameters(), 1.0)
            optimizer.step()

            num_gpus = torch.cuda.device_count()
            loss_val = loss.item() / max(num_gpus, 1)  # normalized logged loss

            train_correct  += (action_scores.argmax(dim=1) == actions).sum().item()
            train_samples  += batch_size
            total_loss     += loss_val
            n_batches      += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_loss      = total_loss / max(n_batches, 1)
        avg_train_acc = train_correct / max(train_samples, 1)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f} | Train Acc: {avg_train_acc:.4f}")

        # ── Validation ───────────────────────────────────────────────────────
        avg_val_acc = 0.0
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
                    B       = states.size(0)

                    vT_batch = None
                    if valuations is not None:
                        vT_list = [
                            valuations[ep_id.item()][s_idx.item()]
                            for ep_id, s_idx in zip(ep_nums, step_idxs)
                        ]
                        vT_batch = torch.stack(vT_list).to(device)

                    probs, action_scores = agent.predict(
                        states,
                        gazes if use_gaze else None,
                        vT=vT_batch,
                    )

                    val_correct += (action_scores.argmax(dim=1) == actions).sum().item()
                    val_samples += B
                    log_scores = torch.log(action_scores.clamp(min=eps_nll))
                    val_loss    += nn.NLLLoss()(log_scores, actions).item()
                    val_n       += 1

            avg_val_acc = val_correct / max(val_samples, 1)
            avg_val_loss = val_loss / max(val_n, 1)
            print(f"Epoch {epoch+1} Val   Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        scheduler.step(avg_val_acc if val_loader else -avg_loss)

        # ── Environment Evaluation ────────────────────────────────────────────
        if (epoch + 1) % args.eval_interval == 0:
            rewards = evaluate_parallel(
                agent, env_name=args.env,
                num_episodes=args.num_eval_episodes, seed=args.seed,
                max_steps=args.eval_max_steps,
                gaze_model_path=(args.gaze_model_path if use_gaze or unnormalized else None),
                use_gaze=use_gaze,
                num_workers=(16 if use_gaze or unnormalized else None),
                gabril_compat=True,
                train_run=True
            )
            mean_reward, std_reward = np.mean(rewards), np.std(rewards)
        else:
            mean_reward, std_reward = float('nan'), float('nan')
        print(f"Epoch {epoch+1} Eval Score: Mean={mean_reward:.2f}  Std={std_reward:.2f}")

        results_log.append({
            'epoch':        epoch + 1,
            'trajectory':   'all',
            'num_episodes': num_iters,
            'mean_reward':  mean_reward,
            'std_reward':   std_reward,
            'train_loss':   avg_loss,
            'train_acc':    avg_train_acc,
            'val_acc':      avg_val_acc,
            'val_loss':     avg_val_loss,
            'gaze':         use_gaze,
        })

        # ── Save checkpoint ───────────────────────────────────────────────────
        save_path = f"{run_dir}/epoch_{epoch+1}.pth"
        agent.save(save_path)
        print(f"Saved model to {save_path}")

        # ── Rolling CSV save (every eval_interval epochs) ─────────────────────
        if (epoch + 1) % args.eval_interval == 0 and results_log:
            _csv_path = os.path.join(run_dir, f"results_{args.epochs}_eval_{args.num_eval_episodes}.csv")
            pd.DataFrame(results_log).to_csv(_csv_path, index=False)
            print(f"Results CSV updated → {_csv_path}")

        # ── Best model / early stopping ───────────────────────────────────────
        if avg_val_loss < best_loss:
            best_loss        = avg_val_loss
            patience_counter = 0
            print(f"--- New Best Val Loss! {best_loss:.4f} ---")
        else:
            patience_counter += 1
            print(f"--- No improvement. Patience: {patience_counter}/{patience} ---")
            if patience_counter >= patience:
                print(f"--- Early stopping after {epoch+1} epochs ---")
                break

        # ── Periodic email ────────────────────────────────────────────────────
        if args.send_email:
            current_time = time.time()
            if (current_time - last_email_time) / 60 >= args.email_interval:
                send_run_update(args, results_log, epoch + 1, {
                    'last_loss':   avg_loss,
                    'train_acc':   avg_train_acc,
                    'val_acc':     avg_val_acc,
                    'best_loss':   best_loss,
                    'last_reward': mean_reward,
                    'best_reward': best_mean_reward,
                }, task_name="NSFR IL Training")
                last_email_time = current_time

    # ── Learning curve summary ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("LEARNING CURVE")
    print("=" * 50)
    for res in results_log:
        print(
            f"  Epoch {res['epoch']:3d} | Traj {res.get('trajectory', '-')} "
            f"| Loss {res.get('train_loss', float('nan')):.4f} "
            f"| T-Acc {res.get('train_acc', 0.0):.4f} "
            f"| V-Acc {res.get('val_acc', 0.0):.4f} "
            f"| Score {res['mean_reward']:.2f} ± {res['std_reward']:.2f}"
        )
    print("=" * 50)

    results_df       = pd.DataFrame(results_log)
    results_csv_path = os.path.join(run_dir, f"results_{args.epochs}.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    # ── Final email ───────────────────────────────────────────────────────────
    if args.send_email:
        last_res = results_log[-1] if results_log else {}
        send_run_update(args, results_log, last_res.get('epoch', 0), {
            'train_loss':  last_res.get('train_loss', 0.0),
            'train_acc':   last_res.get('train_acc', 0.0),
            'val_acc':     last_res.get('val_acc', 0.0),
            'best_loss':   best_loss,
            'last_reward': last_res.get('mean_reward', 0.0),
            'best_reward': best_mean_reward,
        }, is_final=True, task_name="NSFR IL Training")


if __name__ == "__main__":
    main()