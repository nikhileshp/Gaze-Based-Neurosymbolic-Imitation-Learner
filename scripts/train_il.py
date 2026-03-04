import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from ocatari.core import OCAtari
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from nudge.agents.imitation_agent import ImitationAgent
from nudge.env import NudgeBaseEnv
from nudge.utils import make_deterministic
from tqdm import tqdm
from collections import Counter
from evaluate_model import evaluate
from scripts.data_utils import PtDataset, ExpertDataset, PRIMITIVE_ACTION_MAP, CSV_FILE, BASE_IMAGE_DIR
from scripts.email_me import send_email
import time

# Dataset classes moved to scripts/data_utils.py

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def add(self, states, actions, gazes, losses, ep_nums=None, step_idxs=None):
        """Add a batch of experiences with their associated losses as priorities."""
        if torch.is_tensor(states): states = states.cpu()
        if torch.is_tensor(actions): actions = actions.cpu()
        if torch.is_tensor(gazes): gazes = gazes.cpu()
        if torch.is_tensor(ep_nums): ep_nums = ep_nums.cpu()
        if torch.is_tensor(step_idxs): step_idxs = step_idxs.cpu()
        
        for i in range(len(states)):
            experience = (
                states[i], 
                actions[i], 
                gazes[i], 
                ep_nums[i] if ep_nums is not None else -1, 
                step_idxs[i] if step_idxs is not None else -1
            )
            priority = (abs(losses[i]) + 1e-6) ** self.alpha
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
                self.priorities.append(priority)
            else:
                self.buffer[self.position] = experience
                self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None
        
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        states = torch.stack([s[0] for s in samples])
        actions = torch.stack([s[1] for s in samples])
        gazes = torch.stack([s[2] for s in samples])
        ep_nums = torch.stack([torch.tensor(s[3]) for s in samples])
        step_idxs = torch.stack([torch.tensor(s[4]) for s in samples])
        
        return states, actions, gazes, ep_nums, step_idxs, indices

    def update_priorities(self, indices, losses):
        for idx, loss in zip(indices, losses):
            self.priorities[idx] = (abs(loss) + 1e-6) ** self.alpha

    def __len__(self):
        return len(self.buffer)


# def evaluate(agent, env, num_episodes=5, seed=42):
#     agent.model.eval()
#     rewards = []
#     for i in range(num_episodes):
#         state = env.reset(i+seed)
#         done = False
#         episode_reward = 0
#         while not done:
#             logic_state, _ = state
#             logic_state_tensor = torch.tensor(logic_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            
#             action = agent.act(logic_state_tensor)
            
#             prednames = agent.model.get_prednames()
#             predicate = prednames[action]
            
#             state, reward, done = env.step(predicate)
#             episode_reward += reward
#         rewards.append(episode_reward)
#         print(f"Episode {i+1} Reward: {episode_reward}")
#     agent.model.train()
#     return rewards


def format_results_table(results_log):
    if not results_log:
        return "No results yet."
    
    header = f"{'Epoch':<6} | {'Loss':<10} | {'Mean Reward':<12} | {'Std Reward':<12}"
    divider = "-" * len(header)
    rows = []
    for res in results_log:
        epoch = res.get('epoch', '-')
        loss = res.get('train_loss', 0.0)
        mean_r = res.get('mean_reward', 0.0)
        std_r = res.get('std_reward', 0.0)
        # Handle NaN for mean_reward if evaluation didn't happen this epoch
        mean_r_str = f"{mean_r:<12.2f}" if not np.isnan(mean_r) else f"{'N/A':<12}"
        std_r_str = f"{std_r:<12.2f}" if not np.isnan(std_r) else f"{'N/A':<12}"
        rows.append(f"{epoch:<6} | {loss:<10.4f} | {mean_r_str} | {std_r_str}")
    
    return "\n".join([header, divider] + rows)

def send_run_update(args, results_log, current_epoch, last_loss, best_loss, last_reward, best_reward, is_final=False):
    gaze_status = "With Gaze" if args.use_gaze else "Without Gaze"
    num_ep = args.num_episodes if args.num_episodes is not None else "All"
    
    status_prefix = "Final" if is_final else "Periodic"
    subject = f"Server Run Update: Run {args.env} - NSFR Training - {gaze_status} | {args.rules} Rules | Using {num_ep} episodes"
    
    # Format NaN rewards
    last_reward_str = f"{last_reward:.2f}" if not np.isnan(last_reward) else "N/A"
    best_reward_str = f"{best_reward:.2f}" if not np.isnan(best_reward) else "N/A"

    body = f"""
Status: {status_prefix} Update
Environment: {args.env}
Ruleset: {args.rules}
Gaze: {gaze_status}
Episodes: {num_ep}

Current Epoch: {current_epoch}
Last Train Loss: {last_loss:.4f}
Best Train Loss: {best_loss:.4f}
Last Mean Reward: {last_reward_str}
Best Mean Reward: {best_reward_str}

Progress Table:
{format_results_table(results_log)}
"""
    send_email(subject, body.strip())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="seaquest", help="Environment name")
    parser.add_argument("--rules", type=str, default="new", help="Ruleset name")
    # .pt dataset (new preferred path)
    parser.add_argument("--dataset", type=str, default=None, help="Path to .pt dataset file (from convert_trajectories_to_pt.py)")
    # Legacy CSV path
    parser.add_argument("--data_path", type=str, default=None, help="Path to expert data (legacy CSV/pkl)")
    parser.add_argument("--epochs", type=int, default=16, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val_split", type=float, default=0.00, help="Fraction of data to use as validation (0 = no validation)")
    parser.add_argument("--gaze_threshold", type=float, default=50.0, help="Threshold for gaze-based valuation scaling")
    parser.add_argument("--use_gaze", action="store_true", help="Use gaze data for training")
    parser.add_argument("--use_gazemap", default=False, action="store_true", help="Use full gaze map for valuation")
    parser.add_argument("--gaze_model_path", type=str, default="models/gaze_predictor/seaquest_gaze_predictor_sigma_10.pth", help="Path to the .pth gaze predictor weights")
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to load from .pt dataset")
    parser.add_argument("--sort_by", type=str, default=None, choices=['length', 'reward_per_step'], help="How to sort episodes before selection")
    parser.add_argument("--valuation_path", type=str, default=None, help="Path to pre-computed valuation.pt")
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluate every N epochs (2 = every other epoch)")
    parser.add_argument("--eval_max_steps", type=int, default=10000, help="Max game steps per eval episode")
    parser.add_argument("--send_email", action="store_true", help="Enable periodic email updates")
    parser.add_argument("--email_interval", type=int, default=30, help="Interval in minutes between email updates")
    args = parser.parse_args()

    if args.use_gazemap:
        args.use_gaze = True
        from scripts.gaze_predictor import Human_Gaze_Predictor
        print(f"Initializing Gaze Predictor from {args.gaze_model_path}...")
        gaze_predictor = Human_Gaze_Predictor(args.env)
        gaze_predictor.init_model(args.gaze_model_path)
        gaze_predictor.model.eval()
    
    # args.use_gaze is now properly synchronized with args.use_gazemap


    make_deterministic(args.seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "cpu":
        device_name = args.device
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Initialize Environment (for evaluation and model init)
    # mode='logic' is required to get logic states
    env = NudgeBaseEnv.from_name(args.env, mode='logic')

    # Initialize Agent
    print(f"Initializing ImitationAgent for {args.env} with rules {args.rules}...")
    agent_gaze_threshold = args.gaze_threshold if args.use_gaze else None
    agent = ImitationAgent(args.env, args.rules, device, gaze_threshold=agent_gaze_threshold)

    # Determine trajectories to iterate over
    # We look at train.csv to find all trajectory numbers
    data_path = args.data_path or CSV_FILE
    if os.path.exists(data_path) and not args.dataset:
        full_df = pd.read_csv(data_path)
        if 'trajectory_number' in full_df.columns:
            trajectories = sorted(full_df['trajectory_number'].unique())
            print(f"Found {len(trajectories)} trajectories: {trajectories}")
        else:
            print("Warning: 'trajectory_number' column not found in CSV. Using single trajectory [1].")
            trajectories = [1]


    # Best model/loss tracking for both loops
    best_mean_reward = -float('inf')
    best_loss = float('inf')

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.dataset:
        # New .pt-based flow: one big dataset, epoch-based training
        full_dataset = PtDataset(args.dataset, use_gaze=args.use_gazemap, num_episodes=args.num_episodes, sort_by=args.sort_by)
        if args.limit:
            full_dataset.logic   = full_dataset.logic[:args.limit]
            full_dataset.actions = full_dataset.actions[:args.limit]
            full_dataset.gaze    = full_dataset.gaze[:args.limit]

        if args.val_split > 0:
            val_n   = max(1, int(len(full_dataset) * args.val_split))
            train_n = len(full_dataset) - val_n
            train_dataset, val_dataset = random_split(
                full_dataset, [train_n, val_n],
                generator=torch.Generator().manual_seed(args.seed)
            )
            print(f"Train: {train_n} samples | Val: {val_n} samples")
        else:
            train_dataset = full_dataset
            val_dataset   = None
            print(f"Train: {len(train_dataset)} samples (no validation split)")

        # Get unique episodes from train_dataset for per-epoch selection
        if isinstance(train_dataset, torch.utils.data.Subset):
            ep_nums_all = train_dataset.dataset.ep_nums[train_dataset.indices]
        else:
            ep_nums_all = train_dataset.ep_nums
        unique_eps = torch.unique(ep_nums_all)
        print(f"Found {len(unique_eps)} unique episodes in training set.")
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        ) if val_dataset else None

        # ── Epoch-based training loop ─────────────────────────────────────────
        print("Starting epoch-based training over full dataset...")
        results_log = []
        os.makedirs(f"models/nsfr/{args.env}", exist_ok=True)
        os.makedirs("out/imitation", exist_ok=True)
        gaze_str = "gaze" if args.use_gazemap else "no_gaze"
        num_iters = args.num_episodes if args.num_episodes is not None else "full"
        experiment_str = f"{args.env}_{args.rules}_il_lr_{args.lr}_num_ep_{num_iters}"

        # Load pre-computed valuations if they exist
        valuations = None
        v_path = args.valuation_path
        if v_path is None:
            # Auto-detect path
            if args.use_gazemap:
                v_path = f"models/nsfr/{args.env}/gaze/valuation.pt"
            else:
                v_path = f"models/nsfr/{args.env}/_no_gaze/valuation.pt"
        
        if os.path.exists(v_path):
            print(f"Loading pre-computed valuations from {v_path}...")
            # Use weights_only=False because valuation.pt involves complex types (list of dicts)
            valuations_raw = torch.load(v_path, map_location=device, weights_only=False)
            
            # Convert list-of-dicts format to ep_id-indexed dict of lists
            if isinstance(valuations_raw, dict) and 'data' in valuations_raw and isinstance(valuations_raw['data'], list):
                print("  Reformatting valuations from list of dicts to episode-based lists...")
                valuations_indexed = {}
                for item in valuations_raw['data']:
                    frame_id = item['frame_id'] # 'ep_0_step_0'
                    try:
                        parts = frame_id.split('_')
                        # Format: 'ep' (0), ID (1), 'step' (2), IDX (3)
                        ep_id = int(parts[1])
                        step_idx = int(parts[3])
                        
                        if ep_id not in valuations_indexed:
                            valuations_indexed[ep_id] = {}
                        
                        atoms = item['atoms']
                        if not isinstance(atoms, torch.Tensor):
                            atoms = torch.tensor(atoms, dtype=torch.float32)
                        valuations_indexed[ep_id][step_idx] = atoms.to(device)
                    except (IndexError, ValueError):
                        continue
                
                # Convert inner dicts to sorted lists for faster access
                valuations = {}
                for ep_id in valuations_indexed:
                    max_step = max(valuations_indexed[ep_id].keys())
                    v_list = [torch.zeros(len(agent.model.atoms)).to(device)] * (max_step + 1)
                    for s_idx, v in valuations_indexed[ep_id].items():
                        v_list[s_idx] = v
                    valuations[ep_id] = v_list
                print(f"  Loaded valuations for {len(valuations)} episodes.")
            else:
                valuations = valuations_raw
        else:
            print(f"No pre-computed valuations found at {v_path}. Training from logic states.")

        # replay_buffer = PrioritizedReplayBuffer(capacity=25000)
        replay_steps = 1
        patience = 12
        patience_counter = 0

        # Use Adam for better stability with scaled logits
        optimizer = torch.optim.Adam(agent.model.parameters(), lr=args.lr)

        last_email_time = time.time()
        for epoch in range(args.epochs):
            
            print(f"\n--- Epoch {epoch+1}/{args.epochs} (all {len(unique_eps)} episodes) ---")

            # Option A: train on ALL N episodes every epoch for clean sample efficiency measurement
            epoch_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device.type == 'cuda'),
            )

            agent.model.train()

            total_loss, n_batches = 0.0, 0
            pbar = tqdm(epoch_loader, desc=f"Epoch {epoch+1}")
            for states, actions, gazes, ep_nums, step_idxs in pbar:
                states, actions, gazes = states.to(device), actions.to(device), gazes.to(device)

                # Look up pre-computed valuations if available
                vT_batch = None
                if valuations is not None:
                    # Construct batch from pre-computed V_T
                    vT_list = []
                    for ep_id, step_idx in zip(ep_nums, step_idxs):
                        ep_id, s_idx = ep_id.item(), step_idx.item()
                        if ep_id in valuations and s_idx < len(valuations[ep_id]):
                            vT_list.append(valuations[ep_id][s_idx])
                        else:
                            # Fallback if step missing
                            vT_list.append(torch.zeros(len(agent.model.atoms)).to(device))
                    vT_batch = torch.stack(vT_list).to(device)

                # ── UPDATE A: Discovery (Fresh Batch) ──
                loss, ind_losses = agent.update(states, actions, gazes if args.use_gaze else None, vT=vT_batch)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
                optimizer.step()
                
                loss_val = loss.item()
                
                # Add to PER
                # replay_buffer.add(states, actions, gazes, ind_losses, ep_nums, step_idxs)
                
                # ── UPDATE B: Focus (Replay Batch) ──
                # if len(replay_buffer) >= args.batch_size:
                #     for _ in range(replay_steps):
                #         sample = replay_buffer.sample(args.batch_size)
                #         if sample:
                #             s_r, a_r, g_r, e_r, st_r, indices = sample
                #             s_r, a_r, g_r = s_r.to(device), a_r.to(device), g_r.to(device)
                            
                #             vT_r = None
                #             if valuations is not None:
                #                 vT_r_list = []
                #                 for ep_id, s_idx in zip(e_r, st_r):
                #                     ep_id, s_idx = ep_id.item(), s_idx.item()
                #                     vT_r_list.append(valuations[ep_id][s_idx] if ep_id in valuations else torch.zeros(len(agent.model.atoms)).to(device))
                #                 vT_r = torch.stack(vT_r_list).to(device)
                            
                #             l_r, ind_l_r = agent.update(s_r, a_r, g_r if args.use_gaze else None, vT=vT_r)
                #             replay_buffer.update_priorities(indices, ind_l_r)

                total_loss += loss_val
                n_batches  += 1
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            # Optional PER Replay for full dataset (sampling from recent successes/failures)
            # 1. Add some samples to buffer from this epoch
            # ... (omitted for brevity in full dataset loop to keep it fast, or add if needed)

            avg_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

            # Optional validation pass
            if val_loader:
                agent.model.eval()
                val_loss, val_n = 0.0, 0
                with torch.no_grad():
                    for states, actions, gazes, ep_nums, step_idxs in val_loader:
                        states, actions, gazes = states.to(device), actions.to(device), gazes.to(device)
                        B = states.size(0)
                        
                        # Use pre-computed valuations for validation if available
                        if valuations is not None:
                            vT_list = []
                            for ep_id, step_idx in zip(ep_nums, step_idxs):
                                vT_list.append(valuations[ep_id.item()][step_idx.item()])
                            vT_batch = torch.stack(vT_list).to(device)
                            probs = agent.model.get_predictions(vT_batch, prednames=agent.model.prednames)
                        else:
                            probs = agent.model(states, gazes if args.use_gaze else None)
                        
                        # Ensure model is in same mode as trainer's update for aggregation
                        # Match the logic in agent.update
                        action_rule_probs = {idx: [] for idx in range(6)}
                        for i, pred in enumerate(agent.model.get_prednames()):
                            prefix = pred.split('_')[0]
                            if prefix in PRIMITIVE_ACTION_MAP:
                                idx = PRIMITIVE_ACTION_MAP[prefix]
                                action_rule_probs[idx].append(probs[:, i])
                        
                        action_scores_list = []
                        for idx in range(6):
                            if action_rule_probs[idx]:
                                stacked = torch.stack(action_rule_probs[idx], dim=1)
                                m, _ = torch.max(stacked, dim=1)
                                action_scores_list.append(m)
                            else:
                                action_scores_list.append(torch.zeros(B, device=device))
                        
                        action_scores = torch.stack(action_scores_list, dim=1) # (B, 6)
                            
                        # Independent Log-Probabilities (No Softmax, matching ImitationAgent.update)
                        eps = 1e-10
                        log_action_probs = torch.log(action_scores + eps)
                        
                        loss = nn.NLLLoss()(log_action_probs, actions) # mean loss
                        val_loss += loss.item()
                        val_n += 1
                print(f"Epoch {epoch+1} Val Loss:   {val_loss / max(val_n, 1):.4f}")
            # Evaluation in environment (every eval_interval epochs)
            if (epoch + 1) % args.eval_interval == 0:
                rewards = evaluate(agent, env, num_episodes=5, seed=args.seed, valuation_interval=0, log_interval=0,
                                   max_steps=args.eval_max_steps,
                                   gaze_predictor=(gaze_predictor if args.use_gazemap else None))
                mean_reward, std_reward = np.mean(rewards), np.std(rewards)
            else:
                mean_reward, std_reward = float('nan'), float('nan')
            print(f"Epoch {epoch+1} Eval Score: Mean={mean_reward:.2f}  Std={std_reward:.2f}")

            results_log.append({
                'epoch': epoch + 1, 'trajectory': 'all',
                'num_episodes': num_iters,
                'mean_reward': mean_reward, 'std_reward': std_reward,
                'train_loss': avg_loss, 'gaze': args.use_gaze,
            })

            run_dir = f"models/nsfr/{args.env}/{gaze_str}/{num_iters}_ep"
            os.makedirs(run_dir, exist_ok=True)
            save_path = f"{run_dir}/epoch_{epoch+1}.pth"
            agent.save(save_path)
            print(f"Saved model to {save_path}")

            # Best Model and Early Stopping
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                patience_counter = 0
                best_model_path = f"{run_dir}/best.pth"
                agent.save(best_model_path)
                print(f"--- New Best Model! Reward: {best_mean_reward:.2f}. Saved to {best_model_path} ---")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                print(f"--- New Best Loss! Loss: {best_loss:.4f}. Saved to {save_path} ---")
            else:
                patience_counter += 1
                print(f"--- No improvement. Patience: {patience_counter}/{patience} ---")
                if patience_counter >= patience:
                    print(f"--- Early stopping triggered after {epoch+1} epochs ---")
                    break
            
            # Periodic Email Update
            if args.send_email:
                current_time = time.time()
                if (current_time - last_email_time) / 60 >= args.email_interval:
                    send_run_update(args, results_log, epoch + 1, avg_loss, best_loss, mean_reward, best_mean_reward)
                    last_email_time = current_time
            

    else:
        # ── Legacy per-trajectory loop ────────────────────────────────────────
        data_path = args.data_path or CSV_FILE
        if os.path.exists(data_path):
            full_df = pd.read_csv(data_path)
            trajectories = sorted(full_df['trajectory_number'].unique()) if 'trajectory_number' in full_df.columns else [1]
        else:
            trajectories = [1]

        # Training Loop
        print("Starting iterative training by trajectory...")
        results_log = []
        
        # Initialize PER, Early Stopping and Best Model tracking
        # replay_buffer = PrioritizedReplayBuffer(capacity=20000)
        patience = 5
        patience_counter = 0
        replay_steps = 5 # Number of replay batches per trajectory epoch
        
        # Use args.epochs as the number of trajectories to process if it's less than total trajectories
        num_iters = min(args.epochs, len(trajectories))
        gaze_str = "gaze" if args.use_gazemap else "no_gaze"
        experiment_str = f"{args.env}_{args.rules}_il_lr_{args.lr}_num_ep_{num_iters}"
        last_email_time = time.time()
        for epoch in range(num_iters):
            traj_num = trajectories[epoch]
            print(f"\n--- Epoch {epoch+1}/{num_iters} (Trajectory {traj_num}) ---")

            dataset = ExpertDataset(args.env, agent.model.prednames, args.data_path, nudge_env=env, limit=args.limit, use_gazemap=args.use_gazemap, trajectory=traj_num)
            if len(dataset) == 0:
                print(f"Skipping empty trajectory {traj_num}")
                continue

            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Training Traj {traj_num}")
            agent.model.train()
            for states, actions, gazes in pbar:
                states  = states.to(device)
                actions = actions.to(device)
                gazes   = gazes.to(device)

                # Perform update using the agent's unified method
                loss, _ = agent.update(states, actions, gazes if args.use_gaze else None)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
                optimizer.step()
                
                loss_val = loss.item()
                total_loss += loss_val
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})
                
            # 1. Add this trajectory's data to the Replay Buffer
            print(f"Adding trajectory {traj_num} data to PER buffer...")
            agent.model.eval()
            with torch.no_grad():
                all_states, all_actions, all_gazes, all_losses = [], [], [], []
                for states, actions, gazes in DataLoader(dataset, batch_size=args.batch_size):
                    states, actions, gazes = states.to(device), actions.to(device), gazes.to(device)
                    probs = agent.model(states, gazes if args.use_gaze else None)
                    B = probs.size(0)
                    act_p = torch.zeros(B, 6, device=device)
                    for i, pred in enumerate(agent.model.get_prednames()):
                        prefix = pred.split('_')[0]
                        if prefix in PRIMITIVE_ACTION_MAP:
                            act_p[:, PRIMITIVE_ACTION_MAP[prefix]] += probs[:, i]
                    log_p = torch.log(act_p + 1e-10)
                    ind_loss = torch.nn.functional.nll_loss(log_p, actions, reduction='none')
                    all_states.append(states.cpu()); all_actions.append(actions.cpu())
                    all_gazes.append(gazes.cpu()); all_losses.append(ind_loss.cpu())
                # replay_buffer.add(torch.cat(all_states), torch.cat(all_actions), torch.cat(all_gazes), torch.cat(all_losses))

            # 2. Perform Replay Training
            # if len(replay_buffer) >= args.batch_size:
            #     print(f"Performing {replay_steps} replay steps from PER buffer...")
            #     agent.model.train()
            #     for _ in range(replay_steps):
            #         sample = replay_buffer.sample(args.batch_size)
            #         if not sample: break
            #         s_b, a_b, g_b, indices = sample
            #         s_b, a_b, g_b = s_b.to(device), a_b.to(device), g_b.to(device)
            #         p_b = agent.model(s_b, g_b if args.use_gaze else None)
            #         act_p_b = torch.zeros(s_b.size(0), 6, device=device)
            #         for i, pred in enumerate(agent.model.get_prednames()):
            #             prefix = pred.split('_')[0]
            #             if prefix in PRIMITIVE_ACTION_MAP:
            #                 act_p_b[:, PRIMITIVE_ACTION_MAP[prefix]] += p_b[:, i]
            #         log_p_b = torch.log(act_p_b + 1e-10)
            #         l_b_ind = torch.nn.functional.nll_loss(log_p_b, a_b, reduction='none')
            #         l_b_ind_mean, l_b_ind = agent.update(s_b, a_b, g_b if args.use_gaze else None)
            #         replay_buffer.update_priorities(indices, l_b_ind)

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

            # Evaluation
            if args.use_gazemap:
                rewards = evaluate(agent, env, num_episodes=5, seed=args.seed, gaze_predictor=gaze_predictor)
            else:
                rewards = evaluate(agent, env, num_episodes=5, seed=args.seed, gaze_predictor=None)
            mean_reward, std_reward = np.mean(rewards), np.std(rewards)
            print(f"Epoch {epoch+1} Evaluation Score: Mean={mean_reward:.2f}, Std={std_reward:.2f}")

            results_log.append({'epoch': epoch+1, 'trajectory': traj_num, 'num_episodes': num_iters, 'mean_reward': mean_reward, 'std_reward': std_reward, 'train_loss': avg_loss, 'gaze': args.use_gaze})

            run_dir = f"models/nsfr/{args.env}/{gaze_str}/{num_iters}_ep"
            os.makedirs(run_dir, exist_ok=True)
            save_path = f"{run_dir}/epoch_{epoch+1}.pth"
            agent.save(save_path)
            
            # Use reward AND loss for early stopping 
            improved = False
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                improved = True
                best_model_path = f"{run_dir}/best.pth"
                agent.save(best_model_path)
                print(f"--- New Best Model! Reward: {best_mean_reward:.2f}. Saved to {best_model_path} ---")
            
            if avg_loss < best_loss: # Use < for minimizing loss
                best_loss = avg_loss
                improved = True
                print(f"--- New Best Loss: {best_loss:.4f} ---")
            
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"--- No improvement in reward/loss. Patience: {patience_counter}/{patience} ---")
                if patience_counter >= patience:
                    print(f"--- Early stopping triggered after {epoch+1} epochs ---")
                    break

            # Periodic Email Update
            if args.send_email:
                current_time = time.time()
                if (current_time - last_email_time) / 60 >= args.email_interval:
                    send_run_update(args, results_log, epoch + 1, avg_loss, best_loss, mean_reward, best_mean_reward)
                    last_email_time = current_time

    # Print and save final learning curve
    print("\n" + "="*50)
    print("LEARNING CURVE")
    print("="*50)
    for res in results_log:
        traj = res.get('trajectory', '-')
        loss = res.get('train_loss', float('nan'))
        print(f"  Epoch {res['epoch']:3d} | Traj {traj} | Loss {loss:.4f} | Score {res['mean_reward']:.2f} ± {res['std_reward']:.2f}")
    print("="*50)

    results_df = pd.DataFrame(results_log)
    # Save CSV co-located with the models for this run
    run_dir = f"models/nsfr/{args.env}/{gaze_str}/{num_iters}_ep"
    os.makedirs(run_dir, exist_ok=True)
    results_csv_path = os.path.join(run_dir, f"results_lr_{args.lr}.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    # Final Email Update
    if args.send_email:
        # Get final values from the last entry in results_log
        last_res = results_log[-1] if results_log else {}
        send_run_update(args, results_log, 
                        last_res.get('epoch', 0), 
                        last_res.get('train_loss', 0.0), 
                        best_loss, 
                        last_res.get('mean_reward', 0.0), 
                        best_mean_reward,
                        is_final=True)

if __name__ == "__main__":
    main()
