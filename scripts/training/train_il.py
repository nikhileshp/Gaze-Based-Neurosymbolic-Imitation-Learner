# import os
# import argparse
# import torch
# import torch.nn as nn
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# from core.utils.utils import get_primitive_action_map
# from core.utils.utils import PtDataset
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
# from nsfr.agents.imitation_agent import ImitationAgent
# from nsfr.utils import make_deterministic
# from nsfr.env import NSFRBaseEnv
# from core.utils.utils import (
#     set_seed_everywhere, format_results_table, 
#     send_run_update, load_pt_dataset
# )
# from scripts.evaluation.evaluate_model import evaluate
# import time

# # Dataset classes and util functions are in core/utils/utils.py


# # def evaluate(agent, env, num_episodes=5, seed=42):
# #     agent.model.eval()
# #     rewards = []
# #     for i in range(num_episodes):
# #         state = env.reset(i+seed)
# #         done = False
# #         episode_reward = 0
# #         while not done:
# #             logic_state, _ = state
# #             logic_state_tensor = torch.tensor(logic_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            
# #             action = agent.act(logic_state_tensor)
            
# #             prednames = agent.model.get_prednames()
# #             predicate = prednames[action]
            
# #             state, reward, done = env.step(predicate)
# #             episode_reward += reward
# #         rewards.append(episode_reward)
# #         print(f"Episode {i+1} Reward: {episode_reward}")
# #     agent.model.train()
# #     return rewards



# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env", type=str, default="seaquest", help="Environment name")
#     parser.add_argument("--rules", type=str, default="new", help="Ruleset name")
#     # .pt dataset (new preferred path)
#     parser.add_argument("--dataset", type=str, default=None, help="Path to .pt dataset file (from convert_trajectories_to_pt.py)")
#     parser.add_argument("--epochs", type=int, default=16, help="Number of training epochs")
#     parser.add_argument("--loss", type=str, default="nll", choices=["nll", "bce"],
#                         help="Loss function for imitation: 'nll' (NLLLoss on aggregated scores) or 'bce' (BCELoss, action-independent)")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
#     parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed")
#     parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
#     parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
#     parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
#     parser.add_argument("--val_split", type=float, default=0.05, help="Fraction of data to use as validation (0 = no validation)")
#     parser.add_argument("--lr_patience", type=int, default=3, help="LR reduction patience")
#     parser.add_argument("--gaze_threshold", type=float, default=50.0, help="Threshold for gaze-based valuation scaling")
#     parser.add_argument("--use_gaze", action="store_true", help="Use gaze data for training")
#     parser.add_argument("--use_gazemap", default=False, action="store_true", help="Use full gaze map for valuation")
#     parser.add_argument("--gaze_model_path", type=str, default="trained_models/gaze_predictor/seaquest_gaze_predictor_sigma_10.pth", help="Path to the .pth gaze predictor weights")
#     parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to load from .pt dataset")
#     parser.add_argument("--sort_by", type=str, default=None, choices=['length', 'reward_per_step'], help="How to sort episodes before selection")
#     parser.add_argument("--valuation_path", type=str, default=None, help="Path to pre-computed valuation.pt")
#     parser.add_argument("--eval_interval", type=int, default=5, help="Evaluate every N epochs (2 = every other epoch)")
#     parser.add_argument("--eval_max_steps", type=int, default=10000, help="Max game steps per eval episode")
#     parser.add_argument("--send_email", action="store_true", help="Enable periodic email updates")
#     parser.add_argument("--email_interval", type=int, default=30, help="Interval in minutes between email updates")
#     args = parser.parse_args()

#     if args.use_gazemap:
#         args.use_gaze = True
#         from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
#         print(f"Initializing Gaze Predictor from {args.gaze_model_path}...")
#         gaze_predictor = Human_Gaze_Predictor(args.env)
#         gaze_predictor.init_model(args.gaze_model_path)
#         gaze_predictor.model.eval()
    
#     # args.use_gaze is now properly synchronized with args.use_gazemap

#     PRIMITIVE_ACTION_MAP = get_primitive_action_map(args.env)
#     make_deterministic(args.seed)
#     device_name = "cuda" if torch.cuda.is_available() else "cpu"
#     if args.device != "cpu":
#         device_name = args.device
#     device = torch.device(device_name)
#     print(f"Using device: {device}")

#     # Initialize Environment (for evaluation and model init)
#     # mode='logic' is required to get logic states
#     env = NSFRBaseEnv.from_name(args.env, mode='logic')

#     # Initialize Agent
#     print(f"Initializing ImitationAgent for {args.env} with rules {args.rules}...")
#     agent_gaze_threshold = args.gaze_threshold if args.use_gaze else None
#     agent = ImitationAgent(args.env, args.rules, device, gaze_threshold=agent_gaze_threshold)

#     if not args.dataset:
#         raise ValueError("--dataset is required. The legacy CSV/data_path flow is no longer supported.")


#     # Best model/loss tracking for both loops
#     best_mean_reward = -float('inf')
#     best_loss = float('inf')

#     # ── Dataset ──────────────────────────────────────────────────────────────
#     if args.dataset:
#         # New .pt-based flow: one big dataset, epoch-based training
#         full_dataset = PtDataset(args.dataset, use_gaze=args.use_gazemap, num_episodes=args.num_episodes, sort_by=args.sort_by)
#         if args.limit:
#             full_dataset.logic   = full_dataset.logic[:args.limit]
#             full_dataset.actions = full_dataset.actions[:args.limit]
#             full_dataset.gaze    = full_dataset.gaze[:args.limit]

#         if args.val_split > 0:
#             val_n   = max(1, int(len(full_dataset) * args.val_split))
#             train_n = len(full_dataset) - val_n
#             train_dataset, val_dataset = random_split(
#                 full_dataset, [train_n, val_n],
#                 generator=torch.Generator().manual_seed(args.seed)
#             )
#             print(f"Train: {train_n} samples | Val: {val_n} samples")
#         else:
#             train_dataset = full_dataset
#             val_dataset   = None
#             print(f"Train: {len(train_dataset)} samples (no validation split)")

#         # Get unique episodes from train_dataset for per-epoch selection
#         if isinstance(train_dataset, torch.utils.data.Subset):
#             ep_nums_all = train_dataset.dataset.ep_nums[train_dataset.indices]
#         else:
#             ep_nums_all = train_dataset.ep_nums
#         unique_eps = torch.unique(ep_nums_all)
#         print(f"Found {len(unique_eps)} unique episodes in training set.")
#         val_loader = DataLoader(
#             val_dataset, batch_size=args.batch_size * 2, shuffle=False,
#             num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
#         ) if val_dataset else None

#         # ── Epoch-based training loop ─────────────────────────────────────────
#         print("Starting epoch-based training over full dataset...")
#         results_log = []
#         if args.use_gazemap:
#             os.makedirs(f"trained_models/{args.env}/grail", exist_ok=True)
#         else:
#             os.makedirs(f"trained_models/{args.env}/nsfr",  exist_ok=True)
        
#         num_iters = args.num_episodes if args.num_episodes is not None else "full"
#         experiment_str = f"{args.env}_{args.rules}_il_lr_{args.lr}_num_ep_{num_iters}"

#         # Load pre-computed valuations if they exist
#         valuations = None
#         v_path = args.valuation_path
#         if v_path is None:
#             # Auto-detect path
#             if args.use_gazemap:
#                 v_path = f"trained_models/{args.env}/grail/valuation.pt"
#             else:
#                 v_path = f"trained_models/{args.env}/nsfr/valuation.pt"
        
#         if os.path.exists(v_path):
#             print(f"Loading pre-computed valuations from {v_path}...")
#             # Use weights_only=False because valuation.pt involves complex types (list of dicts)
#             valuations_raw = torch.load(v_path, map_location=device, weights_only=False)
            
#             # Convert list-of-dicts format to ep_id-indexed dict of lists
#             if isinstance(valuations_raw, dict) and 'data' in valuations_raw and isinstance(valuations_raw['data'], list):
#                 print("  Reformatting valuations from list of dicts to episode-based lists...")
#                 valuations_indexed = {}
#                 for item in valuations_raw['data']:
#                     frame_id = item['frame_id'] # 'ep_0_step_0'
#                     try:
#                         parts = frame_id.split('_')
#                         # Format: 'ep' (0), ID (1), 'step' (2), IDX (3)
#                         ep_id = int(parts[1])
#                         step_idx = int(parts[3])
                        
#                         if ep_id not in valuations_indexed:
#                             valuations_indexed[ep_id] = {}
                        
#                         atoms = item['atoms']
#                         if not isinstance(atoms, torch.Tensor):
#                             atoms = torch.tensor(atoms, dtype=torch.float32)
#                         valuations_indexed[ep_id][step_idx] = atoms.to(device)
#                     except (IndexError, ValueError):
#                         continue
                
#                 # Convert inner dicts to sorted lists for faster access
#                 valuations = {}
#                 for ep_id in valuations_indexed:
#                     max_step = max(valuations_indexed[ep_id].keys())
#                     v_list = [torch.zeros(len(agent.model.atoms)).to(device)] * (max_step + 1)
#                     for s_idx, v in valuations_indexed[ep_id].items():
#                         v_list[s_idx] = v
#                     valuations[ep_id] = v_list
#                 print(f"  Loaded valuations for {len(valuations)} episodes.")
#             else:
#                 valuations = valuations_raw
#         else:
#             print(f"No pre-computed valuations found at {v_path}. Training from logic states.")

#         patience = 12
#         patience_counter = 0

#         # Use Adam for better stability with scaled logits
#         optimizer = torch.optim.Adam(agent.model.parameters(), lr=args.lr)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience)

#         last_email_time = time.time()
#         for epoch in range(args.epochs):
            
#             print(f"\n--- Epoch {epoch+1}/{args.epochs} (all {len(unique_eps)} episodes) ---")

#             # Option A: train on ALL N episodes every epoch for clean sample efficiency measurement
#             epoch_loader = DataLoader(
#                 train_dataset,
#                 batch_size=args.batch_size,
#                 shuffle=True,
#                 num_workers=args.num_workers,
#                 pin_memory=(device.type == 'cuda'),
#             )

#             agent.model.train()

#             total_loss, n_batches = 0.0, 0
#             train_correct, train_samples = 0, 0
#             pbar = tqdm(epoch_loader, desc=f"Epoch {epoch+1}")
#             for states, actions, gazes, ep_nums, step_idxs in pbar:
#                 states, actions, gazes = states.to(device), actions.to(device), gazes.to(device)
#                 batch_size = states.size(0)

#                 # Look up pre-computed valuations if available
#                 vT_batch = None
#                 if valuations is not None:
#                     # Construct batch from pre-computed V_T
#                     vT_list = []
#                     for ep_id, step_idx in zip(ep_nums, step_idxs):
#                         ep_id, s_idx = ep_id.item(), step_idx.item()
#                         if ep_id in valuations and s_idx < len(valuations[ep_id]):
#                             vT_list.append(valuations[ep_id][s_idx])
#                         else:
#                             # Fallback if step missing
#                             vT_list.append(torch.zeros(len(agent.model.atoms)).to(device))
#                     vT_batch = torch.stack(vT_list).to(device)

#                 # ── Forward + Loss ── Seems to be having 2 forward passes
#                 # loss, _, _ = agent.update(states, actions, gazes if args.use_gaze else None, vT=vT_batch, loss_type=args.loss)
                
#                 # optimizer.zero_grad()
#                 # loss.backward()
#                 # torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
#                 # optimizer.step()
                
#                 # loss_val = loss.item()

#                 # # Calculate Accuracy (matching train_il_new.py logic)
#                 # with torch.no_grad():
#                 #     if vT_batch is not None:
#                 #         probs = agent.model.get_predictions(vT_batch, prednames=agent.model.prednames)
#                 #     else:
#                 #         probs = agent.model(states, gazes if args.use_gaze else None)
                    
#                 #     num_actions = max(PRIMITIVE_ACTION_MAP.values()) + 1
#                 #     action_probs = torch.zeros(batch_size, num_actions, device=device)
#                 #     for i, pred in enumerate(agent.model.get_prednames()):
#                 #         prefix = pred.split('_')[0]
#                 #         if prefix in PRIMITIVE_ACTION_MAP:
#                 #             act_idx = PRIMITIVE_ACTION_MAP[prefix]
#                 #             action_probs[:, act_idx] += probs[:, i]
                    
#                 #     train_correct += (action_probs.argmax(dim=1) == actions).sum().item()
#                 #     train_samples += batch_size

#                 # MIGHT NEED TO VERIFY - FOLLOWING CODE HAS ONLY 1 FORWARD PASS
#                 # AFTER: one forward pass, reuse action_scores directly
#                 loss, probs, action_scores = agent.update(states, actions, gazes if args.use_gaze else None, vT=vT_batch, loss_type=args.loss)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
#                 optimizer.step()

#                 loss_val = loss.item()

#                 # action_scores already computed inside update() — no second forward pass needed
#                 with torch.no_grad():
#                     train_correct += (action_scores.detach().argmax(dim=1) == actions).sum().item()
#                     train_samples += batch_size

#                 total_loss += loss_val
#                 n_batches  += 1
#                 pbar.set_postfix({"loss": f"{loss_val:.4f}"})

#             avg_loss = total_loss / max(n_batches, 1)
#             avg_train_acc = train_correct / max(train_samples, 1)
#             print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f} | Train Acc: {avg_train_acc:.4f}")

#             # Optional validation pass
#             avg_val_acc = 0.0
#             if val_loader:
#                 agent.model.eval()
#                 val_loss, val_n = 0.0, 0
#                 val_correct, val_samples = 0, 0
#                 with torch.no_grad():
#                     for states, actions, gazes, ep_nums, step_idxs in val_loader:
#                         states, actions, gazes = states.to(device), actions.to(device), gazes.to(device)
#                         B = states.size(0)
                        
#                         # Use pre-computed valuations for validation if available
#                         if valuations is not None:
#                             vT_list = []
#                             for ep_id, step_idx in zip(ep_nums, step_idxs):
#                                 vT_list.append(valuations[ep_id.item()][step_idx.item()])
#                             vT_batch = torch.stack(vT_list).to(device)
#                             probs = agent.model.get_predictions(vT_batch, prednames=agent.model.prednames)
#                         else:
#                             probs = agent.model(states, gazes if args.use_gaze else None)
                        
#                         # Accuracy calculation
#                         num_actions = max(PRIMITIVE_ACTION_MAP.values()) + 1
#                         action_probs = torch.zeros(B, num_actions, device=device)
#                         for i, pred in enumerate(agent.model.get_prednames()):
#                             prefix = pred.split('_')[0]
#                             if prefix in PRIMITIVE_ACTION_MAP:
#                                 act_idx = PRIMITIVE_ACTION_MAP[prefix]
#                                 action_probs[:, act_idx] += probs[:, i]
                        
#                         val_correct += (action_probs.argmax(dim=1) == actions).sum().item()
#                         val_samples += B

#                         # Independent Log-Probabilities (No Softmax, matching ImitationAgent.update)
#                         # Re-calculate score aggregation for loss consistency
#                         action_rule_probs = {idx: [] for idx in range(6)}
#                         for i, pred in enumerate(agent.model.get_prednames()):
#                             prefix = pred.split('_')[0]
#                             if prefix in PRIMITIVE_ACTION_MAP:
#                                 idx = PRIMITIVE_ACTION_MAP[prefix]
#                                 action_rule_probs[idx].append(probs[:, i])
                        
#                         action_scores_list = []
#                         for idx in range(6):
#                             if action_rule_probs[idx]:
#                                 stacked = torch.stack(action_rule_probs[idx], dim=1)
#                                 m, _ = torch.max(stacked, dim=1)
#                                 action_scores_list.append(m)
#                             else:
#                                 action_scores_list.append(torch.zeros(B, device=device))
                        
#                         action_scores = torch.stack(action_scores_list, dim=1) # (B, 6)
#                         eps = 1e-10
#                         log_action_probs = torch.log(action_scores + eps)
                        
#                         loss = nn.NLLLoss()(log_action_probs, actions) # mean loss
#                         val_loss += loss.item()
#                         val_n += 1
                
#                 avg_val_acc = val_correct / max(val_samples, 1)
#                 print(f"Epoch {epoch+1} Val Loss:   {val_loss / max(val_n, 1):.4f} | Val Acc: {avg_val_acc:.4f}")
            
#             # Step the scheduler
#             scheduler.step(avg_val_acc if val_loader else -avg_loss)
#             # Evaluation in environment (every eval_interval epochs)
#             if (epoch + 1) % args.eval_interval == 0:
#                 rewards = evaluate(agent, env, num_episodes=5, seed=args.seed, valuation_interval=0, log_interval=0,
#                                    max_steps=args.eval_max_steps,
#                                    gaze_predictor=(gaze_predictor if args.use_gazemap else None))
#                 mean_reward, std_reward = np.mean(rewards), np.std(rewards)
#             else:
#                 mean_reward, std_reward = float('nan'), float('nan')
#             print(f"Epoch {epoch+1} Eval Score: Mean={mean_reward:.2f}  Std={std_reward:.2f}")

#             results_log.append({
#                 'epoch': epoch + 1, 'trajectory': 'all',
#                 'num_episodes': num_iters,
#                 'mean_reward': mean_reward, 'std_reward': std_reward,
#                 'train_loss': avg_loss, 
#                 'train_acc': avg_train_acc,
#                 'val_acc': avg_val_acc,
#                 'gaze': args.use_gaze,
#             })
            
#             if args.gazemap:
#                 run_dir = f"trained_models/{args.env}/grail/{num_iters}_ep"
#             else:
#                 run_dir = f"trained_models/{args.env}/nsfr/{num_iters}_ep"
#             os.makedirs(run_dir, exist_ok=True)
#             save_path = f"{run_dir}/epoch_{epoch+1}.pth"
#             agent.save(save_path)
#             print(f"Saved model to {save_path}")

#             # Best Model and Early Stopping
#             if mean_reward > best_mean_reward:
#                 best_mean_reward = mean_reward
#                 patience_counter = 0
#                 best_model_path = f"{run_dir}/best.pth"
#                 agent.save(best_model_path)
#                 print(f"--- New Best Model! Reward: {best_mean_reward:.2f}. Saved to {best_model_path} ---")
            
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 patience_counter = 0
#                 print(f"--- New Best Loss! Loss: {best_loss:.4f}. Saved to {save_path} ---")
#             else:
#                 patience_counter += 1
#                 print(f"--- No improvement. Patience: {patience_counter}/{patience} ---")
#                 if patience_counter >= patience:
#                     print(f"--- Early stopping triggered after {epoch+1} epochs ---")
#                     break
            
#             # Periodic Email Update
#             if args.send_email:
#                 current_time = time.time()
#                 if (current_time - last_email_time) / 60 >= args.email_interval:
#                     send_run_update(args, results_log, epoch + 1, {
#                         'last_loss': avg_loss,
#                         'train_acc': avg_train_acc,
#                         'val_acc': avg_val_acc,
#                         'best_loss': best_loss,
#                         'last_reward': mean_reward,
#                         'best_reward': best_mean_reward
#                     }, task_name="NSFR IL Training")
#                     last_email_time = current_time
            

#     else:
#         print("No dataset found. There is no implementation for taking .csv as input")
#         quit()
#     # Print and save final learning curve
#     print("\n" + "="*50)
#     print("LEARNING CURVE")
#     print("="*50)
#     for res in results_log:
#         traj = res.get('trajectory', '-')
#         loss = res.get('train_loss', float('nan'))
#         t_acc = res.get('train_acc', 0.0)
#         v_acc = res.get('val_acc', 0.0)
#         print(f"  Epoch {res['epoch']:3d} | Traj {traj} | Loss {loss:.4f} | T-Acc {t_acc:.4f} | V-Acc {v_acc:.4f} | Score {res['mean_reward']:.2f} ± {res['std_reward']:.2f}")
#     print("="*50)

#     results_df = pd.DataFrame(results_log)
#     # Save CSV co-located with the models for this run
#     if args.use_gazemap:
#         run_dir = f"trained_models/{args.env}/grail/{num_iters}_ep"
#     else:
#         run_dir = f"trained_models/{args.env}/nsfr/{num_iters}_ep"
    
#     os.makedirs(run_dir, exist_ok=True)
#     results_csv_path = os.path.join(run_dir, f"results_lr_{args.lr}.csv")
#     results_df.to_csv(results_csv_path, index=False)
#     print(f"Results saved to {results_csv_path}")

#     # Final Email Update
#     if args.send_email:
#         # Get final values from the last entry in results_log
#         last_res = results_log[-1] if results_log else {}
#         send_run_update(args, results_log, last_res.get('epoch', 0), {
#             'train_loss': last_res.get('train_loss', 0.0),
#             'train_acc': last_res.get('train_acc', 0.0),
#             'val_acc': last_res.get('val_acc', 0.0),
#             'best_loss': best_loss,
#             'last_reward': last_res.get('mean_reward', 0.0),
#             'best_reward': best_mean_reward
#         }, is_final=True, task_name="NSFR IL Training")

# if __name__ == "__main__":
#     main()

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from core.utils.utils import get_primitive_action_map
from core.utils.utils import PtDataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from nsfr.agents.imitation_agent import ImitationAgent
from nsfr.utils import make_deterministic
from nsfr.env import NSFRBaseEnv
from core.utils.utils import (
    set_seed_everywhere, format_results_table,
    send_run_update, load_pt_dataset
)
from scripts.evaluation.evaluate_model import evaluate
import time


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
        loss, _, _ = agent.update(states, actions, gazes if args.use_gaze else None)
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
            loss, _, _ = agent.update(states, actions, gazes if args.use_gaze else None)
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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation fraction")
    parser.add_argument("--lr_patience", type=int, default=3, help="LR reduction patience")
    parser.add_argument("--gaze_threshold", type=float, default=50.0, help="Gaze valuation threshold")
    parser.add_argument("--use_gaze", action="store_true", help="Use gaze data for training")
    parser.add_argument("--use_gazemap", default=False, action="store_true", help="Use full gaze map")
    parser.add_argument("--gaze_model_path", type=str,
                        default="trained_models/gaze_predictor/seaquest_gaze_predictor_sigma_10.pth")
    parser.add_argument("--num_episodes", type=int, default=None, help="Episodes to load from .pt dataset")
    parser.add_argument("--sort_by", type=str, default=None, choices=['length', 'reward_per_step'])
    parser.add_argument("--valuation_path", type=str, default=None, help="Path to pre-computed valuation.pt")
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--eval_max_steps", type=int, default=10000, help="Max steps per eval episode")
    parser.add_argument("--send_email", action="store_true", help="Enable periodic email updates")
    parser.add_argument("--email_interval", type=int, default=30, help="Minutes between email updates")
    parser.add_argument("--diagnose", action="store_true", help="Profile bottlenecks and exit")
    args = parser.parse_args()

    # ── Gaze setup ───────────────────────────────────────────────────────────
    gaze_predictor = None
    if args.use_gazemap:
        args.use_gaze = True
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
    agent_gaze_threshold = args.gaze_threshold if args.use_gaze else None
    agent = ImitationAgent(args.env, args.rules, device, gaze_threshold=agent_gaze_threshold)

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
        use_gaze=args.use_gazemap,
        num_episodes=args.num_episodes,
        sort_by=args.sort_by,
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

    # ── Load pre-computed valuations ─────────────────────────────────────────
    valuations = None
    v_path = args.valuation_path
    if v_path is None:
        v_path = (f"trained_models/{args.env}/grail/valuation.pt"
                  if args.use_gazemap
                  else f"trained_models/{args.env}/nsfr/valuation.pt")

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
                v_list   = [torch.zeros(len(agent.model.atoms)).to(device)] * (max_step + 1)
                for s_idx, v in steps.items():
                    v_list[s_idx] = v
                valuations[ep_id] = v_list
            print(f"  Loaded valuations for {len(valuations)} episodes.")
        else:
            valuations = valuations_raw
    else:
        print(f"No pre-computed valuations at {v_path}. Training from logic states.")

    # ── Paths ─────────────────────────────────────────────────────────────────
    num_iters = args.num_episodes if args.num_episodes is not None else "full"
    run_dir   = (f"trained_models/{args.env}/grail/{num_iters}_ep"
                 if args.use_gazemap
                 else f"trained_models/{args.env}/nsfr/{num_iters}_ep")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("out/imitation", exist_ok=True)

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.lr_patience,
    )

    best_mean_reward = -float('inf')
    best_loss        = float('inf')
    patience         = 12
    patience_counter = 0
    results_log      = []
    last_email_time  = time.time()

    print("Starting epoch-based training over full dataset...")

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} (all {len(unique_eps)} episodes) ---")

        epoch_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
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
                        vT_list.append(torch.zeros(len(agent.model.atoms), device=device))
                vT_batch = torch.stack(vT_list).to(device)

            # Single forward pass — update() returns probs & action_scores
            loss, probs, action_scores = agent.update(
                states, actions,
                gazes if args.use_gaze else None,
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
                        gazes if args.use_gaze else None,
                        vT=vT_batch,
                    )

                    val_correct += (action_scores.argmax(dim=1) == actions).sum().item()
                    val_samples += B
                    log_scores = torch.log(action_scores.clamp(min=eps_nll))
                    val_loss    += nn.NLLLoss()(log_scores, actions).item()
                    val_n       += 1

            avg_val_acc = val_correct / max(val_samples, 1)
            print(f"Epoch {epoch+1} Val   Loss: {val_loss / max(val_n, 1):.4f} | Val Acc: {avg_val_acc:.4f}")

        scheduler.step(avg_val_acc if val_loader else -avg_loss)

        # ── Environment Evaluation ────────────────────────────────────────────
        if (epoch + 1) % args.eval_interval == 0:
            rewards = evaluate(
                agent, env,
                num_episodes=5, seed=args.seed,
                valuation_interval=0, log_interval=0,
                max_steps=args.eval_max_steps,
                gaze_predictor=(gaze_predictor if args.use_gazemap else None),
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
            'gaze':         args.use_gaze,
        })

        # ── Save checkpoint ───────────────────────────────────────────────────
        save_path = f"{run_dir}/epoch_{epoch+1}.pth"
        agent.save(save_path)
        print(f"Saved model to {save_path}")

        # ── Best model / early stopping ───────────────────────────────────────
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            patience_counter = 0
            best_model_path  = f"{run_dir}/best.pth"
            agent.save(best_model_path)
            print(f"--- New Best Model! Reward: {best_mean_reward:.2f} → {best_model_path} ---")

        if avg_loss < best_loss:
            best_loss        = avg_loss
            patience_counter = 0
            print(f"--- New Best Loss! {best_loss:.4f} ---")
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
    results_csv_path = os.path.join(run_dir, f"results_lr_{args.lr}.csv")
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