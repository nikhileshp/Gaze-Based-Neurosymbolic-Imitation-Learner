import os
import argparse
import torch
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

# Dataset classes moved to scripts/data_utils.py

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def add(self, states, actions, gazes, losses):
        """Add a batch of experiences with their associated losses as priorities."""
        if torch.is_tensor(states): states = states.cpu()
        if torch.is_tensor(actions): actions = actions.cpu()
        if torch.is_tensor(gazes): gazes = gazes.cpu()
        
        for i in range(len(states)):
            experience = (states[i], actions[i], gazes[i])
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
        
        return states, actions, gazes, indices

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
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val_split", type=float, default=0.00, help="Fraction of data to use as validation (0 = no validation)")
    parser.add_argument("--gaze_threshold", type=float, default=50.0, help="Threshold for gaze-based valuation scaling")
    parser.add_argument("--use_gaze", action="store_true", help="Use gaze data for training")
    parser.add_argument("--use_gazemap", default=False, action="store_true", help="Use full gaze map for valuation")
    parser.add_argument("--gaze_model_path", type=str, default="seaquest_gaze_predictor_2.pth", help="Path to the .pth gaze predictor weights")
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to load from .pt dataset")
    parser.add_argument("--sort_by", type=str, default=None, choices=['length', 'reward_per_step'], help="How to sort episodes before selection")
    args = parser.parse_args()

    if args.use_gazemap:
        args.use_gaze = True
        from scripts.gaze_predictor import Human_Gaze_Predictor
        print(f"Initializing Gaze Predictor from {args.gaze_model_path}...")
        gaze_predictor = Human_Gaze_Predictor(args.env)
        gaze_predictor.init_model(args.gaze_model_path)
        gaze_predictor.model.eval()
    # Prioritize use_gazemap over use_gaze if both set? Or allow both?
    # Agent expects `use_gaze` for logic. Let's set args.use_gaze = True if use_gazemap is True
    if args.use_gazemap:
        args.use_gaze = True


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
    agent = ImitationAgent(args.env, args.rules, device, lr=args.lr, gaze_threshold=agent_gaze_threshold)

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

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        ) if val_dataset else None

        # ── Epoch-based training loop ─────────────────────────────────────────
        print("Starting epoch-based training over full dataset...")
        results_log = []
        os.makedirs(f"models/nsfr/{args.env}", exist_ok=True)
        os.makedirs("out/imitation", exist_ok=True)
        gaze_str = "_with_gazemap_values" if args.use_gazemap else (f"_with_gaze_{args.gaze_threshold}" if args.use_gaze else "_no_gaze")
        
        # Metadata for saving
        num_iters = args.num_episodes if args.num_episodes is not None else "full"
        experiment_str = f"{args.env}_{args.rules}_il_lr_{args.lr}_num_ep_{num_iters}"

        # Initialize PER, Early Stopping and Best Model tracking
        replay_buffer = PrioritizedReplayBuffer(capacity=50000)
        best_mean_reward = -float('inf')
        patience = 5
        patience_counter = 0
        replay_steps = 10 

        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
            agent.model.train()

            total_loss, n_batches = 0.0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for states, actions, gazes in pbar:
                states  = states.to(device)
                actions = actions.to(device)
                gazes   = gazes.to(device)

                if args.use_gaze:
                    probs = agent.model(states, gazes)
                else:
                    probs = agent.model(states, None)

                B = probs.size(0)
                action_probs = torch.zeros(B, 6, device=device)
                for i, pred in enumerate(agent.model.get_prednames()):
                    prefix = pred.split('_')[0]
                    if prefix in PRIMITIVE_ACTION_MAP:
                        action_probs[:, PRIMITIVE_ACTION_MAP[prefix]] += probs[:, i]

                log_probs = torch.log(action_probs + 1e-10)
                loss = agent.loss_fn(log_probs, actions)

                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

                total_loss += loss.item()
                n_batches  += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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
                    for states, actions, gazes in val_loader:
                        states, actions, gazes = states.to(device), actions.to(device), gazes.to(device)
                        probs = agent.model(states, gazes if args.use_gaze else None)
                        B = probs.size(0)
                        act_p = torch.zeros(B, 6, device=device)
                        for i, pred in enumerate(agent.model.get_prednames()):
                            prefix = pred.split('_')[0]
                            if prefix in PRIMITIVE_ACTION_MAP:
                                act_p[:, PRIMITIVE_ACTION_MAP[prefix]] += probs[:, i]
                        log_p = torch.log(act_p + 1e-10)
                        val_loss += agent.loss_fn(log_p, actions).item()
                        val_n += 1
                print(f"Epoch {epoch+1} Val Loss:   {val_loss / max(val_n, 1):.4f}")
            # Evaluation in environment
            rewards = evaluate(agent, env, num_episodes=5, seed=args.seed, valuation_interval=100, log_interval=100, gaze_predictor=(gaze_predictor if args.use_gazemap else None))
            mean_reward, std_reward = np.mean(rewards), np.std(rewards)
            print(f"Epoch {epoch+1} Eval Score: Mean={mean_reward:.2f}  Std={std_reward:.2f}")

            results_log.append({
                'epoch': epoch + 1, 'trajectory': 'all',
                'mean_reward': mean_reward, 'std_reward': std_reward,
                'train_loss': avg_loss, 'gaze': args.use_gaze,
            })

            os.makedirs(f"models/nsfr/{args.env}/{gaze_str}/{experiment_str}", exist_ok=True)
            save_path = f"models/nsfr/{args.env}/{gaze_str}/{experiment_str}/epoch_{epoch+1}.pth"
            agent.save(save_path)

            # Best Model and Early Stopping
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                patience_counter = 0
                best_model_path = f"models/nsfr/{args.env}/{gaze_str}/{experiment_str}/best.pth"
                agent.save(best_model_path)
                print(f"--- New Best Model! Reward: {best_mean_reward:.2f}. Saved to {best_model_path} ---")
            else:
                patience_counter += 1
                print(f"--- No improvement. Patience: {patience_counter}/{patience} ---")
                if patience_counter >= patience:
                    print(f"--- Early stopping triggered after {epoch+1} epochs ---")
                    break

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
        replay_buffer = PrioritizedReplayBuffer(capacity=20000)
        best_mean_reward = -float('inf')
        patience = 5
        patience_counter = 0
        replay_steps = 5 # Number of replay batches per trajectory epoch
        
        # Use args.epochs as the number of trajectories to process if it's less than total trajectories
        num_iters = min(args.epochs, len(trajectories))
        gaze_str = "_with_gazemap_values" if args.use_gazemap else (f"_with_gaze_{args.gaze_threshold}" if args.use_gaze else "_no_gaze")
        experiment_str = f"{args.env}_{args.rules}_il_lr_{args.lr}_num_ep_{num_iters}"
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

                if args.use_gaze:
                    probs = agent.model(states, gazes)
                else:
                    probs = agent.model(states, None)

                B = probs.size(0)
                action_probs = torch.zeros(B, 6, device=device)
                prednames = agent.model.get_prednames()
                for i, pred in enumerate(prednames):
                    prefix = pred.split('_')[0]
                    if prefix in PRIMITIVE_ACTION_MAP:
                        action_probs[:, PRIMITIVE_ACTION_MAP[prefix]] += probs[:, i]

                log_probs = torch.log(action_probs + 1e-10)
                loss = agent.loss_fn(log_probs, actions)
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()
                
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
                replay_buffer.add(torch.cat(all_states), torch.cat(all_actions), torch.cat(all_gazes), torch.cat(all_losses))

            # 2. Perform Replay Training
            if len(replay_buffer) >= args.batch_size:
                print(f"Performing {replay_steps} replay steps from PER buffer...")
                agent.model.train()
                for _ in range(replay_steps):
                    sample = replay_buffer.sample(args.batch_size)
                    if not sample: break
                    s_b, a_b, g_b, indices = sample
                    s_b, a_b, g_b = s_b.to(device), a_b.to(device), g_b.to(device)
                    p_b = agent.model(s_b, g_b if args.use_gaze else None)
                    act_p_b = torch.zeros(s_b.size(0), 6, device=device)
                    for i, pred in enumerate(agent.model.get_prednames()):
                        prefix = pred.split('_')[0]
                        if prefix in PRIMITIVE_ACTION_MAP:
                            act_p_b[:, PRIMITIVE_ACTION_MAP[prefix]] += p_b[:, i]
                    log_p_b = torch.log(act_p_b + 1e-10)
                    l_b_ind = torch.nn.functional.nll_loss(log_p_b, a_b, reduction='none')
                    agent.optimizer.zero_grad()
                    l_b_ind.mean().backward()
                    agent.optimizer.step()
                    replay_buffer.update_priorities(indices, l_b_ind.detach().cpu().numpy())

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

            # Evaluation
            if args.use_gazemap:
                rewards = evaluate(agent, env, num_episodes=5, seed=args.seed, gaze_predictor=gaze_predictor)
            else:
                rewards = evaluate(agent, env, num_episodes=5, seed=args.seed, gaze_predictor=None)
            mean_reward, std_reward = np.mean(rewards), np.std(rewards)
            print(f"Epoch {epoch+1} Evaluation Score: Mean={mean_reward:.2f}, Std={std_reward:.2f}")

            results_log.append({'epoch': epoch+1, 'trajectory': traj_num, 'mean_reward': mean_reward, 'std_reward': std_reward, 'train_loss': avg_loss, 'gaze': args.use_gaze})

            # Save per-epoch model
            os.makedirs(f"models/nsfr/{args.env}/{gaze_str}/{experiment_str}", exist_ok=True)
            save_path = f"models/nsfr/{args.env}/{gaze_str}/{experiment_str}/epoch_{epoch+1}.pth"
            agent.save(save_path)
            
            # Check for Best Model and Early Stopping
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                patience_counter = 0
                best_model_path = f"out/imitation/best_{args.env}{gaze_str}.pth"
                agent.save(best_model_path)
                print(f"--- New Best Model! Reward: {best_mean_reward:.2f}. Saved to {best_model_path} ---")
            else:
                patience_counter += 1
                print(f"--- No improvement. Patience: {patience_counter}/{patience} ---")
                if patience_counter >= patience:
                    print(f"--- Early stopping triggered after {epoch+1} epochs ---")
                    break

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
    results_csv_path = os.path.join("out/imitation", f"{args.env}_{args.rules}_lr_{args.lr}_results.csv")
    if os.path.exists(results_csv_path):
        results_df.to_csv(results_csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

if __name__ == "__main__":
    main()
