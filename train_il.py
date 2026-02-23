import os
import argparse
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from ocatari.core import OCAtari
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from nudge.agents.imitation_agent import ImitationAgent
from nudge.env import NudgeBaseEnv
from nudge.utils import make_deterministic
from tqdm import tqdm
from collections import Counter
from evaluate_model import evaluate
# Configuration from train_per_action.py
CSV_FILE = "data/seaquest/train.csv"
BASE_IMAGE_DIR = "data/seaquest/trajectories"

# Mapping from Predicate Name to ALE Action Index
# 0: noop, 1: fire, 2: up, 3: right, 4: left, 5: down
PRIMITIVE_ACTION_MAP = {
    'noop': 0,
    'fire': 1,
    'up': 2,
    'right': 3,
    'left': 4,
    'down': 5
}


 

class ExpertDataset(Dataset):
    def __init__(self, env_name, agent_prednames, data_path=None, nudge_env=None, limit=None, use_gazemap=False, trajectory=None):

        self.env_name = env_name
        self.agent_prednames = agent_prednames
        self.nudge_env = nudge_env
        self.data = None
        self.df = None
        self.use_gazemap = False
        self.gaze_masks = None
        
        # Check if use_gazemap is passed via kwargs or modify init signature
        # Since I can't easily change init signature without updating main call, I'll rely on a new argument 'use_gazemap' 
        # But wait, I AM updating main call. So I can change signature.

        
        # Initialize OCAtari for object detection
        # We use the game name from env_name (e.g. seaquest -> Seaquest)
        game_name = env_name.capitalize()
        print(f"Initializing OCAtari for {game_name}...")
        self.oc = OCAtari(game_name, mode="vision", render_mode="rgb_array")
        
        if data_path and os.path.exists(data_path):
            if data_path.endswith('.pkl'):
                print(f"Loading pre-computed data from {data_path}...")
                self.precomputed_data = torch.load(data_path)
                self.data = self.precomputed_data['data']
                self.atom_names = self.precomputed_data['atom_names']
                self.df = None
                print(f"Loaded {len(self.data)} samples with {len(self.atom_names)} atoms.")
            else:
                print(f"Loading data from {data_path}...")
                self.df = pd.read_csv(data_path)
        else:
            print(f"Data path {data_path} not found. Using default CSV: {CSV_FILE}")
            if os.path.exists(CSV_FILE):
                self.df = pd.read_csv(CSV_FILE)
            else:
                print("Default CSV not found. Using dummy data.")
                self.df = None

        self.use_gazemap = use_gazemap
        if self.use_gazemap:
            mask_path = 'data/seaquest/gaze_masks.pt'
            if os.path.exists(mask_path):
                print(f"Loading gaze masks from {mask_path}...")
                self.gaze_masks = torch.load(mask_path)
                print(f"Loaded gaze masks tensor: {self.gaze_masks.shape}")
            else:
                print(f"Warning: Gaze masks not found at {mask_path}. Falling back to default.")
                self.use_gazemap = False

        if self.df is not None and limit:
            print(f"Limiting dataset to {limit} samples.")
            self.df = self.df.head(limit)

        # Filter out NOOP (0) and actions not in our map target values
        if self.df is not None:
             initial_len = len(self.df)
             supported_actions = set(PRIMITIVE_ACTION_MAP.values())
             self.df = self.df[self.df['action'].isin(supported_actions)]
             
             # Filter by trajectory if specified
             if trajectory is not None:
                 print(f"Filtering dataset for trajectory {trajectory}...")
                 self.df = self.df[self.df['trajectory_number'] == trajectory]
             
             print(f"Filtered to supported actions/trajectory: {len(self.df)} samples")
        
        # Filter pre-computed data
        if self.data is not None and isinstance(self.data, list) and len(self.data) > 0 and 'action' in self.data[0]:
             
             # Add original index to each item before filtering to maintain alignment with gaze_masks
             if self.use_gazemap and self.gaze_masks is not None:
                 for i, item in enumerate(self.data):
                     if isinstance(item, dict):
                        item['original_index'] = i
             
             # Filter supported actions
             supported_actions = set(PRIMITIVE_ACTION_MAP.values())
             initial_len = len(self.data)
             self.data = [d for d in self.data if d.get('action') in supported_actions]
             print(f"Filtered pre-computed data to supported actions: {initial_len} -> {len(self.data)} samples")
             
             # Filter by trajectory
             if trajectory is not None:
                 print(f"Filtering pre-computed data for trajectory {trajectory}...")
                 self.data = [d for d in self.data if d.get('trajectory_number') == trajectory]

             if limit:
                 self.data = self.data[:limit]

        # Action mapping logic omitted as in original
        
        if self.df is None and self.data is None:
            # Dummy data logic
            obs = nudge_env.reset()
            logic_state = obs[0]
            logic_state_tensor = torch.tensor(logic_state, dtype=torch.float32)
            self.state_shape = logic_state_tensor.shape
            self.data = []
            for _ in range(100):
                state = torch.rand(self.state_shape)
                action = torch.randint(0, len(agent_prednames), (1,)).item()
                gaze = torch.rand(2) * 160 
                self.data.append((state, action, gaze))
        elif self.df is None and self.data is not None:
            pass # Data already loaded
        else:
            self.data = None # Use df

    def __len__(self):
        if self.df is not None:
            return len(self.df)
        return len(self.data)

    def __getitem__(self, idx):
        if self.df is None:
            item = self.data[idx]
            # Check if it's new format (dict) or old dummy format (tuple)
            if isinstance(item, dict):
                # Pre-computed format
                atoms = torch.tensor(item['atoms'], dtype=torch.float32)
                action = torch.tensor(item['action'], dtype=torch.long)
                # Gaze?
                gaze_val = item.get('gaze', [0.0, 0.0])
                gaze = torch.tensor(gaze_val, dtype=torch.float32)
                
                # If use_gazemap, try to get mask using original_index
                if self.use_gazemap and self.gaze_masks is not None:
                    original_idx = item.get('original_index', -1)
                    if original_idx >= 0 and original_idx < len(self.gaze_masks):
                        return atoms, action, self.gaze_masks[original_idx]
                
                return atoms, action, gaze
            else:
                state, action, gaze = item
                return torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.long), gaze
        
        row = self.df.iloc[idx]
        # Mapping from frame_id to traj_folder
        traj_folder_map = {}
        for folder in os.listdir(BASE_IMAGE_DIR):
            folder_path = os.path.join(BASE_IMAGE_DIR, folder)
            if not os.path.isdir(folder_path):
                continue
            parts = folder.split('_')
            if len(parts) < 3:
                continue
            traj_folder_map[parts[1] + "_" + parts[2]] = folder

        traj_folder = traj_folder_map[row['frame_id'].split('_')[0]+"_"+row['frame_id'].split('_')[1]]
        img_name = f"{row['frame_id'].split('_')[2]}.png"
        img_name = row['frame_id'].split('_')[0]+"_"+row['frame_id'].split('_')[1]+'_'+img_name
        traj_folder = traj_folder.replace('.txt', '')
        img_path = os.path.join(BASE_IMAGE_DIR, traj_folder, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_array = np.array(image)
        except Exception as e:
            image_array = np.zeros((210, 160, 3), dtype=np.uint8)
            print(f"Error loading image {img_path}: {e}")
            exit()
            
        if hasattr(self.oc, "_env") and hasattr(self.oc._env, "unwrapped") and hasattr(self.oc._env.unwrapped, "ale"):
            real_ale = self.oc._env.unwrapped.ale
            
            class MockALE:
                def __getattr__(self, name):
                    return getattr(real_ale, name)
                    
                def getScreenRGB(self, *args):
                    return image_array
            
            self.oc._env.unwrapped.ale = MockALE()
            
            try:
                self.oc.detect_objects()
            except ValueError as e:
                if "exceeds the maximum number of objects" in str(e):
                    pass
                else:
                    print(f"Warning: OCAtari detection failed: {e}")
                pass
            finally:
                self.oc._env.unwrapped.ale = real_ale
        else:
            try:
                self.oc.detect_objects(image_array)
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not mock ALE and detect_objects failed: {e}")
                
        objects = self.oc.objects
        logic_state, _ = self.nudge_env.convert_state(objects)
        action_idx = int(row['action'])
        
        gaze_center = torch.zeros(2, dtype=torch.float32)
        if 'gaze_positions' in row and pd.notna(row['gaze_positions']):
            try:
                gaze_vals = [float(x) for x in str(row['gaze_positions']).split(',')]
                if len(gaze_vals) > 0 and len(gaze_vals) % 2 == 0:
                    gaze_vals = np.array(gaze_vals).reshape(-1, 2)
                    mean_gaze = np.mean(gaze_vals, axis=0)
                    gaze_center = torch.tensor(mean_gaze, dtype=torch.float32)
            except ValueError:
                pass 

        if self.use_gazemap and self.gaze_masks is not None:
            # Use original index from dataframe to access tensor
            original_idx = row.name 
            if original_idx < len(self.gaze_masks):
                # Return mask (84, 84). The model forward expects it as 'gaze' arg.
                return torch.tensor(logic_state, dtype=torch.float32), torch.tensor(action_idx, dtype=torch.long), self.gaze_masks[original_idx]
            else:
                # Should not happen if indices are aligned
                return torch.tensor(logic_state, dtype=torch.float32), torch.tensor(action_idx, dtype=torch.long), torch.zeros((84, 84))
        
        if self.use_gazemap and self.gaze_masks is not None:
             # Use original index from dataframe to access tensor
             original_idx = row.name 
             if original_idx < len(self.gaze_masks):
                 # Return mask (84, 84). The model forward expects it as 'gaze' arg.
                 return torch.tensor(logic_state, dtype=torch.float32), torch.tensor(action_idx, dtype=torch.long), self.gaze_masks[original_idx]
             else:
                 # Should not happen if indices are aligned
                 return torch.tensor(logic_state, dtype=torch.float32), torch.tensor(action_idx, dtype=torch.long), torch.zeros((84, 84))
                 
        return torch.tensor(logic_state, dtype=torch.float32), torch.tensor(action_idx, dtype=torch.long), gaze_center


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
    parser.add_argument("--data_path", type=str, default=None, help="Path to expert data")
    parser.add_argument("--epochs", type=int, default=16, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--gaze_threshold", type=float, default=50.0, help="Threshold for gaze-based valuation scaling")
    parser.add_argument("--use_gaze", action="store_true", help="Use gaze data for training")
    parser.add_argument("--use_gazemap", action="store_true", help="Use full gaze map for valuation")
    parser.add_argument("--gaze_model_path", type=str, default="seaquest_gaze_predictor_2.pth", help="Path to the .pth gaze predictor weights")
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
    if os.path.exists(data_path):
        full_df = pd.read_csv(data_path)
        if 'trajectory_number' in full_df.columns:
            trajectories = sorted(full_df['trajectory_number'].unique())
            print(f"Found {len(trajectories)} trajectories: {trajectories}")
        else:
            print("Warning: 'trajectory_number' column not found in CSV. Using single trajectory [1].")
            trajectories = [1]
    else:
        print(f"Warning: Data path {data_path} not found for trajectory analysis. Using default [1].")
        trajectories = [1]

    # Training Loop
    print("Starting iterative training by trajectory...")
    results_log = []
    
    # Use args.epochs as the number of trajectories to process if it's less than total trajectories
    num_iters = min(args.epochs, len(trajectories))
    
    for epoch in range(num_iters):
        traj_num = trajectories[epoch]
        print(f"\n--- Epoch {epoch+1}/{num_iters} (Trajectory {traj_num}) ---")
        
        # Load Data for this specific trajectory
        dataset = ExpertDataset(args.env, agent.model.prednames, args.data_path, nudge_env=env, limit=args.limit, use_gazemap=args.use_gazemap, trajectory=traj_num)
        
        if len(dataset) == 0:
            print(f"Skipping empty trajectory {traj_num}")
            continue

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Training Traj {traj_num}")
        for states, actions, gazes in pbar:
            states = states.to(device)
            actions = actions.to(device)
            gazes = gazes.to(device)
            
            # Forward pass
            if args.use_gaze:
                probs = agent.model(states, gazes)
            else:
                probs = agent.model(states, None)
            
            # Aggregate probabilities for each action
            batch_size = probs.size(0)
            num_actions = 6
            action_probs = torch.zeros(batch_size, num_actions, device=device)
            
            prednames = agent.model.get_prednames()
            for i, pred in enumerate(prednames):
                prefix = pred.split('_')[0]
                if prefix in PRIMITIVE_ACTION_MAP:
                    act_idx = PRIMITIVE_ACTION_MAP[prefix]
                    action_probs[:, act_idx] += probs[:, i]
            
            log_probs = torch.log(action_probs + 1e-10)
            loss = agent.loss_fn(log_probs, actions)
            
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            
            loss_val = loss.item()
            total_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Evaluation
        if args.use_gazemap:
            rewards = evaluate(agent, env, num_episodes=5, seed=args.seed, gaze_predictor=gaze_predictor)
        else:
            rewards = evaluate(agent, env, num_episodes=5, seed=args.seed, gaze_predictor=None)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"Epoch {epoch+1} Evaluation Score: Mean={mean_reward:.2f}, Std={std_reward:.2f}")
        
        results_log.append({
            'epoch': epoch + 1,
            'trajectory': traj_num,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'gaze': args.use_gaze,    
        })

        # Save Model
        os.makedirs("out/imitation", exist_ok=True)
        gaze_str = f"_with_gaze_{args.gaze_threshold}" if args.use_gaze else "_no_gaze"
        gaze_str = f"_with_gazemap_values" if args.use_gazemap else gaze_str
        save_path = f"out/imitation/new_{args.env}_{args.rules}_il_epoch_{epoch+1}_lr_{args.lr}{gaze_str}.pth"
        agent.save(save_path)
        print(f"Model saved to {save_path}")

    # Print and Save final learning curve log
    print("\n" + "="*30)
    print("LEARNING CURVE LOG")
    print("="*30)
    print("Epoch | Trajectory | Mean Score | Std Dev")
    for res in results_log:
        print(f"{res['epoch']:5d} | {res['trajectory']:10d} | {res['mean_reward']:10.2f} | {res['std_reward']:7.2f}")
    print("="*30)

    # Save results to CSV

    results_df = pd.DataFrame(results_log)
    results_csv_path = os.path.join("out/imitation", f"{args.env}_{args.rules}_lr_{args.lr}_results.csv")
    #If results_csv_path exists, append to file
    if os.path.exists(results_csv_path):
        results_df.to_csv(results_csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

if __name__ == "__main__":
    main()
