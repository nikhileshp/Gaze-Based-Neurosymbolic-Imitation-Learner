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

# Configuration from train_per_action.py
CSV_FILE = "data/seaquest/train.csv"
BASE_IMAGE_DIR = "data/seaquest/trajectories"

# Mapping from Predicate Name to ALE Action Index
# 0: noop, 1: fire, 2: up, 3: right, 4: left, 5: down
PREDICATE_TO_ACTION_MAP = {
    'up_air': 2,
    'fire_left': 1,
    'fire_right': 1,
    'left_aim': 4,
    'right_aim': 3,
    'down_aim': 5,
    'up_aim': 2,
    'up_evade': 2,
    'down_evade': 5,
    'left_to_diver': 4,
    'right_to_diver': 3,
    'up_to_diver': 2,
    'down_to_diver': 5,
    'noop': 0
}

class ExpertDataset(Dataset):
    def __init__(self, env_name, agent_prednames, data_path=None, nudge_env=None, limit=None):
        self.env_name = env_name
        self.agent_prednames = agent_prednames
        self.nudge_env = nudge_env
        
        # Initialize OCAtari for object detection
        # We use the game name from env_name (e.g. seaquest -> Seaquest)
        game_name = env_name.capitalize()
        print(f"Initializing OCAtari for {game_name}...")
        self.oc = OCAtari(game_name, mode="vision", render_mode="rgb_array")
        
        if data_path and os.path.exists(data_path):
            print(f"Loading data from {data_path}...")
            self.df = pd.read_csv(data_path)
        else:
            print(f"Data path {data_path} not found. Using default CSV: {CSV_FILE}")
            if os.path.exists(CSV_FILE):
                self.df = pd.read_csv(CSV_FILE)
            else:
                print("Default CSV not found. Using dummy data.")
                self.df = None

        if self.df is not None and limit:
            print(f"Limiting dataset to {limit} samples.")
            self.df = self.df.head(limit)

        # Filter out NOOP (0) and actions not in our map target values
        # Our map targets are {0, 1, 2, 3, 4, 5}
        # We also need to filter out actions that don't have ANY predicate mapping to them?
        # The map covers 0, 1, 2, 3, 4, 5. So we don't need to filter out 0 anymore.
        if self.df is not None:
             initial_len = len(self.df)
             # Filter rows where action is not 0
             # self.df = self.df[self.df['action'] != 0]
             # print(f"Filtered out NOOPs: {initial_len} -> {len(self.df)} samples")
             
             # Also ensure action is in our supported set {0, 1, 2, 3, 4, 5}
             supported_actions = set(PREDICATE_TO_ACTION_MAP.values())
             self.df = self.df[self.df['action'].isin(supported_actions)]
             print(f"Filtered to supported actions: {len(self.df)} samples")

        # Action mapping (Action Index -> Predicate Name)
        # We need to map the dataset action (e.g. 0-5) to the predicate index in agent
        # NudgeEnv.pred2action maps Predicate Name -> Action Index
        # We need Action Index -> Predicate Name
        # self.action2pred = {v: k for k, v in nudge_env.pred2action.items()}
        
        if self.df is None:
            # Dummy data logic
            obs = nudge_env.reset()
            logic_state = obs[0]
            logic_state_tensor = torch.tensor(logic_state, dtype=torch.float32)
            self.state_shape = logic_state_tensor.shape
            self.data = []
            for _ in range(100):
                state = torch.rand(self.state_shape)
                action = torch.randint(0, len(agent_prednames), (1,)).item()
                gaze = torch.rand(2) * 160 # dummy gaze in image range
                self.data.append((state, action, gaze))
        else:
            self.data = None # Use df

    def __len__(self):
        if self.df is not None:
            return len(self.df)
        return len(self.data)

    def __getitem__(self, idx):
        if self.df is None:
            state, action, gaze = self.data[idx]
            return torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.long), gaze
        
        row = self.df.iloc[idx]
        # Mapping from frame_id to traj_folder
        #Map Folders to trajectory_folder_parts
        traj_folder_map = {}
        for folder in os.listdir(BASE_IMAGE_DIR):
            # Skip non-directories (like .DS_Store) and ensure proper format
            folder_path = os.path.join(BASE_IMAGE_DIR, folder)
            if not os.path.isdir(folder_path):
                continue
            
            # Split folder name and verify it has at least 3 parts
            parts = folder.split('_')
            if len(parts) < 3:
                continue
            
            # Create mapping: "part1_part2" -> full_folder_name
            traj_folder_map[parts[1] + "_" + parts[2]] = folder

        
        #Dictionary to map traj_folder_part to traj_folder
        traj_folder = traj_folder_map[row['frame_id'].split('_')[0]+"_"+row['frame_id'].split('_')[1]]
        img_name = f"{row['frame_id'].split('_')[2]}.png"
        img_name = row['frame_id'].split('_')[0]+"_"+row['frame_id'].split('_')[1]+'_'+img_name
        traj_folder = traj_folder.replace('.txt', '')
        img_path = os.path.join(BASE_IMAGE_DIR, traj_folder, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_array = np.array(image)
        except Exception as e:
            # Fallback for missing images
            # print(f"Error loading image {img_path}: {e}")
            image_array = np.zeros((210, 160, 3), dtype=np.uint8)
            print(f"Error loading image {img_path}: {e}")
            exit()
        # Extract Objects using OCAtari
        # OCAtari detects objects and updates internal state
        # We need to mock the ALE getScreenRGB method to return our image
        # because detect_objects() reads from the environment state.
        
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
                # print(f"Warning: OCAtari detection failed: {e}")
                # Check for excessive object count warning
                if "exceeds the maximum number of objects" in str(e):
                    # print(f"WARNING: FRAME {row['frameid']} excessive missiles detected")
                    pass
                else:
                    print(f"Warning: OCAtari detection failed: {e}")
                # Fallback to empty objects or previous objects?
                # For now, we'll let it proceed with whatever objects are there or empty
                pass
            finally:
                # Restore original ALE
                self.oc._env.unwrapped.ale = real_ale
        else:
            # Fallback
            try:
                self.oc.detect_objects(image_array)
            except (TypeError, ValueError) as e:
                print(f"Warning: Could not mock ALE and detect_objects failed: {e}")
                
        objects = self.oc.objects
        
        # Convert to Logic State using NudgeEnv
        # NudgeEnv.convert_state expects list of objects
        # It returns (logic_state, neural_state)
        logic_state, _ = self.nudge_env.convert_state(objects)
        
        # Get Action
        # Assuming 'action' column contains the action index
        action_idx = int(row['action'])
        
        # Extract Gaze Data
        gaze_center = torch.zeros(2, dtype=torch.float32)
        if 'gaze_positions' in row and pd.notna(row['gaze_positions']):
            try:
                # gaze_positions is a string of comma-separated floats "x1,y1,x2,y2,..."
                gaze_vals = [float(x) for x in str(row['gaze_positions']).split(',')]
                if len(gaze_vals) > 0 and len(gaze_vals) % 2 == 0:
                    gaze_vals = np.array(gaze_vals).reshape(-1, 2)
                    mean_gaze = np.mean(gaze_vals, axis=0)
                    gaze_center = torch.tensor(mean_gaze, dtype=torch.float32)
            except ValueError:
                pass # Use default (0,0) if parsing fails

        # We return the raw action index (e.g. 2 for UP)
        # The loss function will handle mapping from predicates to this index
            
        return torch.tensor(logic_state, dtype=torch.float32), torch.tensor(action_idx, dtype=torch.long), gaze_center


def evaluate(agent, env, num_episodes=5):
    agent.model.eval()
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Extract logic state
            # state is (logic_state, neural_state)
            logic_state, _ = state
            logic_state_tensor = torch.tensor(logic_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            
            action = agent.act(logic_state_tensor)
            
            # Map action index to predicate name if needed?
            # LogicAgent.select_action returns predicate name.
            # NudgeBaseEnv.step expects predicate name?
            # Let's check LogicPPO.select_action:
            # predicate = self.prednames[action.item()]
            # return predicate
            
            # So we need prednames
            prednames = agent.model.get_prednames()
            predicate = prednames[action]
            
            state, reward, done = env.step(predicate)
            episode_reward += reward
        total_reward += episode_reward
    agent.model.train()
    return total_reward / num_episodes

def get_balanced_sampler(dataset):
    """
    Creates a WeightedRandomSampler to balance the action distribution in the batches.
    """
    print("Computing class weights for balanced sampling...")
    
    # Extract all targets (actions)
    if dataset.df is not None:
        targets = dataset.df['action'].values
    else:
        # Dummy data is list of (state, action_tensor)
        targets = [item[1].item() for item in dataset.data]
        
    targets = np.array(targets)
    class_counts = Counter(targets)
    
    print(f"Class counts: {class_counts}")
    
    # Compute weight for each class: 1.0 / count
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    
    # Assign weight to each sample
    samples_weights = np.array([class_weights[t] for t in targets])
    samples_weights = torch.from_numpy(samples_weights).double()
    
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    return sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="getout", help="Environment name")
    parser.add_argument("--rules", type=str, default="getout_human", help="Ruleset name")
    parser.add_argument("--data_path", type=str, default=None, help="Path to expert data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--sampler", type=str, default="balanced", choices=["none", "balanced"], help="Sampler to use (none/balanced)")
    parser.add_argument("--gaze_threshold", type=float, default=50.0, help="Threshold for gaze-based valuation scaling")
    parser.add_argument("--use_gaze", action="store_true", help="Use gaze data for training")
    args = parser.parse_args()

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

    # Load Data
    # Pass env_name, prednames, and nudge_env to Dataset
    dataset = ExpertDataset(args.env, agent.model.prednames, args.data_path, nudge_env=env, limit=args.limit)

    # Create Sampler
    sampler = None
    shuffle = True
    if args.sampler == "balanced":
        sampler = get_balanced_sampler(dataset)
        shuffle = False # Shuffle must be False when using a sampler

    # Use num_workers=0 because OCAtari might not be fork-safe or thread-safe in Dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=shuffle, num_workers=0)

    # Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for states, actions, gazes in pbar:
            states = states.to(device)
            actions = actions.to(device)
            gazes = gazes.to(device)
            
            # Custom update logic to handle predicate aggregation
            # agent.update expects standard NLL loss. We need to override or modify it.
            # Ideally we modify ImitationAgent, but let's do it here for now to avoid breaking other things.
            
            # Forward pass
            # probs: (batch, num_preds)
            if args.use_gaze:
                probs = agent.model(states, gazes)
            else:
                probs = agent.model(states, None)
            
            # Aggregate probabilities for each action
            # We need a tensor of shape (batch, num_actions) where num_actions = 6 (0-5)
            # Initialize with zeros
            batch_size = probs.size(0)
            num_actions = 6
            action_probs = torch.zeros(batch_size, num_actions, device=device)
            
            # Get prednames from agent
            prednames = agent.model.get_prednames()
            
            # Sum probs for each predicate mapping to an action
            for i, pred in enumerate(prednames):
                if pred in PREDICATE_TO_ACTION_MAP:
                    act_idx = PREDICATE_TO_ACTION_MAP[pred]
                    action_probs[:, act_idx] += probs[:, i]
            
            # Normalize? Probs should sum to <= 1 (since we filtered NOOP, sum might be < 1 if NOOP was a predicate)
            # But here we are summing disjoint sets of predicates.
            # If all predicates map to SOME action, sum should be 1.
            # If some predicates map to NOOP (which we filtered), sum < 1.
            # We should re-normalize to avoid log(0) for valid actions?
            # Or just use the sum.
            
            # Compute NLL Loss
            # Add epsilon
            log_probs = torch.log(action_probs + 1e-10)
            loss = agent.loss_fn(log_probs, actions)
            
            # Backward pass
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            
            loss_val = loss.item()
            total_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % 5 == 0:
            avg_reward = evaluate(agent, env)
            print(f"Evaluation Reward: {avg_reward:.2f}")

        # Save Model
        os.makedirs("out/imitation", exist_ok=True)
        gaze_str = f"_with_gaze_{args.gaze_threshold}" if args.use_gaze else "_no_gaze"
        save_path = f"out/imitation/{args.env}_{args.rules}_il{gaze_str}.pth"
        agent.save(save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
