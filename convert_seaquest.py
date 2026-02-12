import os
import pandas as pd
import numpy as np
import json
import torch
from PIL import Image
from tqdm import tqdm
from ocatari.core import OCAtari
from nudge.env import NudgeBaseEnv

# Configuration
CSV_FILE = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/train.csv"
BASE_IMAGE_DIR = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/data/seaquest/gaze_data_tmp"
OUTPUT_FILE = "results/bs_data/seaquest_il_expert.json"
ENV_NAME = "seaquest"

def main():
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    # Sort by episode_id and frameid to ensure correct order
    df = df.sort_values(by=['episode_id', 'frameid'])
    
    # Initialize Environment and OCAtari
    print(f"Initializing {ENV_NAME} environment...")
    env = NudgeBaseEnv.from_name(ENV_NAME, mode='logic')
    game_name = ENV_NAME.capitalize()
    oc = OCAtari(game_name, mode="vision", render_mode="rgb_array")
    
    # Data containers
    actions = []
    logic_states = []
    neural_states = []
    action_probs = []
    logprobs = []
    rewards = []
    terminated = []
    predictions = [] # Placeholder
    
    # Pre-calculate termination
    # A frame is terminal if it's the last frame of an episode
    # We can check if the next row has a different episode_id
    df['next_episode_id'] = df['episode_id'].shift(-1)
    df['terminated'] = df['episode_id'] != df['next_episode_id']
    # The last row is always terminal
    df.loc[df.index[-1], 'terminated'] = True
    
    print("Processing frames...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Load Image
        traj_folder = row['trajectory']
        img_name = f"{row['frameid']}.png"
        img_path = os.path.join(BASE_IMAGE_DIR, traj_folder, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_array = np.array(image)
        except Exception as e:
            # print(f"Error loading image {img_path}: {e}")
            image_array = np.zeros((210, 160, 3), dtype=np.uint8)

        # Extract Objects using OCAtari (Mock ALE)
        if hasattr(oc, "_env") and hasattr(oc._env, "unwrapped") and hasattr(oc._env.unwrapped, "ale"):
            real_ale = oc._env.unwrapped.ale
            
            class MockALE:
                def __getattr__(self, name):
                    return getattr(real_ale, name)
                def getScreenRGB(self, *args):
                    return image_array
            
            oc._env.unwrapped.ale = MockALE()
            
            try:
                oc.detect_objects()
            except Exception as e:
                pass
            finally:
                oc._env.unwrapped.ale = real_ale
        else:
            try:
                oc.detect_objects(image_array)
            except Exception as e:
                pass
                
        objects = oc.objects
        
        # Convert to Logic State
        logic_state, neural_state = env.convert_state(objects)
        
        if isinstance(logic_state, torch.Tensor):
            logic_state = logic_state.tolist()
        if isinstance(neural_state, torch.Tensor):
            neural_state = neural_state.tolist()
        
        # Action
        action = int(row['action'])
        
        # Action Probs (One-hot)
        # Seaquest has 6 actions (0-5)
        probs = [0.0] * 6
        if 0 <= action < 6:
            probs[action] = 1.0
        
        # Log Probs (Log of one-hot... avoid log(0))
        # We'll use 0 for the taken action and a large negative number for others
        lprobs = [-1e10] * 6
        if 0 <= action < 6:
            lprobs[action] = 0.0
            
        # Append to lists
        actions.append(action)
        logic_states.append(logic_state)
        neural_states.append(neural_state)
        action_probs.append(probs)
        logprobs.append(lprobs)
        rewards.append(float(row['unclipped_reward']))
        terminated.append(int(row['terminated'])) # 0 or 1
        predictions.append(0) # Placeholder
        
    # Save to JSON
    data = {
        "actions": actions,
        "logic_states": logic_states,
        "neural_states": neural_states,
        "action_probs": action_probs,
        "logprobs": logprobs,
        "reward": rewards,
        "terminated": terminated,
        "predictions": predictions
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f)
    print("Done!")

if __name__ == "__main__":
    main()
