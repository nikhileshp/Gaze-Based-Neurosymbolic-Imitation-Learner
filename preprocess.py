import os
import glob
import pandas as pd
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# Import necessary classes for atom generation from train_il
# Ensure PRIMITIVE_ACTION_MAP is imported to maintain consistency
from train_il import ExpertDataset, PRIMITIVE_ACTION_MAP
from nudge.agents.imitation_agent import ImitationAgent
from nudge.env import NudgeBaseEnv

def parse_trajectory_file(txt_file):
    """
    Manually parse the trajectory .txt file.
    Handles variable number of commas in the gaze_positions field.
    Filters out actions >= 6.
    """
    try:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {txt_file}: {e}")
        return None
    
    if not lines:
        return None
        
    data = []
    # Header: frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions
    for i, line in enumerate(lines):
        if i == 0:
            continue # Skip header
            
        parts = line.strip().split(',')
        if len(parts) < 6:
            # Malformed or incomplete line
            continue
            
        try:
            # Extract fixed fields
            frame_id = parts[0]
            episode_id = parts[1]
            score = parts[2]
            duration = parts[3]
            unclipped_reward = parts[4]
            action = int(parts[5])
            
            # Action filtering requirement
            if action >= 6:
                continue
                
            # Gaze positions might contain multiple x,y pairs (variable commas)
            # Rejoin the remaining parts
            gaze_positions = ",".join(parts[6:])
            
            data.append({
                'frame_id': frame_id,
                'episode_id': episode_id,
                'score': score,
                'duration': duration,
                'unclipped_reward': unclipped_reward,
                'action': action,
                'gaze_positions': gaze_positions
            })
        except ValueError:
            # Handle cases where action column might not be numeric in bad data
            continue
            
    if not data:
        return None
        
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Preprocess Atari trajectories into individual CSV and atom valuation files.")
    parser.add_argument("--input_folder", type=str, default="data/seaquest/trajectories", help="Path to input .txt files.")
    parser.add_argument("--output_root", type=str, default="data/seaquest", help="Root directory for subdirectories.")
    parser.add_argument("--env", type=str, default="seaquest", help="Environment name (default: seaquest).")
    parser.add_argument("--device", type=str, default="cpu", help="Device for atom generation (cpu or cuda).")
    parser.add_argument("--single_traj", type=str, default=None, help="If provided, only process this specific .txt file name (e.g. 'traj1.txt').")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        return

    # Find all .txt trajectory files
    if args.single_traj:
        txt_files = [os.path.join(args.input_folder, args.single_traj)]
        if not os.path.exists(txt_files[0]):
            print(f"Error: Single trajectory file '{txt_files[0]}' not found.")
            return
    else:
        txt_files = glob.glob(os.path.join(args.input_folder, "*.txt"))
        txt_files.sort() # Ensure consistent order
        
    print(f"Found {len(txt_files)} trajectory file(s).")
    
    for txt_file in txt_files:
        traj_filename = os.path.basename(txt_file)
        traj_name = traj_filename.replace('.txt', '')
        
        print(f"\n{'='*20}")
        print(f"Processing trajectory: {traj_name}")
        print(f"{'='*20}")
        
        # Create destination subdirectory
        # dest_dir = os.path.join(args.output_root, traj_name)
        dest_dir = os.path.join(args.output_root, 'sub_trajectories')
        os.makedirs(dest_dir, exist_ok=True)
        
        # 1. Parse and filter the .txt file
        df = parse_trajectory_file(txt_file)
        
        if df is None or len(df) == 0:
            print(f"Skipping {traj_name}: No valid data found after action filtering.")
            continue
            
        # 2. Save to CSV in its own subdirectory
        # Similar format to data/seaquest/train.csv but for one trajectory
        csv_path = os.path.join(dest_dir, f"{traj_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved processed CSV: {csv_path} ({len(df)} records)")
        

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()
