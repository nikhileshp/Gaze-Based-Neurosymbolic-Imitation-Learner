import os
import argparse
import glob
import pandas as pd
import numpy as np
import torch

from nudge.agents.imitation_agent import ImitationAgent
from nudge.env import NudgeBaseEnv
from nudge.utils import make_deterministic
from evaluate_model import evaluate

try:
    from scripts.gaze_predictor import Human_Gaze_Predictor
except ImportError:
    Human_Gaze_Predictor = None

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple models over training episodes")
    parser.add_argument("--num_episodes", type=int, required=True, help="Total number of episodes/files to evaluate")
    parser.add_argument("--use_gazemap", action="store_true", help="Set to use gaze models")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate folder to lookup")
    parser.add_argument("--env", type=str, default="seaquest", help="Environment name")
    parser.add_argument("--rules", type=str, default="new", help="Ruleset name")
    parser.add_argument("--eval_episodes", type=int, default=50, help="Number of episodes to evaluate each model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--gaze_threshold", type=float, default=50.0, help="Gaze threshold if use_gazemap is set")
    parser.add_argument("--gaze_model_path", type=str, default="seaquest_gaze_predictor_2.pth", help="Path to gaze predictor weights")
    args = parser.parse_args()

    # Determine Device
    device_name = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Construct the folder path based on arguments
    gaze_dir = "with_gaze" if args.use_gazemap else "no_gaze"
    folder_path = os.path.join("out", "imitation", gaze_dir, f"lr_{args.lr}")
    
    if not os.path.exists(folder_path):
        print(f"Error: Directory {folder_path} does not exist.")
        return

    # Initialize Gaze Predictor if use_gazemap is True
    gaze_predictor = None
    if args.use_gazemap:
        if Human_Gaze_Predictor is None:
            print("Error: Could not import Human_Gaze_Predictor. Ensure gaze_predictor.py is in scripts/.")
            return
        print(f"Initializing Gaze Predictor from {args.gaze_model_path}...")
        gaze_predictor = Human_Gaze_Predictor(args.env)
        gaze_predictor.init_model(args.gaze_model_path)
        gaze_predictor.model.eval()

    # Set seed
    make_deterministic(args.seed)

    # Initialize Environment
    print(f"Initializing Environment: {args.env}...")
    env = NudgeBaseEnv.from_name(args.env, mode='logic')

    # Initialize Agent
    print(f"Initializing Agent for rules: {args.rules}...")
    gaze_threshold = args.gaze_threshold if args.use_gazemap else None
    agent = ImitationAgent(args.env, args.rules, device, gaze_threshold=gaze_threshold)
    
    results = []

    for ep in range(1, args.num_episodes + 1):
        # We need to find the exact file for this epoch.
        # It could be named *epoch_{ep}_* or *traj_{ep}_*
        search_pattern_1 = os.path.join(folder_path, f"*epoch_{ep}_*.pth")
        search_pattern_2 = os.path.join(folder_path, f"*traj_{ep}_*.pth")
        
        files = glob.glob(search_pattern_1) + glob.glob(search_pattern_2)
        if not files:
            print(f"Warning: Could not find model for episode/epoch {ep} in {folder_path}")
            continue
            
        model_path = files[0] # Take the first matching pattern
        print(f"\n--- Evaluating Model for Episode/Epoch {ep} ---")
        print(f"Loading model from {model_path}...")
        try:
            agent.load(model_path)
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue

        rewards = evaluate(agent, env, num_episodes=args.eval_episodes, gaze_predictor=gaze_predictor, seed=args.seed)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        print(f"Episode/Epoch {ep} (Model: {os.path.basename(model_path)}): Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        
        results.append({
            'trajectory': ep,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'gaze': args.use_gazemap
        })

    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        csv_filename = f"{args.env}_{args.rules}_lr_{args.lr}_{gaze_dir}_eval_results.csv"
        csv_path = os.path.join(folder_path, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved evaluation results to {csv_path}")
        print("You can plot this using: python plot_learning_curve.py --csv_path " + csv_path)
    else:
        print("No models were evaluated.")

if __name__ == "__main__":
    main()
