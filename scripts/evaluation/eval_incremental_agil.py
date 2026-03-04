import os
import glob
import pandas as pd
import subprocess
import re

run_base_dir = "models/bc/incremental_agil"
output_csv = "bc_sample_efficiency_incremental_agil_50ep.csv"
episodes_to_eval = 50

# Collect all epoch folders
ep_dirs = glob.glob(os.path.join(run_base_dir, "*_ep"))

results = []

# Sort by the integer before '_ep'
sorted_dirs = sorted(ep_dirs, key=lambda x: int(os.path.basename(x).replace('_ep', '')))

for ep_dir in sorted_dirs:
    ep_num = int(os.path.basename(ep_dir).replace('_ep', ''))
    
    print(f"\nEvaluating Episode {ep_num}")
    
    cmd = [
        "conda", "run", "-n", "nesy-il", "python", "-u", "evaluate_bc_model.py",
        "--run_dir", ep_dir,
        "--ckpt_prefix", "best_",
        "--gaze_method", "AGIL",
        "--episodes", str(episodes_to_eval),
        "--seed", "42",
        "--stack", "1"
    ]
    
    try:
        # Run evaluation and capture stdout
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = result.stdout
        
        # Parse Mean Reward from output
        # Output format: Mean Reward: 100.00 ± 0.00
        match = re.search(r"Mean Reward:\s*([\d\.-]+)\s*±\s*([\d\.-]+)", out)
        if match:
            mean_reward = float(match.group(1))
            std_reward = float(match.group(2))
        else:
            print(f"Failed to parse output for episode {ep_num}")
            print(out)
            mean_reward = 0.0
            std_reward = 0.0
            
    except Exception as e:
        print(f"Error evaluating episode {ep_num}: {e}")
        mean_reward = 0.0
        std_reward = 0.0

    results.append({
        "num_episodes": ep_num,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "model_dir": ep_dir
    })
    
    # Save incrementally
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved: Episode {ep_num} -> {mean_reward} ± {std_reward}")

print(f"\nFinished! Saved all results to {output_csv}")
