import os
import glob
import pandas as pd
import numpy as np
import subprocess
import re

run_dir = "models/Seaquest/AGIL/independent_seed_42_lr_0.0001_nep28_zdim256_stack_4_epochs_10_gaze_AGIL/2026_03_01_12_31_57"
output_csv = "eval_every_10_epochs.csv"
episodes_to_eval = 50

# Collect all epoch folders
epoch_dirs = glob.glob(os.path.join(run_dir, "ep*"))

results = []

for ep_dir in sorted(epoch_dirs, key=lambda x: int(os.path.basename(x).replace('ep', ''))):
    epoch_num = int(os.path.basename(ep_dir).replace('ep', ''))
    
    # In independent mode, every 10 epochs is 1 trajectory
    trajectory = epoch_num // 10
    
    print(f"\\nEvaluating Epoch {epoch_num} (Trajectory {trajectory})")
    
    cmd = [
        "conda", "run", "-n", "nesy-il", "python", "-u", "evaluate_bc_model.py",
        "--run_dir", ep_dir,
        "--ckpt_prefix", f"ep{epoch_num}_",
        "--gaze_method", "AGIL",
        "--episodes", str(episodes_to_eval),
        "--seed", "42"
    ]
    
    try:
        # Run evaluation and capture stdout
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = result.stdout
        
        # Parse Mean Reward from output
        # Output format: Mean Reward: 100.00 ± 0.00
        match = re.search(r"Mean Reward:\s*([\d\.]+)\s*±\s*([\d\.]+)", out)
        if match:
            mean_reward = float(match.group(1))
            std_reward = float(match.group(2))
        else:
            print(f"Failed to parse output for epoch {epoch_num}")
            mean_reward = 0.0
            std_reward = 0.0
            
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating epoch {epoch_num}: {e}")
        mean_reward = 0.0
        std_reward = 0.0

    results.append({
        "trajectory": trajectory,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "model": f"AGIL_indep_ep{epoch_num}"
    })
    
    # Save incrementally
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved: Trajectory {trajectory} -> {mean_reward} ± {std_reward}")

print(f"\\nFinished! Saved all results to {output_csv}")
