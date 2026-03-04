import os
import pandas as pd
import subprocess
import re

eval_configs = [
    {
        "name": "Mask (1-obj) 50ep",
        "run_dir": "models/bc/bc_mask_fewer_objs_1_max_epoch_50/all_ep",
        "gaze_method": "Mask",
        "gaze_model_path": "models/gaze_predictor/seaquest_gaze_predictor_limit_1.pth",
    },
    {
        "name": "Mask (2-obj) 50ep",
        "run_dir": "models/bc/bc_mask_fewer_objs_2_max_epoch_50/all_ep",
        "gaze_method": "Mask",
        "gaze_model_path": "models/gaze_predictor/seaquest_visual_gaze_predictor_limit_2.pth",
    }
]

output_csv = "fewer_objs_eval_50ep.csv"
episodes_to_eval = 50

results = []

# Load existing results if file exists to append
if os.path.exists(output_csv):
    try:
        existing_df = pd.read_csv(output_csv)
        results = existing_df.to_dict('records')
    except Exception as e:
        print(f"Warning: Could not load existing CSV: {e}")

for config in eval_configs:
    print(f"\n>>> Evaluating {config['name']} in {config['run_dir']}...")
    
    cmd = [
        "conda", "run", "-n", "nesy-il", "python", "-u", "evaluate_bc_model.py",
        "--run_dir", config['run_dir'],
        "--ckpt_prefix", "best_",
        "--gaze_method", config['gaze_method'],
        "--episodes", str(episodes_to_eval),
        "--seed", "42",
        "--stack", "1",
        "--use_gazemap"
    ]
    
    if config['gaze_model_path']:
        cmd += ["--gaze_model_path", config['gaze_model_path']]
    
    try:
        # Run evaluation and capture stdout
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = result.stdout
        
        # Parse Mean Reward from output
        match = re.search(r"Mean Reward:\s*([\d\.-]+)\s*±\s*([\d\.-]+)", out)
        if match:
            mean_reward = float(match.group(1))
            std_reward = float(match.group(2))
        else:
            print(f"Failed to parse output for {config['name']}")
            print("--- STDOUT ---")
            print(out)
            mean_reward = 0.0
            std_reward = 0.0
            
    except Exception as e:
        print(f"Error evaluating {config['name']}: {e}")
        mean_reward = 0.0
        std_reward = 0.0

    results.append({
        "config_name": config['name'],
        "run_dir": config['run_dir'],
        "gaze_method": config['gaze_method'],
        "mean_reward": mean_reward,
        "std_reward": std_reward
    })
    
    # Save incrementally
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Result: {config['name']} -> {mean_reward} ± {std_reward}")

print(f"\nFinished! All results updated in {output_csv}")
