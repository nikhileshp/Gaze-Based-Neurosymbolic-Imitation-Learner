import json
import torch

try:
    with open('results/bs_data/getout.json', 'r') as f:
        data = json.load(f)

    print("Keys in dataset:", data.keys())
    
    idx = 0
    print(f"\n--- State at Index {idx} ---")
    for key, value in data.items():
        if isinstance(value, list):
            # Handle nested lists/tensors if necessary, but for now just print the item
            item = value[idx]
            # If it's a long list (like logic state), maybe truncate or summarize?
            # Let's just print it.
            print(f"{key}: {item}")
        else:
            print(f"{key}: {value} (Not a list)")

except Exception as e:
    print(f"Error: {e}")
