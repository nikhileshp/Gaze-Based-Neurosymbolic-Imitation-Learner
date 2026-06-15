import torch
import os

path = "/home/nxp190053/Projects/Gaze-Based-Neurosymbolic-Imitation-Learner/data/freeway/Freeway_full_data_6_episodes_10p0_sigma_win_10_obj_12.pt"
if os.path.exists(path):
    data = torch.load(path, weights_only=False)
    if 'logic_state' in data:
        states = data['logic_state']
        print(f"Logic state shape: {states.shape}")
        # Print first object of first 5 frames to see trends
        for i in range(5):
            print(f"Frame {i}, Obj 0: {states[i, 0]}")
            print(f"Frame {i}, Obj 1: {states[i, 1]}")
        
        # Check ranges for each feature
        for f in range(6):
            feat = states[:, :, f]
            print(f"Feature {f}: min={feat.min().item():.2f}, max={feat.max().item():.2f}, mean={feat.mean().item():.2f}")
    else:
        print(f"Key 'logic_state' not found. Keys: {data.keys()}")
else:
    print(f"File not found: {path}")
