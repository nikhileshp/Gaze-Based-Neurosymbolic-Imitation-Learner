import torch
import os

path = "/home/nxp190053/Projects/Gaze-Based-Neurosymbolic-Imitation-Learner/data/freeway/Freeway_full_data_6_episodes_10p0_sigma_win_10_obj_12.pt"
if os.path.exists(path):
    data = torch.load(path, weights_only=False)
    print(f"Keys: {data.keys()}")
    if 'states' in data:
        states = data['states']
        print(f"States shape: {states.shape}")
        # Check first object of first frame
        print(f"First object features (frame 0, obj 0): {states[0, 0]}")
        print(f"Feature dimension: {states.shape[-1]}")
    elif 'logic_states' in data:
        states = data['logic_states']
        print(f"Logic states shape: {states.shape}")
        print(f"First object features: {states[0, 0]}")
        print(f"Feature dimension: {states.shape[-1]}")
    else:
        # Just print some keys and shapes
        for k, v in data.items():
            if hasattr(v, 'shape'):
                print(f"{k} shape: {v.shape}")
            else:
                print(f"{k}: {type(v)}")
else:
    print(f"File not found: {path}")
