import torch
import os

datasets = {
    "asterix": "/home/nxp190053/Projects/Gaze-Based-Neurosymbolic-Imitation-Learner/data/asterix/asterix/Asterix_full_data_12_episodes_10p0_sigma_win_10_obj_25.pt",
    "seaquest": "/home/nxp190053/Projects/Gaze-Based-Neurosymbolic-Imitation-Learner/data/seaquest/full_data_16_episodes_10p0_sigma_win_10_obj_limit_1.pt"
}

for name, path in datasets.items():
    if os.path.exists(path):
        print(f"\n--- {name.upper()} ---")
        data = torch.load(path, weights_only=False)
        print(f"Keys: {data.keys()}")
        state_key = 'logic_state' if 'logic_state' in data else 'states'
        if state_key in data:
            states = data[state_key]
            print(f"States shape: {states.shape}")
            print(f"First object (frame 0): {states[0, 0]}")
            for f in range(states.shape[-1]):
                feat = states[:, :, f]
                print(f"Feature {f}: min={feat.min().item():.2f}, max={feat.max().item():.2f}")
    else:
        print(f"File not found: {path}")
