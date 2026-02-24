import torch
from collections import Counter

pt_path = "data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt"
data = torch.load(pt_path, map_location='cpu', weights_only=False)
actions = data['actions']
if isinstance(actions, torch.Tensor):
    actions = actions.numpy()

counts = Counter(actions)
print(f"Action counts: {dict(sorted(counts.items()))}")
