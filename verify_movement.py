import torch
import numpy as np

pt_path = "data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt"
data = torch.load(pt_path, map_location='cpu', weights_only=False)
logic = data['logic_state'] # (N, 49, 7)
actions = data['actions'] # (N,)

# Player is obj0. Y-coord is index 2.
player_y = logic[:, 0, 2]

up_diffs = []
down_diffs = []

for i in range(len(actions)-1):
    if actions[i] == 2: # UP
        up_diffs.append(player_y[i+1] - player_y[i])
    elif actions[i] == 5: # DOWN
        down_diffs.append(player_y[i+1] - player_y[i])

print(f"UP (2) mean Y diff: {np.mean(up_diffs):.4f} (expected negative if Y increases downwards)")
print(f"DOWN (5) mean Y diff: {np.mean(down_diffs):.4f} (expected positive if Y increases downwards)")

# Also check LEFT/RIGHT (X at index 1)
player_x = logic[:, 0, 1]
left_diffs = []
right_diffs = []
for i in range(len(actions)-1):
    if actions[i] == 4: # LEFT
        left_diffs.append(player_x[i+1] - player_x[i])
    elif actions[i] == 3: # RIGHT
        right_diffs.append(player_x[i+1] - player_x[i])

print(f"LEFT (4) mean X diff: {np.mean(left_diffs):.4f} (expected negative)")
print(f"RIGHT (3) mean X diff: {np.mean(right_diffs):.4f} (expected positive)")
