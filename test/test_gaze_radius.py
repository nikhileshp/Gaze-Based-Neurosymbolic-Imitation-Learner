import torch
import numpy as np
from scripts.generate_full_gaze_tensor import GazeToMask

gaze_mask_sigma = 15.0  # Gamma
gaze_mask_coef = 0.7    # Alpha
variance_expansion = 0.99 # Beta
k_window = 4

saliency_sigmas = [gaze_mask_sigma / (variance_expansion**d) for d in range(k_window + 1)]
coeficients = [gaze_mask_coef**d for d in range(k_window + 1)]

MASK = GazeToMask(84, saliency_sigmas, coeficients=coeficients)

# single fixation
temp_map = MASK.find_suitable_map(Nx2=168, index=0, mean_x=0.5, mean_y=0.5)

# Calculate radius (e.g. pixels > 0.5 or > 0.1)
print(f"Sigma=15")
print(f"Pixels > 0.5: {(temp_map > 0.5).sum().item()}")
print(f"Pixels > 0.1: {(temp_map > 0.1).sum().item()}")
print(f"Total Pixels: {84*84}")

