import pandas as pd
import torch
import numpy as np
import os
from tqdm import tqdm

# --- GazeToMask Class Definition ---
class GazeToMask():
    def __init__(self, N=84, sigmas=[10,10,10,10], coeficients = [1,1,1,1]):
        self.N = N
        assert len(sigmas) == len(coeficients)
        self.sigmas = sigmas
        self.coeficients = coeficients
        self.masks = self.initialize_mask()

    def generate_single_gaussian_tensor(self, size, center_x, center_y, sigma):
        x = torch.arange(0, size, 1).float()
        y = torch.arange(0, size, 1).float()
        y, x = torch.meshgrid(y, x)
        
        gaussian = torch.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        return gaussian

    def initialize_mask(self):
        temp_map = []
        N = self.N
        for i in range(len(self.sigmas)):
            # The indices in sigmas correspond to "distance" (0=current, 1=next/prev, etc.)
            temp = self.generate_single_gaussian_tensor(2 * N, N - 1, N - 1, self.sigmas[i])
            temp = temp/ temp.max()
            temp_map.append(self.coeficients[i]*temp) 

        temp_map = torch.stack(temp_map, 0)
        return temp_map

    def find_suitable_map(self, Nx2=168, index=0, mean_x=0.5, mean_y=0.5):
        start_x, start_y = int((1 - mean_x) * Nx2 / 2), int((1 - mean_y) * Nx2 / 2)
        
        start_x = max(0, min(start_x, Nx2 // 2))
        start_y = max(0, min(start_y, Nx2 // 2))
        
        desired_map = self.masks[index][start_y:start_y + Nx2 // 2, start_x:start_x + Nx2 // 2]
        return desired_map

# --- Utility Functions ---

def parse_gaze_positions(gaze_str):
    if pd.isna(gaze_str):
        return []
    try:
        parts = [float(x) for x in gaze_str.split(',')]
        pairs = []
        for i in range(0, len(parts), 2):
            if i+1 < len(parts):
                # Normalize to 0-1 range for 160x210 frame
                x_norm = parts[i] / 160.0
                y_norm = parts[i+1] / 210.0
                pairs.append([x_norm, y_norm])
        return pairs
    except Exception as e:
        return []

def main():
    # Configuration
    csv_path = 'data/seaquest/train.csv'
    output_tensor_path = 'data/seaquest/gaze_masks.pt'
    k_window = 4  # Symmetric window size
    
    # Parameters
    gaze_mask_sigma = 15.0  # Gamma
    gaze_mask_coef = 0.7    # Alpha
    variance_expansion = 0.99 # Beta
    
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=['frame_id', 'gaze_positions', 'episode_id'])
    # Optional: Filter or sort? train.csv should be ordered by frame_id usually.
    # But frame_id is a string that might not sort numerically easily across different episodes.
    # We should assume the CSV order IS the temporal order.
    
    print(f"Found {len(df)} rows. Generating full gaze tensor...")
    
    # Sigmas and coefficients
    saliency_sigmas = [gaze_mask_sigma / (variance_expansion**d) for d in range(k_window + 1)]
    coeficients = [gaze_mask_coef**d for d in range(k_window + 1)]
    
    MASK = GazeToMask(84, saliency_sigmas, coeficients=coeficients)
    
    # Pre-allocate tensor
    # Shape: (N, 84, 84)
    # Use float16 to save space? Standard implies float32.
    # 84*84 * 4 bytes * 1M rows ~= 28GB.
    # 84*84 * 17200 rows (one run) ~= 480MB.
    # How big is train.csv?
    # If 80GB CSV... this tensor will be huge.
    # Wait, the user said "Store the mask for every frame in train.csv as a tensor".
    # If it fits in memory.
    # Let's check CSV size first.
    # But I'll assume for now I process it all.
    
    # Note: Processing row by row and stacking might be slow and memory intensive if list grows.
    # Better to pre-allocate if we know length.
    num_frames = len(df)
    print(f"Total frames: {num_frames}")
    
    # List to store tensors, then stack (might be heavy)
    # Or write to disk incrementally?
    # torch.save usually expects full object.
    # Let's try to collect in CPU memory.
    
    all_masks = []
    
    pbar = tqdm(total=num_frames)
    
    # Optimize: Pre-parse gaze positions?
    # Parsing strings is slow.
    # We can do it on the fly.
    
    # We need to be careful with window boundaries across episodes.
    # Symmetric window shouldn't cross episodes ideally.
    # We can group by episode_id.
    
    episode_groups = df.groupby('episode_id')
    
    # We need to maintain the original order for the final tensor to match CSV rows index-wise.
    # So we should iterate the DF, but handle boundaries.
    # Actually, grouping by episode_id and processing each group is safer.
    # But we need to put them back in original order.
    # Does groupby preserve order? Yes if sort=False.
    
    # Let's process by episode to respect boundaries.
    
    # Create a tensor of zeros to fill
    full_tensor = torch.zeros((num_frames, 84, 84), dtype=torch.float32)
    
    current_idx = 0
    
    for episode_id, group in tqdm(df.groupby('episode_id', sort=False)):
        # Reset index for the group to 0..n
        group = group.reset_index(drop=True)
        group_len = len(group)
        
        # We need to fill full_tensor[current_idx : current_idx + group_len]
        
        # Extract gaze lists for this episode
        # Optimization: Parse all strings in this group first?
        # Maybe
        
        for i in range(group_len):
            # Window in group context
            start_idx = max(0, i - k_window)
            end_idx = min(group_len - 1, i + k_window)
            
            accumulated_mask = torch.zeros([84, 84])
            
            for j in range(start_idx, end_idx + 1):
                distance = abs(j - i)
                row_gaze_str = group.iloc[j]['gaze_positions']
                gaze_pairs = parse_gaze_positions(row_gaze_str)
                
                for (gx, gy) in gaze_pairs:
                    temp_map = MASK.find_suitable_map(Nx2=168, index=distance, mean_x=gx, mean_y=gy)
                    accumulated_mask = accumulated_mask + temp_map
            
            if accumulated_mask.max() > 0:
                accumulated_mask = accumulated_mask / accumulated_mask.max()
            
            full_tensor[current_idx + i] = accumulated_mask
            
        current_idx += group_len
            
    print(f"Saving tensor to {output_tensor_path}...")
    torch.save(full_tensor, output_tensor_path)
    print("Done.")

if __name__ == "__main__":
    main()
