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
    import argparse
    parser = argparse.ArgumentParser(description="Generate full gaze tensor")
    parser.add_argument('--csv_path', type=str, default='data/seaquest/train.csv', help='Path to input CSV')
    parser.add_argument('--output_tensor_path', type=str, default='data/seaquest/gaze_masks.pt', help='Path to output tensor')
    parser.add_argument('--gaze_mask_sigma', type=float, default=5.0, help='Sigma for gaze mask (Gamma)')
    args = parser.parse_args()

    # Configuration
    csv_path = args.csv_path
    output_tensor_path = args.output_tensor_path
    k_window = 4  # Symmetric window size
    
    # Parameters
    gaze_mask_sigma = args.gaze_mask_sigma  # Gamma
    gaze_mask_coef = 0.7    # Alpha
    variance_expansion = 0.99 # Beta
    
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=['frame_id', 'gaze_positions', 'episode_id'])
    # Optional: Filter or sort? train.csv should be ordered by frame_id usually.
    # But frame_id is a string that might not sort numerically easily across different episodes.
    # We should assume the CSV order IS the temporal order.
    
    print(f"Found {len(df)} rows. Generating full gaze tensor...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Sigmas and coefficients
    saliency_sigmas = [gaze_mask_sigma / (variance_expansion**d) for d in range(k_window + 1)]
    coeficients = [gaze_mask_coef**d for d in range(k_window + 1)]
    
    MASK = GazeToMask(84, saliency_sigmas, coeficients=coeficients)
    MASK.masks = MASK.masks.to(device)
    
    num_frames = len(df)
    print(f"Total frames: {num_frames}")
    
    # We should keep full_tensor on CPU if it's too large, but doing accumulations on GPU.
    full_tensor = torch.zeros((num_frames, 84, 84), dtype=torch.float32)
    
    current_idx = 0
    
    pbar = tqdm(total=num_frames, desc="Generating mask")
    for episode_id, group in df.groupby('episode_id', sort=False):
        group = group.reset_index(drop=True)
        group_len = len(group)
        
        for i in range(group_len):
            start_idx = max(0, i - k_window)
            end_idx = min(group_len - 1, i + k_window)
            
            accumulated_mask = torch.zeros([84, 84], device=device)
            
            for j in range(start_idx, end_idx + 1):
                distance = abs(j - i)
                row_gaze_str = group.iloc[j]['gaze_positions']
                gaze_pairs = parse_gaze_positions(row_gaze_str)
                
                for (gx, gy) in gaze_pairs:
                    temp_map = MASK.find_suitable_map(Nx2=168, index=distance, mean_x=gx, mean_y=gy)
                    accumulated_mask = accumulated_mask + temp_map
            
            if accumulated_mask.max() > 0:
                accumulated_mask = accumulated_mask / accumulated_mask.max()
            
            full_tensor[current_idx + i] = accumulated_mask.cpu()
            pbar.update(1)
            
        current_idx += group_len
            
    pbar.close()
    print(f"Saving tensor to {output_tensor_path}...")
    torch.save(full_tensor, output_tensor_path)
    print("Done.")

if __name__ == "__main__":
    main()
