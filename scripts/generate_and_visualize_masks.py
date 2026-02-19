import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import ast

# --- GazeToMask Class Definition (Copied from gabril_utils.py) ---
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
            temp_map.append(self.coeficients[i]*temp) # This should be element-wise multiplication

        temp_map = torch.stack(temp_map, 0)
        return temp_map

    def find_suitable_map(self, Nx2=168, index=0, mean_x=0.5, mean_y=0.5):
        start_x, start_y = int((1 - mean_x) * Nx2 / 2), int((1 - mean_y) * Nx2 / 2)
        
        start_x = max(0, min(start_x, Nx2 // 2))
        start_y = max(0, min(start_y, Nx2 // 2))
        
        desired_map = self.masks[index][start_y:start_y + Nx2 // 2, start_x:start_x + Nx2 // 2]
        return desired_map

    def find_bunch_of_maps(self, means=[[0.5, 0.5]], offset_start=0):
        # This function is not suitable for symmetric window logic directly.
        # We will manually iterate and call find_suitable_map.
        pass

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
        # print(f"Error parsing: {e}") 
        return []

def main():
    # Configuration
    csv_path = 'data/seaquest/train.csv'
    traj_dir = 'data/seaquest/trajectories/54_RZ_2461867_Aug-11-09-35-18'
    output_dir = 'data/seaquest/generated_masks'
    run_id = '2461867'
    k_window = 4  # Symmetric window size: [i-k, i+k]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=['frame_id', 'gaze_positions'])
    
    print(f"Filtering for run {run_id}...")
    target_df = df[df['frame_id'].str.contains(run_id, na=False)].reset_index(drop=True)
    
    if target_df.empty:
        print(f"No rows found matches run ID {run_id}. Exiting.")
        return

    print(f"Found {len(target_df)} rows for run {run_id}.")
    
    # Initialize GazeToMask
    # We need coefficients for distances d=0 to d=k
    # Sigma(d) = BaseSigma / (0.99^d)  (Assuming decay implies larger variance for further away points)
    # Coef(d) = BaseCoef / (0.99^d) ?? No, usually intensity decays with distance.
    # Original logic: coef[i] = gaze_mask_coef**(short_memory_length - i) where i goes 0..short_memory_length.
    # i=short_memory_length (closest/newest) -> coef^0 = 1.
    # i=0 (furthest) -> coef^k.
    # So Coef(d) = gaze_mask_coef**d ? (d=0 -> 1, d=k -> small)
    
    gaze_mask_sigma = 15.0  # Gamma (Variance Scale)
    gaze_mask_coef = 0.7    # Alpha (Intensity Decay)
    variance_expansion = 0.99 # Beta (Variance Expansion)
    
    # Sigmas for d=0..k (Variance increases with distance)
    # Sigma_d = Gamma / (Beta^d)
    saliency_sigmas = [gaze_mask_sigma / (variance_expansion**d) for d in range(k_window + 1)]
    
    # Coefficients for d=0..k (Intensity decays with distance)
    # Coef_d = Alpha^d
    coeficients = [gaze_mask_coef**d for d in range(k_window + 1)]
    
    print(f"Sigmas (d=0 to {k_window}): {saliency_sigmas}")
    print(f"Coefficients (d=0 to {k_window}): {coeficients}")
    
    # Initialize for distances 0..k
    MASK = GazeToMask(84, saliency_sigmas, coeficients=coeficients)
    
    # Select sample frames to process
    sample_indices = [100, 200, 300, 400, 500] 
    
    pbar = tqdm(total=len(sample_indices))
    
    for i in sample_indices:
        if i >= len(target_df):
            break
            
        current_row = target_df.iloc[i]
        frame_id_str = current_row['frame_id']
        frame_num = frame_id_str.split('_')[-1]
        
        # Collect window indices [i-k, i+k]
        start_idx = max(0, i - k_window)
        end_idx = min(len(target_df) - 1, i + k_window)
        
        # Accumulate mask
        accumulated_mask = torch.zeros([84, 84])
        
        # Iterate through window
        for j in range(start_idx, end_idx + 1):
            distance = abs(j - i)  # 0 if current frame, >0 otherwise
            
            # Get gaze points for row j
            row_gaze_str = target_df.iloc[j]['gaze_positions']
            gaze_pairs = parse_gaze_positions(row_gaze_str)
            
            # Add Gaussian for each point
            for (gx, gy) in gaze_pairs:
                # Use template corresponding to distance
                # Template index 'distance' maps to sigmas[distance]
                temp_map = MASK.find_suitable_map(Nx2=168, index=distance, mean_x=gx, mean_y=gy)
                accumulated_mask = accumulated_mask + temp_map
        
        # Normalize
        if accumulated_mask.max() > 0:
            accumulated_mask = accumulated_mask / accumulated_mask.max()
            
        mask_np = accumulated_mask.numpy()
        
        # Visualization
        img_filename = f"RZ_{run_id}_{frame_num}.png"
        img_path = os.path.join(traj_dir, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            pbar.update(1)
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            pbar.update(1)
            continue
            
        heatmap = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
        heatmap_norm = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        
        base_fn = f"mask_vis_{frame_num}"
        cv2.imwrite(os.path.join(output_dir, f"{base_fn}_mask.png"), heatmap_norm)
        cv2.imwrite(os.path.join(output_dir, f"{base_fn}_overlay.png"), overlay)
        cv2.imwrite(os.path.join(output_dir, f"{base_fn}_original.png"), img)
        
        pbar.update(1)
        
    pbar.close()
    print(f"Processing complete. Saved to {output_dir}")

if __name__ == "__main__":
    main()
