import pandas as pd
import torch
import numpy as np
import os
import cv2
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
    # Use the specific trajectory requested
    traj_dir = 'data/seaquest/trajectories/54_RZ_2461867_Aug-11-09-35-18'
    output_video_path = 'data/seaquest/trajectory_54_RZ.mp4'
    run_id = '2461867'
    k_window = 4  # Symmetric window size
    
    # Parameters
    gaze_mask_sigma = 15.0  # Gamma
    gaze_mask_coef = 0.7    # Alpha
    variance_expansion = 0.99 # Beta
    
    # Check dependencies
    if not os.path.exists(traj_dir):
        print(f"Trajectory directory not found: {traj_dir}")
        return

    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=['frame_id', 'gaze_positions'])
    
    print(f"Filtering for run {run_id}...")
    target_df = df[df['frame_id'].str.contains(run_id, na=False)].reset_index(drop=True)
    
    if target_df.empty:
        print(f"No rows found matches run ID {run_id}. Exiting.")
        return

    print(f"Found {len(target_df)} rows. Generating video...")
    
    # Sigmas and coefficients
    saliency_sigmas = [gaze_mask_sigma / (variance_expansion**d) for d in range(k_window + 1)]
    coeficients = [gaze_mask_coef**d for d in range(k_window + 1)]
    
    MASK = GazeToMask(84, saliency_sigmas, coeficients=coeficients)
    
    # Video Writer
    # Assume frame size 160x210
    frame_width = 160
    frame_height = 210
    fps = 30  # Standard frame rate
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Process all frames
    # Sort to be safe, though CSV is usually ordered
    # Extract frame number for sorting
    # Format: RZ_2461867_1
    target_df['frame_num'] = target_df['frame_id'].apply(lambda x: int(x.split('_')[-1]))
    target_df = target_df.sort_values('frame_num').reset_index(drop=True)
    
    # Limit number of frames if needed for testing (e.g. first 2000)
    # target_df = target_df.iloc[:2000]
    
    pbar = tqdm(total=len(target_df))
    
    for i in range(len(target_df)):
        current_row = target_df.iloc[i]
        frame_num = current_row['frame_num']
        
        # 1. Generate Mask (Window Logic)
        start_idx = max(0, i - k_window)
        end_idx = min(len(target_df) - 1, i + k_window)
        
        accumulated_mask = torch.zeros([84, 84])
        
        for j in range(start_idx, end_idx + 1):
            distance = abs(j - i)
            row_gaze_str = target_df.iloc[j]['gaze_positions']
            gaze_pairs = parse_gaze_positions(row_gaze_str)
            
            for (gx, gy) in gaze_pairs:
                temp_map = MASK.find_suitable_map(Nx2=168, index=distance, mean_x=gx, mean_y=gy)
                accumulated_mask = accumulated_mask + temp_map
        
        if accumulated_mask.max() > 0:
            accumulated_mask = accumulated_mask / accumulated_mask.max()
            
        mask_np = accumulated_mask.numpy()
        
        # 2. Get Image
        img_filename = f"RZ_{run_id}_{frame_num}.png"
        img_path = os.path.join(traj_dir, img_filename)
        
        if not os.path.exists(img_path):
            # Skip or write black frame? Skip is safer to keep sync if possible, or print warning.
            # print(f"Warning: Image {img_filename} not found.")
            pbar.update(1)
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            pbar.update(1)
            continue
            
        # 3. Create Overlay
        heatmap = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
        heatmap_norm = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        
        # 4. Plot Specific Gaze Points (Current Frame Only)
        current_gaze_str = current_row['gaze_positions']
        current_gaze_pairs = parse_gaze_positions(current_gaze_str)
        
        for (gx, gy) in current_gaze_pairs:
            # Convert normalized coordinates back to pixel coordinates
            px = int(gx * frame_width)
            py = int(gy * frame_height)
            
            # Draw marker (White circle with black outline for visibility)
            cv2.circle(overlay, (px, py), 2, (0, 0, 0), -1) # Black center
            cv2.circle(overlay, (px, py), 3, (255, 255, 255), 1) # White rim
            
        # 5. Write to Video
        out.write(overlay)
        pbar.update(1)
        
    out.release()
    pbar.close()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    main()
