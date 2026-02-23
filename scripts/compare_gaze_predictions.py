import argparse
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from load_data import Dataset
from gaze_predictor import Human_Gaze_Predictor

def make_video(args):
    print("Loading Dataset...")
    d = Dataset(args.trajectories_dir, args.labels_csv)
    d.generate_data_for_gaze_prediction()

    print(f"Loading ground truth masks from {args.gaze_masks}...")
    masks_tensor = torch.load(args.gaze_masks, map_location='cpu')

    print(f"Loading Gaze Predictor Model from {args.model_weights}...")
    gp = Human_Gaze_Predictor(args.game_name)
    gp.init_model(args.model_weights)
    gp.model.eval()

    # Predict all gaze for valid indices
    valid_indices = d.original_indices[3:]
    
    num_frames = min(args.num_frames, len(d.gaze_imgs))
    
    # Convert imgs to tensor NCHW
    imgs_tensor = torch.tensor(d.gaze_imgs[:num_frames], dtype=torch.float32).permute(0, 3, 1, 2).to(gp.device)
    
    print("Generating predictions...")
    with torch.no_grad():
        preds = []
        batch_size = 64
        for i in tqdm(range(0, len(imgs_tensor), batch_size)):
             batch = imgs_tensor[i:i+batch_size]
             out = gp.model(batch)
             preds.append(out.cpu().numpy())
        preds = np.concatenate(preds, axis=0) # (num_frames, 1, 84, 84)

    # Make video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = 84, 84
    scale = 4
    out_h = h * scale
    out_w = w * scale * 3
    out = cv2.VideoWriter(args.output, fourcc, 15.0, (out_w, out_h))
    
    print("Creating video frames...")
    for i in tqdm(range(num_frames)):
        orig = (d.train_imgs[i+3] * 255).astype(np.uint8)
        orig_color = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        orig_resized = cv2.resize(orig_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        
        # Ground truth
        gt_mask = masks_tensor[valid_indices[i]].numpy()
        mx = gt_mask.max()
        if mx > 0: gt_mask = gt_mask / mx
        
        gt_color = cv2.applyColorMap(np.clip(gt_mask * 255 * args.multiplier, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        gt_resized = cv2.resize(gt_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        gt_overlay = cv2.addWeighted(orig_resized, 0.4, gt_resized, 0.6, 0)
        
        # Predicted
        pred_mask = preds[i, 0]
        mx2 = pred_mask.max()
        if mx2 > 0: pred_mask = pred_mask / mx2
        
        pred_color = cv2.applyColorMap(np.clip(pred_mask * 255 * args.multiplier, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        pred_resized = cv2.resize(pred_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        pred_overlay = cv2.addWeighted(orig_resized, 0.4, pred_resized, 0.6, 0)
        
        # Combine horizontally
        combined = np.hstack((orig_resized, gt_overlay, pred_overlay))
        
        # Add labels
        cv2.putText(combined, "Original Game", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(combined, "Ground Truth Gaze", (w*scale + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(combined, "Predicted Gaze", (w*scale*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        out.write(combined)
        
    out.release()
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories_dir', default="data/seaquest/trajectories")
    parser.add_argument('--labels_csv', default="data/seaquest/train_data_16_traj.csv")
    parser.add_argument('--game_name', default="seaquest")
    parser.add_argument('--model_weights', default="seaquest_gaze_predictor_2.pth")
    parser.add_argument('--gaze_masks', default="data/seaquest/gaze_masks.pt")
    parser.add_argument('--output', default="gaze_comparison_2.mp4")
    parser.add_argument('--num_frames', type=int, default=3000)
    parser.add_argument('--multiplier', type=float, default=2.0)
    args = parser.parse_args()
    make_video(args)
