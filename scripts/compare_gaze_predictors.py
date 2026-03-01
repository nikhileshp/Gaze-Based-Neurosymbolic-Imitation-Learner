import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from tqdm import tqdm
from scripts.data_utils import load_gaze_predictor_data
from scripts.gaze_predictor import HumanGazeNet

def compare_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load data
    dataset_path = "data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt"
    # load_gaze_predictor_data returns (N, H, W, stack), (T, H, W), and valid_indices
    imgs_nhwc, gaze_masks, valid_indices = load_gaze_predictor_data(
        dataset_path, frame_stack=4, device=device
    )

    print(f"\nProcessing {len(imgs_nhwc)} samples...")

    # Normalize ground truth masks to sum to 1 (probability distribution)
    # Gaze masks are (T, H, W). We only want the ones corresponding to valid_indices
    gt_masks = gaze_masks[valid_indices].unsqueeze(1) # (N, 1, 84, 84)
    # Add epsilon to prevent division by zero
    gt_sums = gt_masks.sum(dim=(2, 3), keepdim=True) + 1e-10
    gt_masks = gt_masks / gt_sums

    # 2. Find models
    model_dir = "models/gaze_predictor/"
    models = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    print(f"\nFound {len(models)} models: {models}\n")

    batch_size = 256
    results = {}

    # Convert NHWC -> NCHW for PyTorch model input
    imgs_nchw = torch.tensor(imgs_nhwc, dtype=torch.float32).permute(0, 3, 1, 2)

    # 3. Evaluate each model
    for model_name in models:
        model_path = os.path.join(model_dir, model_name)
        print(f"=== Evaluating {model_name} ===")
        
        # Initialize and load
        model = HumanGazeNet(in_channels=4).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        total_mse = 0.0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(imgs_nchw), batch_size), desc=f"Predicting"):
                batch_imgs = imgs_nchw[i:i+batch_size].to(device)
                batch_gt = gt_masks[i:i+batch_size].to(device)

                # Predictions are already softmaxed (sum to 1)
                preds = model(batch_imgs)
                
                # Compute MSE sum for the batch
                mse = torch.nn.functional.mse_loss(preds, batch_gt, reduction='sum')
                total_mse += mse.item()

        # Compute average MSE over all samples
        # Divided by number of samples. The loss is computed per pixel, but
        # sum over spatial dims makes sense when we mean over batch.
        # Let's mean over everything (N, C, H, W)
        avg_mse = total_mse / (len(imgs_nchw) * 1 * 84 * 84)
        results[model_name] = avg_mse
        print(f"Average MSE: {avg_mse:.8f}\n")

    # 4. Report
    print("=== SUMMARY OF RESULTS ===")
    print(f"{'Model Name':<45} | {'MSE':<15}")
    print("-" * 65)
    
    # Sort by MSE (lowest is best)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for name, mse in sorted_results:
        print(f"{name:<45} | {mse:.8f}")

    print(f"\nMost accurate model (lowest MSE): {sorted_results[0][0]}")

if __name__ == "__main__":
    compare_models()
