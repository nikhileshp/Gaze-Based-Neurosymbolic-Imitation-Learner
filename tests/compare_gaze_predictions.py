import sys, os as _os
_scripts_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_project_root = _os.path.dirname(_scripts_dir)
for _p in [_scripts_dir, _project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import argparse
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from scripts.gaze.gaze_predictor import Human_Gaze_Predictor


def load_from_pt(pt_path, num_frames):
    """Load observations and ground-truth gaze masks from a .pt dataset file."""
    print(f"Loading .pt dataset from {pt_path}...")
    data = torch.load(pt_path, map_location='cpu', weights_only=False)

    obs = data['observations']  # (N, 84, 84) uint8
    if not isinstance(obs, torch.Tensor):
        obs = torch.from_numpy(obs)
    obs = obs.float() / 255.0  # normalise to [0, 1]

    gaze_masks = data.get('gaze_image', None)  # (N, 84, 84) float32
    if gaze_masks is not None and not isinstance(gaze_masks, torch.Tensor):
        gaze_masks = torch.from_numpy(gaze_masks)

    num_frames = min(num_frames, len(obs))

    # Build stacked frames (stack of 4 for gaze predictor input)
    print("Building frame stacks for gaze predictor...")
    stacked = []
    for i in tqdm(range(num_frames)):
        frames = []
        for k in range(4):
            idx = max(0, i - (3 - k))
            frames.append(obs[idx].numpy())
        stacked.append(np.stack(frames, axis=-1))  # (84, 84, 4)
    stacked = np.array(stacked)  # (N, 84, 84, 4)

    # Original single-frame grayscale images for display
    orig_imgs = obs[:num_frames].numpy()  # (N, 84, 84) float [0,1]

    # Gaze masks for ground truth
    gt_masks = gaze_masks[:num_frames] if gaze_masks is not None else None

    return stacked, orig_imgs, gt_masks, num_frames


def load_from_csv(args):
    """Load data from trajectories directory + CSV (legacy flow)."""
    from load_data import Dataset
    print("Loading Dataset from CSV...")
    d = Dataset(args.trajectories_dir, args.labels_csv)
    d.generate_data_for_gaze_prediction()

    print(f"Loading ground truth masks from {args.gaze_masks}...")
    masks_tensor = torch.load(args.gaze_masks, map_location='cpu')

    valid_indices = d.original_indices[3:]
    num_frames = min(args.num_frames, len(d.gaze_imgs))

    # gaze_imgs is already (N, 84, 84, 4) NHWC
    stacked = d.gaze_imgs[:num_frames]

    # Original grayscale frames (single channel for display)
    orig_imgs = d.train_imgs[3:3+num_frames]  # (N, 84, 84) float [0,1]

    # Ground truth masks indexed by valid_indices
    gt_masks_list = []
    for i in range(num_frames):
        gt_masks_list.append(masks_tensor[valid_indices[i]])
    gt_masks = torch.stack(gt_masks_list)  # (N, 84, 84)

    return stacked, orig_imgs, gt_masks, num_frames


def make_video(args):
    # ── Load data ─────────────────────────────────────────────────────────────
    if args.dataset and os.path.exists(args.dataset):
        stacked, orig_imgs, gt_masks, num_frames = load_from_pt(args.dataset, args.num_frames)
    else:
        stacked, orig_imgs, gt_masks, num_frames = load_from_csv(args)

    # ── Load gaze predictor ───────────────────────────────────────────────────
    print(f"Loading Gaze Predictor Model from {args.model_weights}...")
    gp = Human_Gaze_Predictor(args.game_name)
    gp.init_model(args.model_weights)
    gp.model.eval()

    # Convert stacked imgs to NCHW tensor for prediction
    imgs_tensor = torch.tensor(stacked, dtype=torch.float32).permute(0, 3, 1, 2).to(gp.device)

    print("Generating predictions...")
    with torch.no_grad():
        preds = []
        batch_size = 64
        for i in tqdm(range(0, len(imgs_tensor), batch_size)):
            batch = imgs_tensor[i:i+batch_size]
            out = gp.model(batch)
            preds.append(out.cpu().numpy())
        preds = np.concatenate(preds, axis=0)  # (num_frames, 1, 84, 84)

    # ── Create video ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = 84, 84
    scale = 4
    out_h = h * scale

    has_gt = gt_masks is not None
    num_panels = 3 if has_gt else 2
    out_w = w * scale * num_panels
    out = cv2.VideoWriter(args.output, fourcc, 15.0, (out_w, out_h))

    print("Creating video frames...")
    for i in tqdm(range(num_frames)):
        # Original frame (grayscale -> BGR for display)
        orig = (orig_imgs[i] * 255).astype(np.uint8)
        orig_color = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        orig_resized = cv2.resize(orig_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        # Predicted gaze
        pred_mask = preds[i, 0]
        mx2 = pred_mask.max()
        if mx2 > 0:
            pred_mask = pred_mask / mx2
        pred_color = cv2.applyColorMap(np.clip(pred_mask * 255 * args.multiplier, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        pred_resized = cv2.resize(pred_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        pred_overlay = cv2.addWeighted(orig_resized, 0.4, pred_resized, 0.6, 0)

        if has_gt:
            # Ground truth gaze
            gt_mask = gt_masks[i].numpy() if isinstance(gt_masks[i], torch.Tensor) else gt_masks[i]
            mx = gt_mask.max()
            if mx > 0:
                gt_mask = gt_mask / mx
            gt_color = cv2.applyColorMap(np.clip(gt_mask * 255 * args.multiplier, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            gt_resized = cv2.resize(gt_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
            gt_overlay = cv2.addWeighted(orig_resized, 0.4, gt_resized, 0.6, 0)

            combined = np.hstack((orig_resized, gt_overlay, pred_overlay))
            cv2.putText(combined, "Original Game", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Ground Truth Gaze", (w * scale + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Predicted Gaze", (w * scale * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            combined = np.hstack((orig_resized, pred_overlay))
            cv2.putText(combined, "Original Game", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Predicted Gaze", (w * scale + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(combined)

    out.release()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ground truth vs predicted gaze heatmaps as a video.")
    # .pt dataset flow (takes priority if provided)
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to .pt dataset (contains observations and optionally gaze_image)')
    # Legacy CSV flow
    parser.add_argument('--trajectories_dir', default="data/seaquest/trajectories")
    parser.add_argument('--labels_csv', default="data/seaquest/train_data_16_traj.csv")
    parser.add_argument('--gaze_masks', default="data/seaquest/gaze_masks.pt",
                        help='Ground truth gaze masks .pt (only used in CSV flow)')
    # Shared
    parser.add_argument('--game_name', default="seaquest")
    parser.add_argument('--model_weights', default="seaquest_gaze_predictor_2.pth")
    parser.add_argument('--output', default="gaze_comparison_2.mp4")
    parser.add_argument('--num_frames', type=int, default=3000)
    parser.add_argument('--multiplier', type=float, default=2.0)
    args = parser.parse_args()
    make_video(args)
