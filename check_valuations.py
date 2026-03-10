import torch
import numpy as np

# Load the NSFR valuations (unscaled)
nsfr_path = "trained_models/seaquest/nsfr_td_0.99/best_rules_0.01_lr_nll/valuations_best.pt"
nsfr_vals = torch.load(nsfr_path, map_location='cpu', weights_only=False)

# Load the GRAIL Normalized Visible Only valuations
grail_path = "trained_models/seaquest/grail_normalized_vis_only/best_rules_0.01_lr_nll/valuations_best.pt"
grail_vals = torch.load(grail_path, map_location='cpu', weights_only=False)

def check_episode(ep_id):
    nsfr_ep = nsfr_vals[ep_id]
    grail_ep = grail_vals[ep_id]
    
    print(f"\nEpisode {ep_id}:")
    print(f"Num frames: NSFR={len(nsfr_ep)}, GRAIL={len(grail_ep)}")
    
    if len(nsfr_ep) == 0 or len(grail_ep) == 0:
        return
        
    num_frames = min(len(nsfr_ep), len(grail_ep))
    num_diffs = 0
    num_scales = 0
    max_frames_to_check = 10
    frames_checked = 0
    
    for i in range(num_frames):
        nv = nsfr_ep[i]
        gv = grail_ep[i]
        
        diff = torch.abs(nv - gv)
        if torch.max(diff) > 1e-4:
            num_diffs += 1
            if frames_checked < max_frames_to_check:
                frames_checked += 1
                diff_indices = torch.where(diff > 1e-4)[0]
                print(f"  Frame {i} has diffs. Showing first few:")
                for idx in diff_indices[:5]:
                    idx = idx.item()
                    print(f"    Atom {idx}: NSFR={nv[idx]:.4f}, GRAIL={gv[idx]:.4f}, scaling={gv[idx]/nv[idx]:.4f}")
                    num_scales += 1
    
    print(f"Total frames with diffs: {num_diffs}/{num_frames}")
    
# Check a few episodes
episodes_to_check = list(nsfr_vals.keys())[:3]
for ep_id in episodes_to_check:
    check_episode(ep_id)
