"""
core/utils/data_utils.py
========================
Dataset classes and data-loading utilities for the GRAIL project.

Classes
-------
PtDataset
    PyTorch Dataset over the .pt file produced by convert_trajectories_to_pt.py.
    Returns (logic_state, action, gaze_image, ep_num, step_idx) tuples.

Functions
---------
load_gaze_predictor_data(pt_path, frame_stack, normalize_obs, device)
    Loads the .pt file and returns (imgs_nhwc, gaze_masks, valid_indices)
    ready for Human_Gaze_Predictor.train_model().

load_fact_gaze_predictor_data(facts_pkl, gaze_pt, frame_stack, device)
    Loads pre-computed NSFR atom vectors and paired gaze masks for FactGazeNet.

Note: The legacy ExpertDataset (CSV/pkl flow) has been removed.
Use the .pt dataset flow exclusively via convert_trajectories_to_pt.py.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Action mapping (shared constant)
# ---------------------------------------------------------------------------
PRIMITIVE_ACTION_MAP = {
    'noop': 0,
    'fire': 1,
    'up':   2,
    'right':3,
    'left': 4,
    'down': 5,
}


# ===========================================================================
# PtDataset — .pt file → (logic_state, action, gaze)
# ===========================================================================

class PtDataset(Dataset):
    """
    Lightweight Dataset backed by the .pt file from convert_trajectories_to_pt.py.

    Each item:
        logic_state : (N_OBJ, N_FEATURES)  float32  symbolic state for one frame
        action      : ()                    long     ground-truth action index
        gaze        : (84, 84)              float32  gaze heatmap (zeros if no gaze)
        ep_num      : ()                    long     episode number (-1 if unknown)
        step_idx    : ()                    long     step within episode (-1 if unknown)
    """

    def __init__(self, pt_path: str, use_gaze: bool = False, num_episodes: int = None, sort_by: str = None):
        print(f"Loading .pt dataset from {pt_path} ...")
        data = torch.load(pt_path, map_location='cpu', weights_only=False)

        logic    = data['logic_state']              # (N, N_OBJ, F)
        actions  = data['actions']                  # (N,)
        gaze     = data.get('gaze_image', None)     # (N, 84, 84) or None
        ep_nums  = data.get('episode_number', None) # (N,)
        rewards  = data.get('episode-rewards', None) # (N,)

        # Convert to tensors
        if not isinstance(logic, torch.Tensor):   logic   = torch.from_numpy(logic)
        if not isinstance(actions, torch.Tensor): actions = torch.from_numpy(actions)
        if ep_nums is not None and not isinstance(ep_nums, torch.Tensor): ep_nums = torch.from_numpy(ep_nums)
        if rewards is not None and not isinstance(rewards, torch.Tensor): rewards = torch.from_numpy(rewards)

        self.logic   = logic.float()
        self.actions = actions.long()
        self.ep_nums = ep_nums.long() if ep_nums is not None else None
        self.rewards = rewards.float() if rewards is not None else None

        if use_gaze and gaze is not None:
            if not isinstance(gaze, torch.Tensor): gaze = torch.from_numpy(gaze)
            self.gaze = gaze.float()
        else:
            self.gaze = torch.zeros(len(self.logic), 84, 84)

        # ── Advanced Filtering and Sorting ───────────────────────────────────
        if self.ep_nums is not None:
            unique_episodes = torch.unique(self.ep_nums)
            ep_metrics = []

            for ep_id in unique_episodes:
                mask   = (self.ep_nums == ep_id)
                length = mask.sum().item()
                total_rew = self.rewards[mask].sum().item() if self.rewards is not None else 0.0
                ep_metrics.append({
                    'id': ep_id.item(),
                    'length': length,
                    'total_reward': total_rew,
                    'reward_per_step': total_rew / max(length, 1)
                })

            if sort_by == 'length':
                ep_metrics.sort(key=lambda x: x['length'], reverse=True)
            elif sort_by == 'reward_per_step':
                ep_metrics.sort(key=lambda x: x['reward_per_step'], reverse=True)

            if num_episodes is not None:
                ep_metrics = ep_metrics[:num_episodes]
                print(f"  Filtering to top {num_episodes} episodes by {sort_by or 'original order'}")

            selected_ids = [m['id'] for m in ep_metrics]
            mask = torch.tensor([e.item() in selected_ids for e in self.ep_nums])

            self.logic   = self.logic[mask]
            self.actions = self.actions[mask]
            self.gaze    = self.gaze[mask]
            self.ep_nums = self.ep_nums[mask]
            if self.rewards is not None:
                self.rewards = self.rewards[mask]

        # Filter to supported actions (0-5)
        mask = (self.actions <= 5)
        self.logic   = self.logic[mask]
        self.actions = self.actions[mask]
        self.gaze    = self.gaze[mask]
        if self.ep_nums is not None: self.ep_nums = self.ep_nums[mask]

        # Compute step index within each episode
        if self.ep_nums is not None:
            self.step_in_ep_idx = torch.zeros_like(self.ep_nums)
            for ep_id in torch.unique(self.ep_nums):
                ep_mask = (self.ep_nums == ep_id)
                self.step_in_ep_idx[ep_mask] = torch.arange(ep_mask.sum())
        else:
            self.step_in_ep_idx = None

        print(f"  PtDataset: {len(self.logic)} samples | "
              f"logic {tuple(self.logic.shape[1:])} | "
              f"episodes: {len(torch.unique(self.ep_nums)) if self.ep_nums is not None else 'N/A'}")

    def __len__(self):
        return len(self.logic)

    def __getitem__(self, idx):
        logic    = self.logic[idx]
        action   = self.actions[idx]
        gaze     = self.gaze[idx]
        ep_num   = self.ep_nums[idx] if self.ep_nums is not None else -1
        step_idx = self.step_in_ep_idx[idx] if self.step_in_ep_idx is not None else -1
        return logic, action, gaze, ep_num, step_idx


# ===========================================================================
# load_gaze_predictor_data — returns arrays for Human_Gaze_Predictor.train_model
# ===========================================================================

def load_gaze_predictor_data(
    pt_path:       str,
    frame_stack:   int  = 4,
    normalize_obs: bool = True,
    device:        str  = "cpu",
):
    """
    Load a .pt file and return arrays suitable for ``Human_Gaze_Predictor.train_model``.

    Returns
    -------
    imgs_nhwc    : np.ndarray   (N, H, W, frame_stack)  float32
    gaze_masks   : torch.Tensor (T, H, W)               float32
    valid_indices: torch.Tensor (N,)                    long
    """
    print(f"Loading gaze-predictor data from {pt_path} ...")
    data = torch.load(pt_path, map_location=device, weights_only=False)

    obs        = data["observations"]   # (T, H, W) uint8 numpy
    gaze_masks = data["gaze_image"]     # (T, H, W) float32

    T, H, W = obs.shape
    print(f"  Frames: {T}  |  Size: {H}x{W}  |  Stack: {frame_stack}")

    obs_f32 = obs.astype(np.float32)
    if normalize_obs:
        obs_f32 /= 255.0

    if not isinstance(gaze_masks, torch.Tensor):
        gaze_masks = torch.from_numpy(gaze_masks)

    k             = frame_stack
    N             = T - (k - 1)
    imgs_nhwc     = np.zeros((N, H, W, k), dtype=np.float32)
    valid_indices = torch.arange(k - 1, T, dtype=torch.long)

    for i in range(N):
        t = i + (k - 1)
        for j in range(k):
            imgs_nhwc[i, :, :, j] = obs_f32[t - (k - 1 - j)]

    print(f"  imgs_nhwc : {imgs_nhwc.shape}")
    print(f"  gaze_masks: {tuple(gaze_masks.shape)}")
    return imgs_nhwc, gaze_masks, valid_indices


# ===========================================================================
# load_fact_gaze_predictor_data — returns facts and gaze for FactGazeNet
# ===========================================================================

def load_fact_gaze_predictor_data(
    facts_pkl:   str,
    gaze_pt:     str,
    frame_stack: int = 4,
    device:      str = "cpu",
):
    """
    Load a .pkl file containing ``atom_probs`` and merge with a gaze masks .pt file.

    Returns
    -------
    facts_stacked : torch.Tensor (N, k * num_atoms)  float32
    gaze_masks    : torch.Tensor (T, H, W)            float32
    valid_indices : torch.Tensor (N,)                 long
    """
    import pickle

    print(f"Loading probabilistic facts from {facts_pkl} ...")
    with open(facts_pkl, 'rb') as f:
        data = pickle.load(f)

    atom_probs = data['atom_probs']  # (T, num_atoms)

    print(f"Loading gaze masks from {gaze_pt} ...")
    gaze_masks = torch.load(gaze_pt, map_location=device, weights_only=False)

    min_len    = min(len(atom_probs), len(gaze_masks))
    atom_probs = atom_probs[:min_len]
    gaze_masks = gaze_masks[:min_len]

    if not isinstance(atom_probs, torch.Tensor):
        atom_probs = torch.tensor(atom_probs, dtype=torch.float32)
    if not isinstance(gaze_masks, torch.Tensor):
        gaze_masks = torch.tensor(gaze_masks, dtype=torch.float32)

    T         = min_len
    k         = frame_stack
    N         = T - (k - 1)
    num_atoms = atom_probs.shape[1]

    facts_stacked = torch.zeros((N, k * num_atoms), dtype=torch.float32)
    valid_indices = torch.arange(k - 1, T, dtype=torch.long)

    for i in range(N):
        t = i + (k - 1)
        window = atom_probs[t - (k - 1): t + 1]  # (k, num_atoms)
        facts_stacked[i] = window.flatten()

    print(f"  facts_stacked : {facts_stacked.shape}")
    print(f"  gaze_masks    : {tuple(gaze_masks.shape)}")
    return facts_stacked, gaze_masks, valid_indices

