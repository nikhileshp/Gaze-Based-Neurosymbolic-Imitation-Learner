"""
scripts/data_utils.py
=====================
Central location for all Dataset classes and data-loading utilities.

Classes
-------
PtDataset
    PyTorch Dataset over the .pt file produced by convert_trajectories_to_pt.py.
    Returns (logic_state, action, gaze_image) tuples for imitation-learning training.

ExpertDataset
    Legacy CSV/pkl-based dataset for train_il.py.  Wraps OCAtari object detection
    and loads data from a CSV trajectory file.

Functions
---------
load_gaze_predictor_data(pt_path, frame_stack, normalize_obs, device)
    Loads the .pt file and returns (imgs_nhwc, gaze_masks, valid_indices) ready
    for Human_Gaze_Predictor.train_model().
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Action mapping (shared constant used in both Dataset classes)
# ---------------------------------------------------------------------------
PRIMITIVE_ACTION_MAP = {
    'noop': 0,
    'fire': 1,
    'up':   2,
    'right':3,
    'left': 4,
    'down': 5,
}

CSV_FILE       = "data/seaquest/train.csv"
BASE_IMAGE_DIR = "data/seaquest/trajectories"


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
    """

    def __init__(self, pt_path: str, use_gaze: bool = False, num_episodes: int = None, sort_by: str = None):
        print(f"Loading .pt dataset from {pt_path} ...")
        data = torch.load(pt_path, map_location='cpu', weights_only=False)

        logic    = data['logic_state']             # (N, N_OBJ, F)
        actions  = data['actions']                 # (N,)
        gaze     = data.get('gaze_image', None)    # (N, 84, 84) or None
        ep_nums  = data.get('episode_number', None) # (N,)
        rewards  = data.get('episode-rewards', None) # (N,)

        # Convert to tensors
        if not isinstance(logic, torch.Tensor): logic = torch.from_numpy(logic)
        if not isinstance(actions, torch.Tensor): actions = torch.from_numpy(actions)
        if ep_nums is not None and not isinstance(ep_nums, torch.Tensor): ep_nums = torch.from_numpy(ep_nums)
        if rewards is not None and not isinstance(rewards, torch.Tensor): rewards = torch.from_numpy(rewards)
        
        self.logic = logic.float()
        self.actions = actions.long()
        self.ep_nums = ep_nums.long() if ep_nums is not None else None
        self.rewards = rewards.float() if rewards is not None else None

        if use_gaze and gaze is not None:
            if not isinstance(gaze, torch.Tensor): gaze = torch.from_numpy(gaze)
            self.gaze = gaze.float()
        else:
            self.gaze = torch.zeros(len(self.logic), 84, 84)

        # ─── Advanced Filtering and Sorting ──────────────────────────────────
        if self.ep_nums is not None:
            unique_episodes = torch.unique(self.ep_nums)
            ep_metrics = []
            
            for ep_id in unique_episodes:
                mask = (self.ep_nums == ep_id)
                length = mask.sum().item()
                total_rew = self.rewards[mask].sum().item() if self.rewards is not None else 0.0
                ep_metrics.append({
                    'id': ep_id.item(),
                    'length': length,
                    'total_reward': total_rew,
                    'reward_per_step': total_rew / max(length, 1)
                })
            
            # Sort episodes if requested
            if sort_by == 'length':
                ep_metrics.sort(key=lambda x: x['length'], reverse=True)
            elif sort_by == 'reward_per_step':
                ep_metrics.sort(key=lambda x: x['reward_per_step'], reverse=True)
            
            # Select top N episodes
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

        print(f"  PtDataset: {len(self.logic)} samples | "
              f"logic {tuple(self.logic.shape[1:])} | "
              f"episodes: {len(torch.unique(self.ep_nums)) if self.ep_nums is not None else 'N/A'}")

    def __len__(self):
        return len(self.logic)

    def __getitem__(self, idx):
        return self.logic[idx], self.actions[idx], self.gaze[idx]


# ===========================================================================
# ExpertDataset — legacy CSV/pkl flow
# ===========================================================================

class ExpertDataset(Dataset):
    def __init__(self, env_name, agent_prednames, data_path=None, nudge_env=None,
                 limit=None, use_gazemap=False, trajectory=None):

        self.env_name        = env_name
        self.agent_prednames = agent_prednames
        self.nudge_env       = nudge_env
        self.data            = None
        self.df              = None
        self.use_gazemap     = False
        self.gaze_masks      = None

        from ocatari.core import OCAtari
        game_name = env_name.capitalize()
        print(f"Initializing OCAtari for {game_name}...")
        self.oc = OCAtari(game_name, mode="vision", render_mode="rgb_array")

        if data_path and os.path.exists(data_path):
            if data_path.endswith('.pkl'):
                print(f"Loading pre-computed data from {data_path}...")
                self.precomputed_data = torch.load(data_path)
                self.data             = self.precomputed_data['data']
                self.atom_names       = self.precomputed_data['atom_names']
                self.df               = None
                print(f"Loaded {len(self.data)} samples with {len(self.atom_names)} atoms.")
            else:
                print(f"Loading data from {data_path}...")
                self.df = pd.read_csv(data_path)
        else:
            print(f"Data path {data_path} not found. Using default CSV: {CSV_FILE}")
            self.df = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else None

        self.use_gazemap = use_gazemap
        if self.use_gazemap:
            mask_path = 'data/seaquest/gaze_masks.pt'
            if os.path.exists(mask_path):
                print(f"Loading gaze masks from {mask_path}...")
                self.gaze_masks = torch.load(mask_path)
                print(f"Loaded gaze masks: {self.gaze_masks.shape}")
            else:
                print(f"Warning: Gaze masks not found at {mask_path}.")
                self.use_gazemap = False

        if self.df is not None and limit:
            self.df = self.df.head(limit)

        if self.df is not None:
            self.df = self.df[self.df['action'] <= 5]
            if trajectory is not None:
                self.df = self.df[self.df['trajectory_number'] == trajectory]
            print(f"Filtered to {len(self.df)} samples")

        if self.data is not None and isinstance(self.data, list) and len(self.data) > 0 and 'action' in self.data[0]:
            if self.use_gazemap and self.gaze_masks is not None:
                for i, item in enumerate(self.data):
                    if isinstance(item, dict):
                        item['original_index'] = i
            self.data = [d for d in self.data if d.get('action', 999) <= 5]
            if trajectory is not None:
                self.data = [d for d in self.data if d.get('trajectory_number') == trajectory]
            if limit:
                self.data = self.data[:limit]

        if self.df is None and self.data is None:
            obs              = nudge_env.reset()
            logic_state      = obs[0]
            logic_state_tensor = torch.tensor(logic_state, dtype=torch.float32)
            self.state_shape = logic_state_tensor.shape
            self.data        = []
            for _ in range(100):
                state  = torch.rand(self.state_shape)
                action = torch.randint(0, len(agent_prednames), (1,)).item()
                gaze   = torch.rand(2) * 160
                self.data.append((state, action, gaze))
        elif self.df is None and self.data is not None:
            pass
        else:
            self.data = None

    def __len__(self):
        if self.df is not None:
            return len(self.df)
        return len(self.data)

    def __getitem__(self, idx):
        if self.df is None:
            item = self.data[idx]
            if isinstance(item, dict):
                atoms  = torch.tensor(item['atoms'], dtype=torch.float32)
                action = torch.tensor(item['action'], dtype=torch.long)
                gaze   = torch.tensor(item.get('gaze', [0.0, 0.0]), dtype=torch.float32)
                if self.use_gazemap and self.gaze_masks is not None:
                    original_idx = item.get('original_index', -1)
                    if 0 <= original_idx < len(self.gaze_masks):
                        return atoms, action, self.gaze_masks[original_idx]
                return atoms, action, gaze
            else:
                state, action, gaze = item
                return (torch.tensor(state, dtype=torch.float32),
                        torch.tensor(action, dtype=torch.long), gaze)

        row       = self.df.iloc[idx]
        image_array = self._load_image(row)

        if hasattr(self.oc, "_env") and hasattr(self.oc._env, "unwrapped") \
                and hasattr(self.oc._env.unwrapped, "ale"):
            real_ale = self.oc._env.unwrapped.ale

            class MockALE:
                def __getattr__(self, name):
                    return getattr(real_ale, name)
                def getScreenRGB(self, *args):
                    return image_array

            self.oc._env.unwrapped.ale = MockALE()
            try:
                self.oc.detect_objects()
            except ValueError:
                pass
            finally:
                self.oc._env.unwrapped.ale = real_ale
        else:
            try:
                self.oc.detect_objects(image_array)
            except (TypeError, ValueError):
                pass

        objects     = self.oc.objects
        logic_state, _ = self.nudge_env.convert_state(objects)
        action_idx  = int(row['action'])

        gaze_center = torch.zeros(2, dtype=torch.float32)
        if 'gaze_positions' in row and pd.notna(row['gaze_positions']):
            try:
                gaze_vals = [float(x) for x in str(row['gaze_positions']).split(',')]
                if gaze_vals and len(gaze_vals) % 2 == 0:
                    mean_gaze   = np.mean(np.array(gaze_vals).reshape(-1, 2), axis=0)
                    gaze_center = torch.tensor(mean_gaze, dtype=torch.float32)
            except ValueError:
                pass

        if self.use_gazemap and self.gaze_masks is not None:
            original_idx = row.name
            if original_idx < len(self.gaze_masks):
                return (torch.tensor(logic_state, dtype=torch.float32),
                        torch.tensor(action_idx, dtype=torch.long),
                        self.gaze_masks[original_idx])
            return (torch.tensor(logic_state, dtype=torch.float32),
                    torch.tensor(action_idx, dtype=torch.long),
                    torch.zeros(84, 84))

        return (torch.tensor(logic_state, dtype=torch.float32),
                torch.tensor(action_idx, dtype=torch.long),
                gaze_center)

    # ------------------------------------------------------------------
    def _load_image(self, row):
        traj_folder_map = {}
        for folder in os.listdir(BASE_IMAGE_DIR):
            folder_path = os.path.join(BASE_IMAGE_DIR, folder)
            if not os.path.isdir(folder_path):
                continue
            parts = folder.split('_')
            if len(parts) >= 3:
                traj_folder_map[parts[1] + "_" + parts[2]] = folder

        parts = row['frame_id'].split('_')
        key   = parts[0] + "_" + parts[1]
        traj_folder = traj_folder_map.get(key, '').replace('.txt', '')
        img_name    = parts[0] + "_" + parts[1] + "_" + parts[2] + ".png"
        img_path    = os.path.join(BASE_IMAGE_DIR, traj_folder, img_name)

        try:
            return np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return np.zeros((210, 160, 3), dtype=np.uint8)


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
    imgs_nhwc    : np.ndarray  (N, H, W, frame_stack)  float32
    gaze_masks   : torch.Tensor  (T, H, W)             float32
    valid_indices: torch.Tensor  (N,)                  long
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
