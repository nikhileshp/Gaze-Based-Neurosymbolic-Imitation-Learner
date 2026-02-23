"""
convert_trajectories_to_pt.py

Converts trajectory session folders (PNG frames + txt/csv gaze+action files) into a .pt dataset:

  observations:      (N, 84, 84)   uint8   - grayscale frames
  gaze_information:  (N, 3)        float64 - [x_norm, y_norm, global_step_id]
  gaze_image:        (N, 84, 84)   float32 - temporally-aggregated gaze saliency map (±k_window frames)
  logic_state:       (N, 47, 5)    int32   - symbolic object state from Ocatari vision
  episode_number:    (N,)          int32   - which episode each frame belongs to
  actions:           (N,)          int32   - action index per frame
  episode-rewards:   (N,)          float64 - reward per frame
  terminateds:       (N,)          bool    - True when unclipped_reward < 0 or last frame
  truncateds:        (N,)          bool    - always False
  steps:             (E,)          int32   - number of steps per episode

Usage:
  cd scripts
  python convert_trajectories_to_pt.py
  python convert_trajectories_to_pt.py --traj_dir ../data/seaquest/trajectories --output ../data/seaquest/my_dataset.pt
"""

import argparse
import os
import glob
import sys
import cv2
import math
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocatari.vision.extract_vision_info import detect_objects_vision
from ocatari.core import OCAtari
from ocatari.ram.seaquest import MAX_NB_OBJECTS as MAX_ESSENTIAL_OBJECTS

IMG_W = 160   # Seaquest game width (pixels)
IMG_H = 210   # Seaquest game height (pixels)
N_FEATURES = 7
# Compute N_OBJECTS dynamically from MAX_ESSENTIAL_OBJECTS (mirrors the EnemyMissile
# override applied in extract_logic_state so the count is always in sync with seaquest.py)
_obj_counts = dict(MAX_ESSENTIAL_OBJECTS)
_obj_counts['EnemyMissile'] = 8   # same override used in extract_logic_state
N_OBJECTS = sum(_obj_counts.values())
GAZE_IMG_SIZE = 84
GAZE_SIGMA = 10.0          # Base Gaussian sigma (pixels in 84×84 space)
GAZE_K_WINDOW = 10         # Symmetric temporal window (±k frames)
GAZE_COEF = 0.7           # Alpha: weight decay per frame of distance
GAZE_VARIANCE_EXP = 0.99  # Beta: sigma shrinks as distance decreases (closer = tighter)


# ─── Object Tracker (from extract_gaze_goals.py) ───────────────────────────

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class TrackedObject:
    def __init__(self, obj, obj_id):
        self.obj = obj
        self.id = obj_id
        self.missing_frames = 0

    @property
    def center(self):
        return (self.obj.x + self.obj.w / 2, self.obj.y + self.obj.h / 2)

    def update(self, new_obj):
        self.obj = new_obj
        self.missing_frames = 0

class ObjectTracker:
    def __init__(self, max_missing_frames=5, match_dist_threshold=30):
        self.next_id = 0
        self.tracked_objects = []
        self.max_missing_frames = max_missing_frames
        self.match_dist_threshold = match_dist_threshold

    def update(self, detections):
        current_tracked = []
        unmatched = list(detections)
        for track in self.tracked_objects:
            best_i, min_d = -1, float('inf')
            for i, det in enumerate(unmatched):
                if det.category != track.obj.category:
                    continue
                d = get_distance((det.x + det.w/2, det.y + det.h/2), track.center)
                if d < min_d and d < self.match_dist_threshold:
                    min_d, best_i = d, i
            if best_i >= 0:
                track.update(unmatched.pop(best_i))
                current_tracked.append(track)
            else:
                track.missing_frames += 1
                if track.missing_frames < self.max_missing_frames:
                    current_tracked.append(track)
        for det in unmatched:
            current_tracked.append(TrackedObject(det, self.next_id))
            self.next_id += 1
        self.tracked_objects = current_tracked
        return self.tracked_objects


# ─── Logic State Extraction ─────────────────────────────────────────────────

TYPE_MAP = {
    'Shark': 0, 'Submarine': 0, 'SurfaceSubmarine': 0,
    'Diver': 1, 'CollectedDiver': 6,
    'OxygenBar': 2,
    'Player': 3,
    'EnemyMissile': 5, 'PlayerMissile': 5,
    'Surface': 7
}

# Known canonical sprite sizes for objects that flicker in detected size.
# The center is reliable; width/height are not. These are used to stabilize the logic state.
CANONICAL_SIZES = {
    'Shark':  (8, 7),   # width=8, height=7
    'Diver':  (8, 10),   # width=8, height=10
}

def extract_logic_state(tracked_objects):
    """
    Convert list of TrackedObjects → (N_OBJECTS, N_FEATURES) int32 tensor.
    Features: [present, x_or_width, y, width, height, orientation, type_id]
    """
    state = np.zeros((N_OBJECTS, N_FEATURES), dtype=np.int32)
    relevant = MAX_ESSENTIAL_OBJECTS.copy()
    if 'EnemyMissile' in relevant:
        relevant['EnemyMissile'] = 8

    offsets, obj_count = {}, {}
    off = 0
    for cat, max_c in relevant.items():
        offsets[cat] = off
        obj_count[cat] = 0
        off += max_c

    for tr in tracked_objects:
        obj = tr.obj
        cat = obj.category
        if cat not in relevant or obj_count.get(cat, 0) >= relevant[cat]:
            continue
        idx = offsets[cat] + obj_count[cat]
        type_id = TYPE_MAP.get(cat, 0)

        if cat == 'OxygenBar':
            state[idx] = [1, int(obj.x), int(obj.y), int(obj.w), int(obj.h), 0, type_id]
        else:
            orient = 0
            if hasattr(obj, 'orientation') and obj.orientation is not None:
                orient = obj.orientation.value if hasattr(obj.orientation, 'value') else int(obj.orientation)
            # Use canonical size if available to prevent flickering
            if cat in CANONICAL_SIZES:
                w, h = CANONICAL_SIZES[cat]
            else:
                w = getattr(obj, "w", 0)
                h = getattr(obj, "h", 0)
            cx, cy = getattr(obj, 'center', (int(obj.x + obj.w / 2), int(obj.y + obj.h / 2)))
            state[idx] = [1, cx, cy, int(w), int(h), orient, type_id]

        obj_count[cat] += 1

    return state


# ─── Gaze Heatmap (temporally aggregated) ───────────────────────────────────

class GazeToMask:
    """
    Pre-builds a lookup of 2D Gaussian masks (one per distance index).
    Each mask is stored at 2×N resolution so it can be cropped to any
    (x_norm, y_norm) center without recomputing the Gaussian.
    Matches the implementation in generate_full_gaze_tensor.py.
    """
    def __init__(self, N=GAZE_IMG_SIZE, sigmas=None, coeficients=None):
        self.N = N
        self.sigmas = sigmas or [GAZE_SIGMA]
        self.coeficients = coeficients or [1.0]
        assert len(self.sigmas) == len(self.coeficients)
        self.masks = self._initialize()  # (n_slots, 2N, 2N) torch.float32

    def _initialize(self):
        N = self.N
        maps = []
        for sigma, coef in zip(self.sigmas, self.coeficients):
            x = torch.arange(2 * N, dtype=torch.float32)
            y = torch.arange(2 * N, dtype=torch.float32)
            Y, X = torch.meshgrid(y, x, indexing='ij')
            g = torch.exp(-((X - (N - 1))**2 + (Y - (N - 1))**2) / (2 * sigma**2))
            g = g / g.max()
            maps.append(coef * g)
        return torch.stack(maps, 0)

    def find_suitable_map(self, index, mean_x, mean_y):
        Nx2 = self.N * 2
        start_x = max(0, min(int((1 - mean_x) * self.N), self.N))
        start_y = max(0, min(int((1 - mean_y) * self.N), self.N))
        return self.masks[index][start_y:start_y + self.N, start_x:start_x + self.N]


def build_episode_gaze_images(
    episode_gaze_pts,           # list[list[(x_norm, y_norm)]], one entry per frame
    k_window=GAZE_K_WINDOW,
    sigma=GAZE_SIGMA,
    coef=GAZE_COEF,
    variance_exp=GAZE_VARIANCE_EXP,
):
    """
    Build temporally-aggregated gaze saliency maps for an entire episode.
    For each frame i, sums weighted Gaussians from frames in [i-k, i+k],
    where distance d uses sigma/variance_exp^d and weight coef^d.
    Returns list of (84, 84) float32 numpy arrays.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_slots = k_window + 1
    saliency_sigmas = [sigma / (variance_exp ** d) for d in range(n_slots)]
    coeficients     = [coef ** d                   for d in range(n_slots)]
    MASK = GazeToMask(GAZE_IMG_SIZE, saliency_sigmas, coeficients)
    MASK.masks = MASK.masks.to(device)   # move lookup table to GPU once

    T = len(episode_gaze_pts)
    result = []
    for i in tqdm(range(T), desc="  gaze post-pass", leave=False):
        accumulated = torch.zeros(GAZE_IMG_SIZE, GAZE_IMG_SIZE, device=device)
        for j in range(max(0, i - k_window), min(T, i + k_window + 1)):
            dist = abs(j - i)
            for (gx, gy) in episode_gaze_pts[j]:
                accumulated += MASK.find_suitable_map(dist, gx, gy)
        if accumulated.max() > 0:
            accumulated = accumulated / accumulated.max()
        result.append(accumulated.cpu().numpy().astype(np.float32))
    return result


# ─── Frame Preprocessing ─────────────────────────────────────────────────────

def preprocess_frame(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)  # uint8


# ─── Gaze File Parsing ───────────────────────────────────────────────────────

def parse_gaze_file(filepath):
    """
    Returns: { frame_int_id -> (x_norm, y_norm, action, reward, terminated) }
    Format: frame_id, episode_id, score, duration, unclipped_reward, action, gaze_x0, gaze_y0, ...
    """
    result = {}
    with open(filepath, 'r') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            try:
                fid = int(parts[0].split('_')[-1])
            except ValueError:
                continue
            try:
                reward = float(parts[4])
            except (ValueError, IndexError):
                reward = 0.0
            try:
                action = int(parts[5])
            except (ValueError, IndexError):
                action = 0
            terminated = reward < 0.0

            raw = []
            for v in parts[6:]:
                v = v.strip()
                if v and v.lower() not in ('null', 'nan', ''):
                    try:
                        raw.append(float(v))
                    except ValueError:
                        pass

            pts = [
                (max(0.0, min(1.0, raw[i] / IMG_W)),
                 max(0.0, min(1.0, raw[i+1] / IMG_H)))
                for i in range(0, len(raw) - 1, 2)
            ]
            if pts:
                x_n = sum(p[0] for p in pts) / len(pts)
                y_n = sum(p[1] for p in pts) / len(pts)
            else:
                pts = [(0.5, 0.5)]
                x_n, y_n = 0.5, 0.5

            result[fid] = (pts, x_n, y_n, action, reward, terminated)
    return result


# ─── Episode Processing ──────────────────────────────────────────────────────

def process_episode(ep_folder, traj_dir, global_step_offset, ep_number_start, oc,
                    gaze_sigma=GAZE_SIGMA, gaze_k_window=GAZE_K_WINDOW):
    """
    Process one trajectory folder, which may contain multiple episodes
    (a new episode begins after any frame with negative reward).
    Returns:
        obs, gaze_info, gaze_imgs, logic_states, ep_nums,
        actions, rewards, terms, truncs  — flat lists over all frames
        sub_ep_steps                     — list of frame counts per sub-episode
    """
    folder_name = os.path.basename(ep_folder)
    # Gaze files live in traj_dir, named after the session folder
    gaze_files = glob.glob(os.path.join(traj_dir, folder_name + '.txt')) + \
                 glob.glob(os.path.join(traj_dir, folder_name + '.csv'))
    gaze_map = parse_gaze_file(gaze_files[0]) if gaze_files else {}
    if gaze_files:
        print(f"  Gaze data: {len(gaze_map)} frames from {os.path.basename(gaze_files[0])}")
    else:
        print(f"  Warning: no gaze/action file for {folder_name} in {traj_dir}")

    png_files = sorted(
        glob.glob(os.path.join(ep_folder, '*.png')),
        key=lambda p: int(os.path.basename(p).rsplit('_', 1)[-1].split('.')[0])
    )
    if not png_files:
        print(f"  Warning: no PNG files in {ep_folder}. Skipping.")
        return [], [], [], [], [], [], [], [], [], []

    tracker = ObjectTracker(match_dist_threshold=40)

    obs_l, gaze_l, logic_l, epnum_l = [], [], [], []
    act_l, rew_l, term_l, trunc_l = [], [], [], []
    raw_pts_l = []   # list[list[(x_norm, y_norm)]] — for windowed post-pass

    sub_ep_steps = []     # frame count per sub-episode within this folder
    current_ep_start = 0  # frame index where the current sub-episode started
    current_ep_num = ep_number_start
    ep_prev_px, ep_prev_py = 0.0, 0.0  # track last known player position

    for png_path in tqdm(png_files, desc=f"  folder{ep_number_start}", leave=False):
        try:
            fid = int(os.path.basename(png_path).rsplit('_', 1)[-1].split('.')[0])
        except (ValueError, IndexError):
            fid = len(obs_l)

        bgr = cv2.imread(png_path)
        if bgr is None:
            continue

        # 1. Observations (84x84 grayscale)
        obs_l.append(preprocess_frame(bgr))

        # 2. Logic state via Ocatari vision
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            # hud=False is much faster. We detect death via player teleportation instead.
            detect_objects_vision(oc.objects, rgb, "Seaquest", hud=False)
            valid = [o for o in oc.objects if o.category != 'NoObject' and o.w > 0]
            active = tracker.update(valid)
        except Exception:
            active = tracker.tracked_objects
        logic_l.append(extract_logic_state(active))

        # 3. Gaze info — store raw points; heatmap built after loop
        if fid in gaze_map:
            pts, x_n, y_n, action, reward, _ = gaze_map[fid]
        else:
            pts, x_n, y_n, action, reward, _ = [(0.5, 0.5)], 0.5, 0.5, 0, 0.0

        global_step = global_step_offset + len(obs_l) - 1
        gaze_l.append([x_n, y_n, float(global_step)])   # mean coord for GABRIL compatibility
        raw_pts_l.append(pts)

        act_l.append(action)
        rew_l.append(reward)
        

        # Detect death: if Player teleports > 30 pixels in a single frame (respawn)
        player_obj = next((tr.obj for tr in active if tr.obj.category == 'Player' and tr.obj.y > 30), None)
        if player_obj is not None:
            curr_px, curr_py = player_obj.x + player_obj.w / 2.0, player_obj.y + player_obj.h / 2.0
            if len(obs_l) - current_ep_start == 1:
                ep_prev_px, ep_prev_py = curr_px, curr_py
                terminated = False
            else:
                dist = ((curr_px - ep_prev_px)**2 + (curr_py - ep_prev_py)**2)**0.5
                terminated = dist > 30.0
                ep_prev_px, ep_prev_py = curr_px, curr_py
        else:
            terminated = False
            
        term_l.append(terminated)
        trunc_l.append(False)
        epnum_l.append(current_ep_num)

        # End of sub-episode: player died
        if terminated:
            sub_ep_steps.append(len(obs_l) - current_ep_start)
            current_ep_start = len(obs_l)
            current_ep_num += 1
            tracker = ObjectTracker(match_dist_threshold=40)  # reset tracker on death
            # Note: ep_prev_px will be reset correctly at start of next sub-ep

    # Close any open sub-episode at end of folder
    if current_ep_start < len(obs_l):
        term_l[-1] = True  # mark last frame as terminal
        sub_ep_steps.append(len(obs_l) - current_ep_start)

    # Build gaze images per sub-episode so the ±k_window never crosses a death boundary
    gaze_img_l = []
    ptr = 0
    for ep_len in sub_ep_steps:
        chunk = raw_pts_l[ptr:ptr + ep_len]
        gaze_img_l.extend(build_episode_gaze_images(chunk, k_window=gaze_k_window, sigma=gaze_sigma))
        ptr += ep_len

    return obs_l, gaze_l, gaze_img_l, logic_l, epnum_l, act_l, rew_l, term_l, trunc_l, sub_ep_steps


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert trajectory PNG+txt folders to .pt dataset.")
    parser.add_argument('--traj_dir', type=str, default='data/seaquest/trajectories')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .pt path. Auto-generated from params if omitted.')
    parser.add_argument('--max_folders', type=int, default=None)
    parser.add_argument('--gaze_sigma', type=float, default=GAZE_SIGMA,
                        help='Gaussian sigma for gaze heatmap (in 84px space)')
    parser.add_argument('--sliding_window', type=int, default=GAZE_K_WINDOW,
                        help='Sliding window size for gaze heatmap')
    args = parser.parse_args()

    traj_dir = os.path.abspath(args.traj_dir)
    if not os.path.isdir(traj_dir):
        print(f"Error: {traj_dir} not found."); sys.exit(1)

    ep_folders = sorted([
        os.path.join(traj_dir, d) for d in os.listdir(traj_dir)
        if os.path.isdir(os.path.join(traj_dir, d))
    ])
    if not ep_folders:
        print("Error: no episode subfolders found."); sys.exit(1)
    if args.max_folders:
        ep_folders = ep_folders[:args.max_folders]

    num_episodes = len(ep_folders)

    # Auto-generate output filename if not provided — filled in after processing
    # when we know the actual episode count

    print(f"Found {len(ep_folders)} folder(s). Initializing Ocatari...\n")
    oc = OCAtari("Seaquest", mode="vision", render_mode="rgb_array")

    all_obs, all_gaze, all_gaze_img, all_logic = [], [], [], []
    all_epnum, all_act, all_rew, all_term, all_trunc = [], [], [], [], []
    steps_per_ep = []
    global_step = 0

    # ep_number_start tracks the global episode index across folders
    ep_number_start = 0

    for folder_idx, ep_folder in enumerate(ep_folders):
        print(f"\nFolder {folder_idx}: {os.path.basename(ep_folder)}")
        out = process_episode(ep_folder, traj_dir, global_step, ep_number_start, oc,
                              gaze_sigma=args.gaze_sigma,
                              gaze_k_window=args.sliding_window)
        obs, gaze, gaze_img, logic, epnum, act, rew, term, trunc, sub_ep_steps = out
        if not obs:
            continue

        all_obs.extend(obs);       all_gaze.extend(gaze)
        all_gaze_img.extend(gaze_img); all_logic.extend(logic)
        all_epnum.extend(epnum);   all_act.extend(act)
        all_rew.extend(rew);       all_term.extend(term)
        all_trunc.extend(trunc)
        steps_per_ep.extend(sub_ep_steps)   # one entry per actual episode, not per folder
        ep_number_start += len(sub_ep_steps)
        global_step += len(obs)
        print(f"  → {len(obs)} frames, {len(sub_ep_steps)} episode(s)")

    if not all_obs:
        print("Error: no frames collected."); sys.exit(1)

    num_episodes = len(steps_per_ep)  # actual episode count (deaths = episode boundaries)

    # Auto-generate output filename now that we know actual episode count
    if args.output is None:
        sigma_str = f"{args.gaze_sigma:.1f}".replace('.', 'p')
        fname = (f"full_data_{num_episodes}_episodes"
                 f"_{sigma_str}_sigma"
                 f"_win_{args.sliding_window}"
                 f"_obj_{N_OBJECTS}.pt")
        args.output = os.path.join(os.path.dirname(traj_dir), fname)

    N = len(all_obs)
    dataset = {
        'observations':    np.array(all_obs,      dtype=np.uint8),    # (N, 84, 84)
        'gaze_information':np.array(all_gaze,     dtype=np.float64),  # (N, 3)
        'gaze_image':      np.array(all_gaze_img, dtype=np.float32),  # (N, 84, 84)
        'logic_state':     np.array(all_logic,    dtype=np.int32),    # (N, N_OBJECTS, 5)
        'episode_number':  np.array(all_epnum,    dtype=np.int32),    # (N,)
        'actions':         np.array(all_act,      dtype=np.int32),    # (N,)
        'episode-rewards': np.array(all_rew,      dtype=np.float64),  # (N,)
        'terminateds':     np.array(all_term,     dtype=bool),        # (N,)
        'truncateds':      np.array(all_trunc,    dtype=bool),        # (N,) all False
        'steps':           np.array(steps_per_ep, dtype=np.int32),    # (E,)
    }

    print("\nDataset summary:")
    for k, v in dataset.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"\nSaving → {out_path}")
    torch.save(dataset, out_path, _use_new_zipfile_serialization=True, pickle_protocol=4)
    print("Done!")
    print(f"  episodes={num_episodes}, gaze_sigma={args.gaze_sigma}, sliding_window={args.sliding_window}, n_objects={N_OBJECTS}")


if __name__ == '__main__':
    main()
