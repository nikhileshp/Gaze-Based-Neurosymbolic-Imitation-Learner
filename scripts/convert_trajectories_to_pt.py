"""
convert_trajectories_to_pt.py

Converts trajectory session folders (PNG frames + txt/csv gaze+action files) into a .pt dataset:

  observations:      (N, 84, 84)   uint8   - grayscale frames
  gaze_information:  (N, 3)        float64 - [x_norm, y_norm, global_step_id]
  gaze_image:        (N, 84, 84)   float32 - Gaussian heatmap centered at gaze point
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
N_OBJECTS = 47
N_FEATURES = 5
GAZE_IMG_SIZE = 84
GAZE_SIGMA = 5.0  # Gaussian sigma in 84x84 pixel space


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
}

def extract_logic_state(tracked_objects):
    """
    Convert list of TrackedObjects → (N_OBJECTS, N_FEATURES) int32 tensor.
    Features: [present, x_or_width, y, orientation, type_id]
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
            state[idx] = [1, int(obj.w), int(obj.y), 0, type_id]
        else:
            orient = 0
            if hasattr(obj, 'orientation') and obj.orientation is not None:
                orient = obj.orientation.value if hasattr(obj.orientation, 'value') else int(obj.orientation)
            cx = int(obj.x + obj.w / 2)
            cy = int(obj.y + obj.h / 2)
            state[idx] = [1, cx, cy, orient, type_id]

        obj_count[cat] += 1

    return state


# ─── Gaze Heatmap ────────────────────────────────────────────────────────────

def make_gaze_heatmap(x_norm, y_norm, size=GAZE_IMG_SIZE, sigma=GAZE_SIGMA):
    """Create a 2D Gaussian heatmap (size×size, float32) centered at normalized gaze."""
    cx = x_norm * size
    cy = y_norm * size
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    heatmap = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    total = heatmap.sum()
    return (heatmap / total).astype(np.float32) if total > 0 else heatmap


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

            pts = [(raw[i] / IMG_W, raw[i+1] / IMG_H) for i in range(0, len(raw)-1, 2)]
            if pts:
                x_n = max(0.0, min(1.0, sum(p[0] for p in pts) / len(pts)))
                y_n = max(0.0, min(1.0, sum(p[1] for p in pts) / len(pts)))
            else:
                x_n, y_n = 0.5, 0.5

            result[fid] = (x_n, y_n, action, reward, terminated)
    return result


# ─── Episode Processing ──────────────────────────────────────────────────────

def process_episode(ep_folder, global_step_offset, ep_number, oc):
    """
    Process one episode folder.
    Returns tuple of lists: (obs, gaze_info, gaze_imgs, logic_states, ep_nums,
                              actions, rewards, terms, truncs)
    """
    gaze_files = glob.glob(os.path.join(ep_folder, '*.txt')) + \
                 glob.glob(os.path.join(ep_folder, '*.csv'))
    gaze_map = parse_gaze_file(gaze_files[0]) if gaze_files else {}
    if gaze_files:
        print(f"  Gaze data: {len(gaze_map)} frames from {os.path.basename(gaze_files[0])}")
    else:
        print(f"  Warning: no gaze/action file in {ep_folder}")

    png_files = sorted(
        glob.glob(os.path.join(ep_folder, '*.png')),
        key=lambda p: int(os.path.basename(p).rsplit('_', 1)[-1].split('.')[0])
    )
    if not png_files:
        print(f"  Warning: no PNG files in {ep_folder}. Skipping.")
        return [], [], [], [], [], [], [], [], []

    tracker = ObjectTracker(match_dist_threshold=40)

    obs_l, gaze_l, gaze_img_l, logic_l, epnum_l = [], [], [], [], []
    act_l, rew_l, term_l, trunc_l = [], [], [], []

    for png_path in tqdm(png_files, desc=f"  ep{ep_number}", leave=False):
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
            detect_objects_vision(oc.objects, rgb, "Seaquest", hud=False)
            valid = [o for o in oc.objects if o.category != 'NoObject' and o.w > 0]
            active = tracker.update(valid)
        except Exception:
            active = tracker.tracked_objects
        logic_l.append(extract_logic_state(active))

        # 3. Gaze info + heatmap
        if fid in gaze_map:
            x_n, y_n, action, reward, terminated = gaze_map[fid]
        else:
            x_n, y_n, action, reward, terminated = 0.5, 0.5, 0, 0.0, False

        global_step = global_step_offset + len(obs_l) - 1
        gaze_l.append([x_n, y_n, float(global_step)])
        gaze_img_l.append(make_gaze_heatmap(x_n, y_n))

        act_l.append(action)
        rew_l.append(reward)
        term_l.append(terminated)
        trunc_l.append(False)
        epnum_l.append(ep_number)

    if term_l:
        term_l[-1] = True  # always mark last frame as done

    return obs_l, gaze_l, gaze_img_l, logic_l, epnum_l, act_l, rew_l, term_l, trunc_l


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert trajectory PNG+txt folders to .pt dataset.")
    parser.add_argument('--traj_dir', type=str, default='../data/seaquest/trajectories')
    parser.add_argument('--output', type=str, default='../data/seaquest/converted_trajectories.pt')
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--gaze_sigma', type=float, default=GAZE_SIGMA,
                        help='Gaussian sigma for gaze heatmap (in 84px space)')
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
    if args.max_episodes:
        ep_folders = ep_folders[:args.max_episodes]

    print(f"Found {len(ep_folders)} episode(s). Initializing Ocatari...\n")
    oc = OCAtari("Seaquest", mode="vision", render_mode="rgb_array")

    all_obs, all_gaze, all_gaze_img, all_logic = [], [], [], []
    all_epnum, all_act, all_rew, all_term, all_trunc = [], [], [], [], []
    steps_per_ep = []
    global_step = 0

    for ep_idx, ep_folder in enumerate(ep_folders):
        print(f"\nEpisode {ep_idx}: {os.path.basename(ep_folder)}")
        out = process_episode(ep_folder, global_step, ep_idx, oc)
        obs, gaze, gaze_img, logic, epnum, act, rew, term, trunc = out
        if not obs:
            continue

        all_obs.extend(obs);       all_gaze.extend(gaze)
        all_gaze_img.extend(gaze_img); all_logic.extend(logic)
        all_epnum.extend(epnum);   all_act.extend(act)
        all_rew.extend(rew);       all_term.extend(term)
        all_trunc.extend(trunc)
        steps_per_ep.append(len(obs))
        global_step += len(obs)
        print(f"  → {len(obs)} frames")

    if not all_obs:
        print("Error: no frames collected."); sys.exit(1)

    N = len(all_obs)
    dataset = {
        'observations':    np.array(all_obs,      dtype=np.uint8),    # (N, 84, 84)
        'gaze_information':np.array(all_gaze,     dtype=np.float64),  # (N, 3)
        'gaze_image':      np.array(all_gaze_img, dtype=np.float32),  # (N, 84, 84)
        'logic_state':     np.array(all_logic,    dtype=np.int32),    # (N, 47, 5)
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
    torch.save(dataset, out_path)
    print("Done!")


if __name__ == '__main__':
    main()
