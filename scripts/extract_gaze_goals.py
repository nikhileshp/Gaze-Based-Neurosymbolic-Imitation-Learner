import os
print("DEBUG: Executing updated extract_gaze_goals.py with LOGIC ID & DEEP SNAPSHOT & FRAME EXPORT & ACTIONS")
import glob
import pandas as pd
import numpy as np
import cv2
import json
import argparse
import sys
from tqdm import tqdm
from collections import Counter, defaultdict
import torch
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nsfr"))

from ocatari.vision.extract_vision_info import detect_objects_vision
from ocatari.vision.game_objects import GameObject
from ocatari.core import OCAtari
from ocatari.ram.seaquest import MAX_NB_OBJECTS as MAX_ESSENTIAL_OBJECTS

from nudge.agents.imitation_agent import ImitationAgent

# Action Mapping
ACTION_MAP = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE"
}

# Primitive Decomposition
PRIMITIVE_ACTION_MAP = {
    0: ["NOOP"],
    1: ["FIRE"],
    2: ["UP"],
    3: ["RIGHT"],
    4: ["LEFT"],
    5: ["DOWN"],
    6: ["UP", "RIGHT"],
    7: ["UP", "LEFT"],
    8: ["DOWN", "RIGHT"],
    9: ["DOWN", "LEFT"],
    10: ["UP", "FIRE"],
    11: ["RIGHT", "FIRE"],
    12: ["LEFT", "FIRE"],
    13: ["DOWN", "FIRE"],
    14: ["UP", "RIGHT", "FIRE"],
    15: ["UP", "LEFT", "FIRE"],
    16: ["DOWN", "RIGHT", "FIRE"],
    17: ["DOWN", "LEFT", "FIRE"]
}

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

class SnapshotObject:
    """Immutable snapshot of a tracked object state."""
    def __init__(self, cat, x, y, w, h, oid, orientation=None):
        self.id = oid
        # Structure compatible with extract_logic_state_from_tracked
        class Obj: pass
        self.obj = Obj()
        self.obj.category = cat
        self.obj.x = x
        self.obj.y = y
        self.obj.w = w
        self.obj.h = h
        self.obj.xywh = (x,y,w,h)
        self.obj.orientation = orientation

    @property
    def center(self):
        return (self.obj.x + self.obj.w / 2, self.obj.y + self.obj.h / 2)
    
    def to_dict(self):
        return {
            "id": self.id,
            "category": self.obj.category,
            "xywh": list(self.obj.xywh),
            "orientation": self.obj.orientation
        }

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class ObjectTracker:
    def __init__(self, max_missing_frames=5, match_dist_threshold=30):
        self.next_id = 0
        self.tracked_objects = [] # List of TrackedObject
        self.max_missing_frames = max_missing_frames
        self.match_dist_threshold = match_dist_threshold

    def update(self, detections):
        current_tracked = []
        unmatched_detections = list(detections)
        
        # Try to match existing tracks
        for track in self.tracked_objects:
            best_match = None
            min_dist = float('inf')
            match_idx = -1
            
            for i, det in enumerate(unmatched_detections):
                if det.category != track.obj.category:
                    continue
                
                # Center distance
                d_center = (det.x + det.w/2, det.y + det.h/2)
                t_center = track.center
                dist = get_distance(d_center, t_center)
                
                if dist < min_dist and dist < self.match_dist_threshold:
                    min_dist = dist
                    best_match = det
                    match_idx = i
            
            if best_match:
                track.update(best_match)
                current_tracked.append(track)
                unmatched_detections.pop(match_idx)
            else:
                track.missing_frames += 1
                if track.missing_frames < self.max_missing_frames:
                    current_tracked.append(track)
        
        # Add new tracks
        for det in unmatched_detections:
            current_tracked.append(TrackedObject(det, self.next_id))
            self.next_id += 1
            
        self.tracked_objects = current_tracked
        return self.tracked_objects

def load_trajectory_data(data_path, txt_file):
    gaze_map = {}
    action_map = {} # Map frame_id -> action_int
    if not os.path.exists(txt_file):
        print(f"Error: {txt_file} does not exist.")
        return gaze_map, action_map

    with open(txt_file, 'r') as f:
        header = f.readline() # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            
            frame_id_str = parts[0]
            try:
                fid = int(frame_id_str.split('_')[-1])
            except ValueError:
                continue
            
            # Action is at index 5 (6th column) based on file check
            # header: frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions
            try:
                action_val = int(parts[5])
            except:
                action_val = 0
            action_map[fid] = action_val
            
            gaze_data = []
            for x in parts[6:]: # Gaze columns start from index 6
                x = x.strip()
                if x and x.lower() != 'null':
                    try:
                        gaze_data.append(float(x))
                    except ValueError:
                        pass
            
            gaze_points = []
            for i in range(0, len(gaze_data), 2):
                if i+1 < len(gaze_data):
                    gaze_points.append((gaze_data[i], gaze_data[i+1]))
            
            gaze_map[fid] = gaze_points
            
    return gaze_map, action_map

def is_gaze_on_object(gaze, obj, margin=15):
    x, y, w, h = obj.obj.xywh
    if (x - margin) <= gaze[0] <= (x + w + margin) and \
       (y - margin) <= gaze[1] <= (y + h + margin):
        return True
    return False

def extract_logic_state_from_tracked(tracked_objects, n_objects=47, n_features=5):
    """
    Adapted from Seaquest NudgeEnv.extract_logic_state.
    Converts list of TrackedObject or SnapshotObject to logic state tensor (n_objects, n_features).
    Returns:
        state: torch.Tensor
        id_map: dict {tracked_object_id: logic_index}
    """
    state = torch.zeros((n_objects, n_features), dtype=torch.int32)
    id_map = {}
    
    relevant_objects_map = MAX_ESSENTIAL_OBJECTS.copy()
    if 'EnemyMissile' in relevant_objects_map:
        relevant_objects_map['EnemyMissile'] = 8
        
    obj_offsets = {}
    offset = 0
    for (obj_cat, max_count) in relevant_objects_map.items():
        obj_offsets[obj_cat] = offset
        offset += max_count
        
    relevant_categories = set(relevant_objects_map.keys())
    
    obj_count = {k: 0 for k in relevant_objects_map.keys()}
    
    type_map = {
        'Shark': 0, 'Submarine': 0, 'SurfaceSubmarine': 0,
        'Diver': 1, 'CollectedDiver': 6,
        'OxygenBar': 2,
        'Player': 3,
        'EnemyMissile': 5, 'PlayerMissile': 5
    }
    
    for tracked in tracked_objects:
        obj = tracked.obj
        if obj.category not in relevant_categories:
            continue
            
        if obj_count[obj.category] >= relevant_objects_map[obj.category]:
            continue
            
        idx = obj_offsets[obj.category] + obj_count[obj.category]
        type_id = type_map.get(obj.category, 0)
        
        # Store mapping
        # tracked.id is the TrackedObject.id
        id_map[tracked.id] = idx
        
        if obj.category == "OxygenBar":
            oxygen_level = obj.w
            state[idx] = torch.tensor([1, int(oxygen_level), int(obj.y), 0, type_id], dtype=torch.int32)
        else:
            orientation = 0 
            if hasattr(obj, "orientation") and obj.orientation is not None:
                 if hasattr(obj.orientation, "value"):
                     orientation = obj.orientation.value
                 else:
                     orientation = obj.orientation
            
            cx = obj.x + obj.w / 2
            cy = obj.y + obj.h / 2
            state[idx] = torch.tensor([1, int(cx), int(cy), int(orientation), type_id], dtype=torch.int32)
            
        obj_count[obj.category] += 1
        
    return state, id_map

import re

def get_atoms_from_state(agent, logic_state, logic_to_tracker_map=None):
    """
    Returns a list of string representations of true atoms (>0.5 probability).
    
    Args:
        agent: ImitationAgent with the NSFR model
        logic_state: Logic state tensor
        logic_to_tracker_map: Optional dict mapping logic state index -> tracker ID
                             If provided, replaces logic indices with tracker IDs in atoms
    """
    batch_state = logic_state.unsqueeze(0) # (1, N, F)
    with torch.no_grad():
        atoms_vals = agent.model.fc(batch_state, agent.model.atoms, agent.model.bk)
        
    probs = atoms_vals[0]
    result_atoms = []
    
    for i, prob in enumerate(probs):
        if prob > 0.5:
            atom_str = str(agent.model.atoms[i])
            
            # Replace logic indices with tracker IDs if mapping provided
            if logic_to_tracker_map:
                # NSFR model outputs atoms like: "diver(obj25)" or "close_by_diver(obj0,obj25)"
                # We need to replace objN with the tracker ID
                def replace_obj_id(match):
                    obj_notation = match.group(0)  # e.g., "obj25"
                    logic_idx = int(match.group(1))  # e.g., 25
                    tracker_id = logic_to_tracker_map.get(logic_idx, logic_idx)
                    return f"obj{tracker_id}"
                
                # Match objN pattern where N is one or more digits
                atom_str = re.sub(r'obj(\d+)', replace_obj_id, atom_str)
            
            result_atoms.append(atom_str)
            
    return result_atoms

def process_episode(episode_folder, txt_file, output_file=None, vis=False):
    print(f"Processing {episode_folder}...")
    
    # 1. Load Gaze & Action Data
    gaze_map, action_map = load_trajectory_data(episode_folder, txt_file)
    if not gaze_map:
        print("No gaze data found.")
        return

    # 2. Get Images
    image_files = sorted(glob.glob(os.path.join(episode_folder, "*.png")), 
                         key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    if not image_files:
        print("No images found.")
        return
        
    # 3. Initialize OCAtari and ImitationAgent
    env_name = "Seaquest" 
    oc = OCAtari(env_name, mode="vision", render_mode="rgb_array")
    
    print("Initializing ImitationAgent for relation extraction...")
    try:
        agent = ImitationAgent("seaquest", "new", "cpu")
        agent.model.eval()
    except Exception as e:
        print(f"Failed to load ImitationAgent: {e}")
        return
        
    tracker = ObjectTracker(match_dist_threshold=40)
    
    frame_data = [] 
    last_fixated_id = None
    current_fixation_duration = 0
    
    print("Extracting object and gaze data...")
    for img_path in tqdm(image_files):
        fid = int(os.path.basename(img_path).split('_')[-1].split('.')[0])
        
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            detect_objects_vision(oc.objects, image_rgb, "Seaquest", hud=False)
        except Exception:
            pass
            
        valid_objects = [o for o in oc.objects if o.category != "NoObject" and o.w > 0 and o.h > 0]
        active_objects = tracker.update(valid_objects)
        
        gaze_pts = gaze_map.get(fid, [])
        if not gaze_pts:
             mean_gaze = (80.0, 105.0) 
             has_gaze = False
        else:
            gx = sum(p[0] for p in gaze_pts) / len(gaze_pts)
            gy = sum(p[1] for p in gaze_pts) / len(gaze_pts)
            mean_gaze = (gx, gy)
            has_gaze = True
            
        fixated_obj = None
        fixated_obj_id = None
        fixated_obj_cat = "None"
        
        if has_gaze:
             candidates = []
             for obj in active_objects:
                 if is_gaze_on_object(mean_gaze, obj, margin=15):
                     dist = get_distance(mean_gaze, obj.center)
                     candidates.append((dist, obj))
             
             if candidates:
                 candidates.sort(key=lambda x: x[0])
                 fixated_obj = candidates[0][1]
             
             if not fixated_obj:
                 player_obj = next((o for o in active_objects if o.obj.category == "Player"), None)
                 if player_obj:
                     if is_gaze_on_object(mean_gaze, player_obj, margin=40):
                         fixated_obj = player_obj
        
        if fixated_obj:
            fixated_obj_id = fixated_obj.id
            fixated_obj_cat = fixated_obj.obj.category
        else:
            fixated_obj_cat = "Background"
            fixated_obj_id = -1
            
        if fixated_obj_id == last_fixated_id:
            current_fixation_duration += 1
        else:
            current_fixation_duration = 1
            last_fixated_id = fixated_obj_id
        
        # Compute logic state for atoms extraction (but keep tracker IDs for object identification)
        logic_state, id_map = extract_logic_state_from_tracked(active_objects)
        
        # Build reverse mapping: logic_index -> tracker_id
        logic_to_tracker = {logic_idx: tracker_id for tracker_id, logic_idx in id_map.items()}
        
        # Deep Snapshot objects to avoid mutation issues
        # Use TRACKER IDs (not logic state indices) for object identification
        snapshot_objects = []
        for o in active_objects:
            orient = None
            if hasattr(o.obj, "orientation") and o.obj.orientation is not None:
                 if hasattr(o.obj.orientation, "value"):
                     orient = o.obj.orientation.value
                 else:
                     orient = o.obj.orientation
            
            # Use TRACKER ID for object identification
            snap = SnapshotObject(
                cat=o.obj.category,
                x=o.obj.x,
                y=o.obj.y,
                w=o.obj.w,
                h=o.obj.h,
                oid=o.id,  # Use tracker ID for consistent object identification
                orientation=orient
            )
            snapshot_objects.append(snap)

        # Extract atoms for this frame, replacing logic indices with tracker IDs
        frame_atoms = get_atoms_from_state(agent, logic_state, logic_to_tracker)

        frame_data.append({
            "frame_id": fid,
            "gaze": mean_gaze,
            "fixated_object": fixated_obj_cat,
            "fixated_object_id": fixated_obj_id,  # Use tracker ID
            "objects": snapshot_objects, 
            "image_path": img_path,
            "action": action_map.get(fid, 0),  # Default to 0 (NOOP)
            "atoms": frame_atoms  # Store atoms for visualization
        })

    # 4. Strict ID-Based Segmentation
    print("Running ID-based Segmentation...")
    segments = []
    
    if not frame_data:
        print("No frame data to segment.")
        return

    current_seg_start = frame_data[0]["frame_id"]
    current_seg_obj_id = frame_data[0]["fixated_object_id"]
    current_seg_obj_cat = frame_data[0]["fixated_object"]
    current_fixation_counts = Counter()
    current_seg_start_idx = 0
    
    if current_seg_obj_id != -1 and current_seg_obj_cat not in ["Background", "None"]:
        # Use tracker ID for segment focus
        key = f"{current_seg_obj_cat.lower()}({current_seg_obj_id})"
        current_fixation_counts[key] += 1

    def get_obj_counts(objects):
        counts = {
            "divers": 0, "sharks": 0, "submarines": 0, 
            "missiles": 0, "collected_divers": 0
        }
        for obj in objects:
            # Handle SnapshotObject which simulates OCatari object structure
            # It has .obj.category
            if hasattr(obj, "obj") and hasattr(obj.obj, "category"):
                cat = obj.obj.category
            else:
                # Fallback if it's a dict (e.g. if dealing with serialized data)
                cat = obj.get("category", "Unknown") if isinstance(obj, dict) else "Unknown"

            if cat == "Diver": counts["divers"] += 1
            elif cat == "Shark": counts["sharks"] += 1
            elif cat in ["Submarine", "SurfaceSubmarine"]: counts["submarines"] += 1
            elif cat == "EnemyMissile": counts["missiles"] += 1
            elif cat == "CollectedDiver": counts["collected_divers"] += 1
        return counts

    def finalize_segment(start_idx, end_idx, provisional_focus, counts):
        # Start frame data only reflects on the second frame of that segment
        
        if start_idx==end_idx:
            start_data = frame_data[start_idx]
        else:
            start_data = frame_data[start_idx+1]

        end_data = frame_data[end_idx]
        
        # Extract atoms from stored frame data (already computed per-frame)
        start_atoms = start_data.get("atoms", [])
        end_atoms = end_data.get("atoms", [])

        # Use tracker ID for focus (already in provisional_focus)
        final_focus = provisional_focus
        
        # 2. Locations
        def get_player_loc(objs):
            p = next((o for o in objs if o.obj.category == "Player"), None)
            if p:
                return (int(p.center[0]), int(p.center[1]))
            return None 
            
        p_start = get_player_loc(start_data["objects"])
        p_end = get_player_loc(end_data["objects"])
        
        g_start = (int(start_data["gaze"][0]), int(start_data["gaze"][1]))
        g_end = (int(end_data["gaze"][0]), int(end_data["gaze"][1]))
        
        # Previous and Next Frame States
        prev_frame_state = []
        if start_idx > 0:
            prev_frame_state = frame_data[start_idx - 1].get("atoms", [])
            
        next_frame_state = []
        if end_idx < len(frame_data) - 1:
            next_frame_state = frame_data[end_idx + 1].get("atoms", [])
            
        # Object Counts
        start_obj_counts = get_obj_counts(start_data["objects"])
        end_obj_counts = get_obj_counts(end_data["objects"])
        
        # 3. Movement Logic
        movement_val = "none"
        if p_start and p_end:
            # Vectors
            vp = (p_end[0] - p_start[0], p_end[1] - p_start[1])
            dist_moved = math.sqrt(vp[0]**2 + vp[1]**2)
            
            # Gaze vector from start player pos
            vg = (g_start[0] - p_start[0], g_start[1] - p_start[1]) 
            dist_gaze = math.sqrt(vg[0]**2 + vg[1]**2)
            
            if dist_moved > 2: 
                dist_start_to_gaze = dist_gaze
                dist_end_to_gaze = math.sqrt((g_start[0] - p_end[0])**2 + (g_start[1] - p_end[1])**2)
                
                dx_p = vp[0]
                dy_p = vp[1]
                dx_g = vg[0]
                dy_g = vg[1]

                same_sign_x = (np.sign(dx_p) == np.sign(dx_g)) and (abs(dx_p) > 1 and abs(dx_g) > 1)
                same_sign_y = (np.sign(dy_p) == np.sign(dy_g)) and (abs(dy_p) > 1 and abs(dy_g) > 1)
                
                if dist_end_to_gaze < dist_start_to_gaze - 2:
                    movement_val = "towards_gaze"
                elif dist_end_to_gaze > dist_start_to_gaze + 2:
                    movement_val = "away_from_gaze"
                else:
                    if same_sign_x and not same_sign_y:
                        movement_val = "towards_gaze"
                    elif same_sign_y and not same_sign_x:
                        movement_val = "towards_gaze"
                    elif (not same_sign_x) and (abs(dx_p) > 1) and (abs(dx_g) > 1):
                        movement_val = "away_from_gaze"
                    elif (not same_sign_y) and (abs(dy_p) > 1) and (abs(dy_g) > 1):
                         movement_val = "away_from_gaze"
        
        # 4. Actions
        action_seq_int = []
        action_seq_str = []
        primitive_action_list = []
        
        for i in range(start_idx, end_idx + 1):
            act = frame_data[i]["action"]
            action_seq_int.append(act)
            action_seq_str.append(ACTION_MAP.get(act, "UNKNOWN"))
            
            # Decompose to primitives
            prims = PRIMITIVE_ACTION_MAP.get(act, ["UNKNOWN"])
            primitive_action_list.extend(prims)
            
        action_counts = Counter(action_seq_str)
        sorted_actions = dict(sorted(action_counts.items(), key=lambda item: item[1], reverse=True))
        
        primitive_counts = Counter(primitive_action_list)
        sorted_primitives = dict(sorted(primitive_counts.items(), key=lambda item: item[1], reverse=True))

        seg = {
            "start_frame": start_data["frame_id"],
            "end_frame": end_data["frame_id"],
            "focus": final_focus,
            "fixation_counts": counts,
            "start_frame_state": start_atoms,
            "end_frame_state": end_atoms,
            "prev_frame_state": prev_frame_state,
            "next_frame_state": next_frame_state,
            "start_frame_obj_counts": start_obj_counts,
            "end_frame_obj_counts": end_obj_counts,
            "player_loc_start": p_start,
            "player_loc_end": p_end,
            "movement": movement_val,
            "action_counts": sorted_actions,
            "primitive_action_counts": sorted_primitives,
            "action_seq": action_seq_int
        }
        
        if g_start == g_end:
            seg["gaze_loc"] = g_start
        else:
            seg["gaze_loc_start"] = g_start
            seg["gaze_loc_end"] = g_end
            
        return seg

    for i in range(1, len(frame_data)):
        fd = frame_data[i]
        fid = fd["frame_id"]
        oid = fd["fixated_object_id"]
        cat = fd["fixated_object"]
        
        if oid != current_seg_obj_id:
            if current_seg_obj_id != -1 and current_seg_obj_cat not in ["Background", "None"]:
                focus = f"{current_seg_obj_cat.lower()}({current_seg_obj_id})"
            else:
                 focus = "Explore"
            
            seg = finalize_segment(current_seg_start_idx, i-1, focus, dict(current_fixation_counts))
            segments.append(seg)
            
            current_seg_start = fid
            current_seg_obj_id = oid
            current_seg_obj_cat = cat
            current_fixation_counts = Counter()
            current_seg_start_idx = i
            
        if oid != -1 and cat not in ["Background", "None"]:
            # Use tracker ID for fixation count key
            key = f"{cat.lower()}({oid})"
            current_fixation_counts[key] += 1
            
    if current_seg_obj_id != -1 and current_seg_obj_cat not in ["Background", "None"]:
        focus = f"{current_seg_obj_cat.lower()}({current_seg_obj_id})"
    else:
        focus = "Explore"
        
    seg = finalize_segment(current_seg_start_idx, len(frame_data)-1, focus, dict(current_fixation_counts))
    segments.append(seg)

    print(f"Found {len(segments)} segments. Running post-processing (Stitching & Labeling)...")
    
    # --- Post-Processing Functions ---

    def label_segment(seg):
        # Intent labeling removed per user request
        # seg["intent"] = "monitoring" 
        return seg

    def get_action_similarity(seg1, seg2):
        # Calculate Cosine Similarity of primitive_action_counts
        c1 = seg1.get("primitive_action_counts", {})
        c2 = seg2.get("primitive_action_counts", {})
        
        # if not c1 and not c2: return 1.0
        # if not c1 or not c2: return 0.0
        
        all_keys = set(c1.keys()) | set(c2.keys())
        
        # Create vectors
        # Note: counts are dicts
        vec1 = np.array([c1.get(k, 0) for k in all_keys])
        vec2 = np.array([c2.get(k, 0) for k in all_keys])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0: return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def stitch_segments(original_segments, threshold=10):
        stitched = []
        i = 0
        while i < len(original_segments):
            curr = original_segments[i]
            
            # --- LOGIC -1: Auto-Merge 'Explore' into Previous Segment ---
            # Use Case: 'Explore' usually means looking around while maintaining the same underlying intent/context.
            # Unlike Logic 0 (which requires similarity), we unconditionally merge Explore if it follows something.
            if curr["focus"] == "Explore" and stitched:
                prev = stitched[-1]
                
                # MERGE CURR INTO PREV
                
                prev["end_frame"] = curr["end_frame"]
                prev["player_loc_end"] = curr["player_loc_end"]
                prev["gaze_loc_end"] = curr.get("gaze_loc_end", curr.get("gaze_loc"))
                prev["end_frame_state"] = curr["end_frame_state"]
                
                # Merge Actions
                prev["action_seq"].extend(curr["action_seq"])
                
                # Add as waver logic for visualization
                w_info = {
                    "frame": curr["start_frame"],
                    "focus": curr["focus"],
                    "duration": curr["end_frame"] - curr["start_frame"],
                    "waver_focus": curr["focus"]
                }
                prev.setdefault("wavers", []).append(w_info)
                
                # Recompute Counts (Merging dicts efficiently)
                # We need full recount because we want sorted order
                all_actions_str = []
                all_prims = []
                for act_int in prev["action_seq"]:
                    all_actions_str.append(ACTION_MAP.get(act_int, "UNKNOWN"))
                    all_prims.extend(PRIMITIVE_ACTION_MAP.get(act_int, ["UNKNOWN"]))
                
                prev["action_counts"] = dict(sorted(Counter(all_actions_str).items(), key=lambda x: x[1], reverse=True))
                prev["primitive_action_counts"] = dict(sorted(Counter(all_prims).items(), key=lambda x: x[1], reverse=True))
                
                # Consume current and continue loop
                i += 1
                continue
            
            # --- LOGIC 0: Try to Merge with PREVIOUS Segment (Extension Rule) ---
            # Use Case: Player/Explore often interrupts a focused task. 
            # If the action distribution during this "distraction" is identical to the previous "focus",
            # it's likely a continuation of the same intent.
            if stitched:
                prev = stitched[-1]
                prev_focus = prev["focus"]
                curr_focus = curr["focus"]
                
                # Check criteria: Prev is Content, Curr is Weak (Player/Explore)
                prev_is_content = "player" not in prev_focus.lower() and "explore" not in prev_focus.lower()
                curr_is_weak = "player" in curr_focus.lower() or "explore" in curr_focus.lower()
                
                if prev_is_content and curr_is_weak:
                    sim = get_action_similarity(prev, curr)
                    # Similarity Threshold (0.8 is quite high, ensuring strong match)
                    if sim > 0.8:
                        # MERGE CURR INTO PREV
                        # Update ends
                        prev["end_frame"] = curr["end_frame"]
                        prev["player_loc_end"] = curr["player_loc_end"]
                        prev["gaze_loc_end"] = curr.get("gaze_loc_end", curr.get("gaze_loc"))
                        prev["end_frame_state"] = curr["end_frame_state"]
                        
                        # Merge Actions
                        prev["action_seq"].extend(curr["action_seq"])
                        
                        # Add as waver logic for visualization? 
                        # Technically it's not a "waver" in the middle, it's an extension.
                        # But we can track it as a sub-segment for debugging.
                        w_info = {
                            "frame": curr["start_frame"],
                            "focus": curr["focus"],
                            "duration": curr["end_frame"] - curr["start_frame"],
                            "waver_focus": curr["focus"]
                        }
                        prev.setdefault("wavers", []).append(w_info)
                        
                        # Recompute Counts (Merging dicts efficiently)
                        # We need full recount because we want sorted order
                        all_actions_str = []
                        all_prims = []
                        for act_int in prev["action_seq"]:
                            all_actions_str.append(ACTION_MAP.get(act_int, "UNKNOWN"))
                            all_prims.extend(PRIMITIVE_ACTION_MAP.get(act_int, ["UNKNOWN"]))
                        
                        prev["action_counts"] = dict(sorted(Counter(all_actions_str).items(), key=lambda x: x[1], reverse=True))
                        prev["primitive_action_counts"] = dict(sorted(Counter(all_prims).items(), key=lambda x: x[1], reverse=True))
                        
                        # Consume current and continue loop
                        i += 1
                        continue
            
            # --- LOGIC 1: Look Ahead Stitching (A -> Waver -> A) ---
            
            # Condition 1: Exploration should not count as anchor for Look Ahead
            if curr["focus"] == "Explore":
                stitched.append(curr)
                i += 1
                continue
            
            # Condition 2: Must be moving towards the focus
            if "towards" not in curr.get("movement", ""):
                 stitched.append(curr)
                 i += 1
                 continue
                 
            # Look ahead logic
            match_found = False
            waver_duration = 0
            j = i + 1
            
            # Allow multiple segments in between as long as they sum <= threshold
            while j < len(original_segments):
                cand = original_segments[j]
                
                # Check for Focus Match
                if cand["focus"] == curr["focus"]:
                    # Condition 3: The second segment must ALSO be moving towards
                    if "towards" in cand.get("movement", ""):
                        # MERGE [i ... j]
                        merged = curr.copy()
                        merged["end_frame"] = cand["end_frame"]
                        merged["player_loc_end"] = cand["player_loc_end"]
                        merged["gaze_loc_end"] = cand.get("gaze_loc_end", cand.get("gaze_loc"))
                        merged["end_frame_state"] = cand["end_frame_state"]
                        
                        # Accumulate Actions & Waver Info
                        all_actions_int = list(curr["action_seq"])
                        wavers_list = curr.get("wavers", [])
                        
                        for k in range(i + 1, j):
                            mid_seg = original_segments[k]
                            all_actions_int.extend(mid_seg["action_seq"])
                            w_info = {
                                "frame": mid_seg["start_frame"],
                                "focus": mid_seg["focus"],
                                "duration": mid_seg["end_frame"] - mid_seg["start_frame"],
                                "waver_focus": mid_seg["focus"] 
                            }
                            wavers_list.append(w_info)
                        
                        all_actions_int.extend(cand["action_seq"])
                        
                        merged["action_seq"] = all_actions_int
                        merged["wavers"] = wavers_list
                        
                        # Recompute Counts
                        all_actions_str = []
                        all_prims = []
                        for act_int in all_actions_int:
                            all_actions_str.append(ACTION_MAP.get(act_int, "UNKNOWN"))
                            all_prims.extend(PRIMITIVE_ACTION_MAP.get(act_int, ["UNKNOWN"]))
                            
                        merged["action_counts"] = dict(sorted(Counter(all_actions_str).items(), key=lambda x: x[1], reverse=True))
                        merged["primitive_action_counts"] = dict(sorted(Counter(all_prims).items(), key=lambda x: x[1], reverse=True))
                        
                        stitched.append(merged)
                        i = j + 1
                        match_found = True
                        break
                    else:
                        break
                else:
                    # Wavering segment
                    waver_duration += (cand["end_frame"] - cand["start_frame"])
                    if waver_duration > threshold:
                        break # Too long, stop looking
                    j += 1
            
            if not match_found:
                stitched.append(curr)
                i += 1
                
        return stitched
    def remove_short_player_segments(segments, threshold=4):
        """
        Remove segments that are shorter than the threshold.
        
        Args:
            segments: List of segments
            threshold: Minimum duration for a segment to be kept
        
        Returns:
            List of segments with short segments removed
        """
        filtered_segments = []

        for seg in segments:
            if "player" not in seg["focus"]:
                filtered_segments.append(seg)
                continue
            if seg["end_frame"] - seg["start_frame"] >= threshold:
                filtered_segments.append(seg)
        return filtered_segments
    
    def merge_sandwiched_player(segments, player_threshold=15):
        """
        Merge short player segments that are sandwiched between two segments
        of the same object. The player segment becomes a waver.
        
        Args:
            segments: List of segments after initial stitching
            player_threshold: Max frames for player segment to be considered short
        
        Returns:
            List of segments with sandwiched players merged
        """
        merged = []
        i = 0
        
        while i < len(segments):
            curr = segments[i]
            
            # Check if this is a player segment
            is_player = "player" in curr["focus"].lower()
            is_surface = "surface" in curr["focus"].lower()
            if (is_player or is_surface) and i >0 and i < len(segments) - 1:
                prev_seg = segments[i - 1]
                next_seg = segments[i + 1]
                
                # Check if prev and next focus on the same object
                if prev_seg["focus"] == next_seg["focus"]:
                    # Check if player segment is short
                    player_duration = curr["end_frame"] - curr["start_frame"] + 1
                    
                    if player_duration <= player_threshold:
                        # MERGE: Extend prev to include player and next
                        # Remove the previously added prev from merged (we'll add extended version)
                        if merged and merged[-1] is prev_seg:
                            merged.pop()
                        
                        # Create extended segment
                        extended = prev_seg.copy()
                        extended["end_frame"] = next_seg["end_frame"]
                        extended["player_loc_end"] = next_seg["player_loc_end"]
                        extended["gaze_loc_end"] = next_seg.get("gaze_loc_end", next_seg.get("gaze_loc"))
                        extended["end_frame_state"] = next_seg["end_frame_state"]
                        
                        # Merge actions from curr and next
                        extended["action_seq"].extend(curr["action_seq"])
                        extended["action_seq"].extend(next_seg["action_seq"])
                        
                        # Add player as waver
                        w_info = {
                            "frame": curr["start_frame"],
                            "focus": curr["focus"],
                            "start_frame": curr["start_frame"],
                            "end_frame": curr["end_frame"],
                            "duration": player_duration,
                            "waver_focus": curr["focus"]
                        }
                        extended.setdefault("wavers", []).append(w_info)
                        
                        # Merge any existing wavers from next segment
                        if "wavers" in next_seg:
                            extended.setdefault("wavers", []).extend(next_seg["wavers"])
                        
                        # Recompute action counts
                        from collections import Counter
                        all_actions_str = []
                        all_prims = []
                        for act_int in extended["action_seq"]:
                            all_actions_str.append(ACTION_MAP.get(act_int, "UNKNOWN"))
                            all_prims.extend(PRIMITIVE_ACTION_MAP.get(act_int, ["UNKNOWN"]))
                        
                        extended["action_counts"] = dict(sorted(Counter(all_actions_str).items(), key=lambda x: x[1], reverse=True))
                        extended["primitive_action_counts"] = dict(sorted(Counter(all_prims).items(), key=lambda x: x[1], reverse=True))
                        
                        merged.append(extended)
                        
                        # Skip curr and next (we merged them into prev)
                        i += 3  # Skip prev (already added), curr, and next
                        continue
            
            # Normal case: add current segment
            merged.append(curr)
            i += 1
        
        return merged

    # Run Stitching
    final_segments = stitch_segments(segments, threshold=10) # User requested threshold 10
    
    # Merge sandwiched player segments
    final_segments = merge_sandwiched_player(final_segments, player_threshold=15)
    
    # Remove short player segments
    final_segments = remove_short_player_segments(final_segments, threshold=4)


    # Recalculate prev_frame_state and next_frame_state
    for i, seg in enumerate(final_segments):
        if i > 0:
            seg["prev_frame_state"] = final_segments[i-1]["end_frame_state"]
        if i < len(final_segments) - 1:
            seg["next_frame_state"] = final_segments[i+1]["start_frame_state"]
    
    
    # Run Labeling
    for i, seg in enumerate(final_segments):
        seg["segment_number"] = i
        label_segment(seg)

    print(f"Final segment count after stitching: {len(final_segments)}")
    
    # Serialize Frame Data (Convert SnapshotObjects to dicts)
    serializable_frame_data = []
    for fd in frame_data:
        serializable_objects = [o.to_dict() for o in fd["objects"]]
        new_fd = fd.copy()
        new_fd["objects"] = serializable_objects
        serializable_frame_data.append(new_fd)
    
    output_data = {
        "segments": final_segments,
        "frame_data": serializable_frame_data
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2) 
            
    if vis:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/nikhilesh/Projects/NUDGE/data/seaquest/237_RZ_9656617_Feb-08-14-12-21")
    parser.add_argument("--txt_file", type=str, default="/home/nikhilesh/Projects/NUDGE/data/seaquest/237_RZ_9656617_Feb-08-14-12-21.txt")
    parser.add_argument("--output", type=str, default="gaze_segments.json")
    
    args = parser.parse_args()
    
    process_episode(args.data_dir, args.txt_file, args.output)
