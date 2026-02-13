import os
print("DEBUG: Executing updated extract_gaze_goals.py with try-except fix")
import glob
import pandas as pd
import numpy as np
import cv2
import json
import argparse
import sys
from tqdm import tqdm
import ruptures as rpt
from collections import Counter, defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ocatari.core import OCAtari
from ocatari.vision.utils import mark_bb, make_darker
from ocatari.vision.game_objects import GameObject

def load_trajectory_data(data_path, txt_file):
    """
    Loads gaze data from the .txt file.
    Returns a dictionary mapping frame_id (int) to gaze points (list of tuples).
    """
    gaze_map = {}
    if not os.path.exists(txt_file):
        print(f"Error: {txt_file} does not exist.")
        return gaze_map

    with open(txt_file, 'r') as f:
        header = f.readline() # Skip header
        for line in f:
            parts = line.strip().split(',')
            frame_id_str = parts[0]
            # Extract int frame id from "RZ_..._123"
            try:
                fid = int(frame_id_str.split('_')[-1])
            except ValueError:
                continue
            
            gaze_data = []
            # Gaze columns start from index 6
            for x in parts[6:]:
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
            
    return gaze_map

def get_object_at_point(objects, x, y):
    """
    Returns the object at the given (x, y) coordinate.
    Prioritizes specific categories if needed.
    """
    # Check intersection with bounding boxes
    # objects is a list of GameObjects
    
    # Hierarchical check? Smallest object first?
    # Or just return all matches?
    # For now, let's return the first match, but prioritize enemies/oxygen?
    
    matches = []
    for obj in objects:
        ox, oy, w, h = obj.x, obj.y, obj.w, obj.h
        if ox <= x <= ox + w and oy <= y <= oy + h:
            matches.append(obj)
            
    if not matches:
        return None
        
    # Heuristic: return the one with the smallest area (more specific)
    matches.sort(key=lambda o: o.w * o.h)
    return matches[0]

def process_episode(episode_folder, txt_file, output_file=None, vis=False):
    print(f"Processing {episode_folder}...")
    
    # 1. Load Gaze Data
    gaze_map = load_trajectory_data(episode_folder, txt_file)
    if not gaze_map:
        print("No gaze data found.")
        return

    # 2. Get Images
    image_files = sorted(glob.glob(os.path.join(episode_folder, "*.png")), 
                         key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    if not image_files:
        print("No images found.")
        return
        
    # 3. Initialize OCAtari
    env_name = "Seaquest" # extraction for seaquest
    oc = OCAtari(env_name, mode="vision", render_mode="rgb_array")
    
    # Collect data for CPD
    # We need a time series of gaze points associated with frame IDs
    # CPD on (x, y) coordinates?
    
    # We will align everything by frame ID
    frame_ids = []
    gaze_sequence = [] # List of mean gaze point per frame (or all points?)
    # If multiple gaze points per frame, maybe take the mean?
    
    # Data storage
    frame_data = [] 
    
    print("Extracting object and gaze data...")
    for img_path in tqdm(image_files):
        fid = int(os.path.basename(img_path).split('_')[-1].split('.')[0])
        
        # Load Image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect Objects (using Mock ALE to pass image to OCAtari)
        if hasattr(oc, "_env") and hasattr(oc._env, "unwrapped") and hasattr(oc._env.unwrapped, "ale"):
            real_ale = oc._env.unwrapped.ale
            
            class MockALE:
                def __getattr__(self, name):
                    return getattr(real_ale, name)
                def getScreenRGB(self, *args):
                    return image_rgb
            
            oc._env.unwrapped.ale = MockALE()
            
            try:
                oc.detect_objects(image_rgb) # Pass logic/hud check if needed, but it reads from ALE
            except TypeError:
                 # Fallback if signature mismatch
                 try:
                     oc.detect_objects()
                 except Exception as e:
                     print(f"Skipping frame {fid} due to fallback error: {e}")
                     continue
            except ValueError as e:
                print(f"Skipping frame {fid} due to OCAtari error: {e}")
                continue
            except Exception as e:
                print(f"Error in detect_objects: {e}")
                continue
            finally:
                oc._env.unwrapped.ale = real_ale
        else:
            # Fallback for other versions
            try:
                oc.detect_objects(image_rgb)
            except Exception as e:
                print(f"Skipping frame {fid} due to fallback error: {e}")
                continue
            
        objects = oc.objects
        
        # Get Gaze
        gaze_pts = gaze_map.get(fid, [])
        if not gaze_pts:
            # Handle missing gaze? Interpolate? Or just mark as unknown?
            # For CPD we need continuous signal.
            # Let's use previous gaze or NaN
            if gaze_sequence:
                mean_gaze = gaze_sequence[-1] # Propagate last known
            else:
                mean_gaze = (80.0, 105.0) # Center default
        else:
            # Mean gaze for this frame
            gx = sum(p[0] for p in gaze_pts) / len(gaze_pts)
            gy = sum(p[1] for p in gaze_pts) / len(gaze_pts)
            mean_gaze = (gx, gy)
            
        frame_ids.append(fid)
        gaze_sequence.append(mean_gaze)
        
        # Identify fixated object
        fixated_obj_cat = "None"
        if gaze_pts:
             # Check all points, vote?
             votes = []
             for (gx, gy) in gaze_pts:
                 # Map to object
                 # Coordinate scaling?
                 # Gaze data seems to be in 160x210 space based on previous inspection (values ~100-120)
                 # OCAtari objects are in 160x210.
                 obj = get_object_at_point(objects, gx, gy)
                 if obj:
                     votes.append(obj.category)
                 else:
                     votes.append("Background")
             
             if votes:
                 fixated_obj_cat = Counter(votes).most_common(1)[0][0]
        
        frame_data.append({
            "frame_id": fid,
            "gaze": mean_gaze,
            "fixated_object": fixated_obj_cat,
            "objects": [(o.category, o.xywh) for o in objects],
            "image_path": img_path
        })

    # 4. Apply Change Point Detection
    print("Running Change Point Detection...")
    signal = np.array(gaze_sequence)
    
    # Penalty needs tuning
    # algo = rpt.Pelt(model="rbf").fit(signal)
    # result = algo.predict(pen=10)
    
    # Or Binary Segmentation
    algo = rpt.Binseg(model="l2").fit(signal)
    result = algo.predict(n_bkps=10) # Assume max 10 subgoals per episode? Or use penalty.
    # result = algo.predict(pen=20) # Try penalty

    print(f"Found {len(result)} segments.")
    
    # 5. Determine Goal per Segment
    segments = []
    start_idx = 0
    for end_idx in result:
        # Segment range: [start_idx, end_idx)
        segment_data = frame_data[start_idx:end_idx]
        
        # Count fixations
        fixation_counts = Counter()
        for fd in segment_data:
            obj = fd["fixated_object"]
            if obj != "None" and obj != "Background":
                fixation_counts[obj] += 1
        
        # Goal is the most fixated object
        if fixation_counts:
            goal = fixation_counts.most_common(1)[0][0]
        else:
            goal = "Explore" # Fallback
            
        # Refine Goal: "Retrieve Diver", "Go to Surface", "Shoot Enemy"
        # Just use object category for now.
            
        segments.append({
            "start_frame": frame_ids[start_idx],
            "end_frame": frame_ids[end_idx-1],
            "goal": goal,
            "fixation_counts": dict(fixation_counts)
        })
        
        start_idx = end_idx
        
    print("Segments:")
    for seg in segments:
        print(f"  Frames {seg['start_frame']}-{seg['end_frame']}: Goal = {seg['goal']}")
        
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(segments, f, indent=2)
            
    # Visualization (Optional)
    if vis:
        visualize_segments(frame_data, result, segments)

def visualize_segments(frame_data, bkps, segments):
    # Create video or show trajectory
    pass # TODO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/nikhilesh/Projects/NUDGE/data/seaquest/237_RZ_9656617_Feb-08-14-12-21")
    parser.add_argument("--txt_file", type=str, default="/home/nikhilesh/Projects/NUDGE/data/seaquest/237_RZ_9656617_Feb-08-14-12-21.txt")
    parser.add_argument("--output", type=str, default="gaze_segments.json")
    
    args = parser.parse_args()
    
    # Check dependencies
    # pip install ruptures
    
    process_episode(args.data_dir, args.txt_file, args.output)
