import cv2
import json
import argparse
import os
import glob
import numpy as np
from tqdm import tqdm

# Constants
SCALE = 4
IMG_W = 160 * SCALE
IMG_H = 210 * SCALE
SIDEBAR_W = 300
TOTAL_W = IMG_W + SIDEBAR_W
TOTAL_H = IMG_H
BG_COLOR = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

def draw_sidebar(img, current_frame_idx, current_seg, segments):
    # Fill sidebar background
    cv2.rectangle(img, (IMG_W, 0), (TOTAL_W, TOTAL_H), BG_COLOR, -1)
    
    x = IMG_W + 10
    y = 30
    line_height = 25
    
    # Frame Info
    # Frame Info
    seg_info = ""
    if current_seg:
        seg_info = f"(Seg: {current_seg['start_frame']}-{current_seg['end_frame']})"
    cv2.putText(img, f"Frame: {current_frame_idx} {seg_info}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    y += line_height
    
    if current_seg:
        # Focus Info - Display in dark red to match bounding box
        focus = current_seg.get("focus", "None")
        DARK_RED = (0, 0, 139)  # BGR format
        cv2.putText(img, f"Focus: {focus}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DARK_RED, 2)
        y += line_height * 1.5
        
        # Wavers - List all with frame ranges in green
        wavers = current_seg.get("wavers", [])
        if wavers:
            cv2.putText(img, "Wavers:", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
            y += line_height
            
            for w in wavers:
                waver_focus = w.get("waver_focus", "Unknown")
                start_frame = w.get("frame", 0)
                duration = w.get("duration", 0)
                end_frame = start_frame + duration - 1
                
                waver_text = f"  {waver_focus} [{start_frame}-{end_frame}]"
                cv2.putText(img, waver_text, (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
                y += int(line_height * 0.9)
            
            y += int(line_height * 0.3)  # Extra spacing after wavers
        
        # Movement Info
        move = current_seg.get("movement", "None")
        cv2.putText(img, f"Move: {move}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
        y += line_height * 1.5

        # Intent
        intent = current_seg.get("intent", "N/A")
        cv2.putText(img, f"Intent: {intent}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW if intent == "action" else TEXT_COLOR, 1)
        y += line_height * 1.5
        
        # Action Sequence - Full List
        action_seq = current_seg.get("action_seq", [])
        if action_seq:
            cv2.putText(img, "Action Sequence:", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
            y += line_height
            
            # Map action integers to their string names
            ACTION_MAP = {
                0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
                6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT",
                10: "UPFIRE", 11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE",
                14: "UPRIGHTFIRE", 15: "UPLEFTFIRE", 16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE"
            }
            
            # Display all actions
            action_str_list = [ACTION_MAP.get(a, f"A{a}") for a in action_seq]
            action_text = ", ".join(action_str_list)
            
            # Word wrap for long sequences
            max_chars_per_line = 25
            words = action_text.split(", ")
            current_line = ""
            
            for word in words:
                test_line = current_line + (", " if current_line else "") + word
                if len(test_line) > max_chars_per_line and current_line:
                    cv2.putText(img, current_line, (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
                    y += int(line_height * 0.8)
                    current_line = word
                else:
                    current_line = test_line
            
            # Print remaining line
            if current_line:
                cv2.putText(img, current_line, (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
                y += int(line_height * 0.8)
                
    else:
        cv2.putText(img, "No Segment Info", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)

def visualize_gaze_segments(json_file, output_video):
    print(f"Loading data from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    frame_data = data["frame_data"]
    segments = data["segments"]
    
    if not frame_data:
        print("No frame data found.")
        return

    # Sort frame data by ID just in case
    frame_data.sort(key=lambda x: x["frame_id"])
    
    # Map frame_id to segment
    frame_to_seg = {}
    for seg in segments:
        for fid in range(seg["start_frame"], seg["end_frame"] + 1):
            frame_to_seg[fid] = seg
            
    # Video Writer
    # Video Writer (50ms per frame = 20 FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (TOTAL_W, TOTAL_H))
    
    print(f"Generating video to {output_video}...")
    
    for i, fd in enumerate(tqdm(frame_data)):
        img_path = fd.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue
            
        # Get Segment Info First
        fid = fd["frame_id"]
        current_seg = frame_to_seg.get(fid)

        # Load and Resize Image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        
        # Create Canvas
        canvas = np.zeros((TOTAL_H, TOTAL_W, 3), dtype=np.uint8)
        canvas[:, :IMG_W] = img
        
        # Draw Overlays
        gaze = fd.get("gaze")
        if gaze:
            gx, gy = int(gaze[0] * SCALE), int(gaze[1] * SCALE)
            cv2.circle(canvas, (gx, gy), 5, BLUE, -1)
            cv2.circle(canvas, (gx, gy), 50, (0, 0, 255), 2) # Gaze radius
            
        # Draw Bounding Box of Fixated Object
        # Logic:
        # - Identify Focus Object ID from current_seg["focus"] -> RED
        # - Identify if we are in a waver -> YELLOW
        
        focus_id = -1
        waver_id = -1
        focus_category = None  # Track focus category for fallback
        
        if current_seg:
            # Parse Focus ID and Category
            # focus string format: "category(ID)" e.g. "shark(3)"
            # If "Explore", no ID
            if "Explore" not in current_seg["focus"]:
                try:
                    focus_str = current_seg["focus"]
                    focus_category = focus_str.split('(')[0].lower()  # e.g., "shark"
                    focus_id_str = focus_str.split('(')[1].split(')')[0]
                    focus_id = int(focus_id_str)
                except:
                    pass
            
            # Check for Waver
            # Wavers list: [{frame, duration, waver_focus}, ...]
            wavers = current_seg.get("wavers", [])
            for w in wavers:
                if w["frame"] <= fid < (w["frame"] + w["duration"]):
                    # We are in this waver
                    # Parse Waver ID
                    if "Explore" not in w.get("waver_focus", ""):
                         try:
                            w_id_str = w["waver_focus"].split('(')[1].split(')')[0]
                            waver_id = int(w_id_str)
                         except:
                            pass
                    break

        all_objs = fd.get("objects", [])
        for obj in all_objs:
            oid = obj["id"]
            color = BLUE # Default (Blue)
            thickness = 1
            
            if oid == focus_id:
                color = (0, 0, 139)  # Dark red (BGR format)
                thickness = 5
            elif oid == waver_id:
                color = GREEN # Waver (Green)
                thickness = 2
                
            # Only draw if it's one of the interesting objects?
            # Or draw all?
            # Original code only drew 'fixated_object_id'.
            # User said "Highlight the goal bounding box... and wavering...".
            # Implies we define the boxes by the Goals, not just what was fixated in that specific frame 
            # (though usually goal == fixated, unless wavering).
            # If I'm stitching, the "Goal" object might NOT be fixated in the current frame (if wavering).
            # So we should draw the Goal object if it exists in the frame, regardless of fixation.
            
            if oid == focus_id or oid == waver_id or oid == fd.get("fixated_object_id"):
                 ox, oy, w, h = obj["xywh"]
                 sx, sy, sw, sh = int(ox*SCALE), int(oy*SCALE), int(w*SCALE), int(h*SCALE)
                 cv2.rectangle(canvas, (sx, sy), (sx+sw, sy+sh), color, thickness)
                 cv2.putText(canvas, obj["category"], (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw Sidebar
        draw_sidebar(canvas, fid, current_seg, segments)
        
        out.write(canvas)
        
    out.release()
    print("Video saved successfully.")

if __name__ == "__main__":
    # Default to the generated file if it exists
    default_json = "gaze_goals_verification.json"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, default=default_json)
    parser.add_argument("--output_video", type=str, default="gaze_segmentation_video.mp4")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"File {args.json_file} not found.")
    else:
        visualize_gaze_segments(args.json_file, args.output_video)
