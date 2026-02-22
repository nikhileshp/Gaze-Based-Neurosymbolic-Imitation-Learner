import cv2
import json
import argparse
import os
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
DARK_RED = (0, 0, 139)  # BGR format
PURPLE = (255, 0, 255)  # For atoms display

def draw_sidebar(img, current_frame_idx, current_seg, segments):
    # Fill sidebar background
    cv2.rectangle(img, (IMG_W, 0), (TOTAL_W, TOTAL_H), BG_COLOR, -1)
    
    x = IMG_W + 10
    y = 30
    line_height = 25
    
    # Frame Info
    seg_info = ""
    if current_seg:
        seg_info = f"(Seg: {current_seg['start_frame']}-{current_seg['end_frame']})"
    cv2.putText(img, f"Frame: {current_frame_idx} {seg_info}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    y += line_height
    
    if current_seg:
        # Focus Info - Display in RED
        focus = current_seg.get("focus", "None")
        cv2.putText(img, f"Focus: {focus}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DARK_RED, 2)
        y += line_height * 1.5
        
        # Wavers - List all with frame ranges in GREEN
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
        
        # Action Sequence
        action_seq = current_seg.get("action_seq", [])
        if action_seq:
            cv2.putText(img, "Action Sequence:", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
            y += line_height
            
            ACTION_MAP = {
                0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
                6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT",
                10: "UPFIRE", 11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE",
                14: "UPRIGHTFIRE", 15: "UPLEFTFIRE", 16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE"
            }
            
            action_str_list = [ACTION_MAP.get(a, f"A{a}") for a in action_seq]
            action_text = ", ".join(action_str_list)
            
            # Word wrap
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
            
            if current_line:
                cv2.putText(img, current_line, (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
                y += int(line_height * 0.8)
        
        # Atoms/Relations Display
        atoms = current_seg.get("atoms", [])
        if atoms and y < TOTAL_H - 100:  # Only show if we have space
            cv2.putText(img, "Relations:", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PURPLE, 2)
            y += line_height
            
            # Extract focus object tracker ID from focus string
            focus_tracker_id = None
            focus_str = current_seg.get("focus", "")
            if focus_str and "(" in focus_str and ")" in focus_str:
                try:
                    focus_tracker_id = int(focus_str.split('(')[1].split(')')[0])
                except:
                    pass
            
            # Separate atoms into focus-related and others
            focus_atoms = []
            other_atoms = []
            
            for atom_str in atoms:
                if focus_tracker_id is not None and f"obj{focus_tracker_id}" in atom_str:
                    focus_atoms.append(atom_str)
                else:
                    other_atoms.append(atom_str)
            
            # Display focus atoms first in RED
            max_atoms_to_show = 15
            atoms_shown = 0
            
            for atom_str in focus_atoms:
                if atoms_shown >= max_atoms_to_show or y > TOTAL_H - 30:
                    break
                    
                # Wrap long atom strings
                if len(atom_str) > 30:
                    atom_str = atom_str[:27] + "..."
                
                cv2.putText(img, f"  {atom_str}", (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, DARK_RED, 1)
                y += int(line_height * 0.7)
                atoms_shown += 1
            
            # Display other atoms in white
            for atom_str in other_atoms:
                if atoms_shown >= max_atoms_to_show or y > TOTAL_H - 30:
                    break
                    
                # Wrap long atom strings
                if len(atom_str) > 30:
                    atom_str = atom_str[:27] + "..."
                
                cv2.putText(img, f"  {atom_str}", (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1)
                y += int(line_height * 0.7)
                atoms_shown += 1
            
            if len(atoms) > atoms_shown:
                cv2.putText(img, f"  ... +{len(atoms) - atoms_shown} more", (int(x) + 5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1)
                
    else:
        cv2.putText(img, "No Segment Info", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)

def visualize_gaze_segments(json_file, output_video):
    print(f"Loading data from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    frame_data = data["frame_data"]
    segments = data["segments"]
    
    if not frame_data:
        printf("No frame data found.")
        return

    # Sort frame data by ID
    frame_data.sort(key=lambda x: x["frame_id"])
    
    # Map frame_id to segment
    frame_to_seg = {}
    for seg in segments:
        for fid in range(seg["start_frame"], seg["end_frame"] + 1):
            frame_to_seg[fid] = seg
            
    # Video Writer (20 FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (TOTAL_W, TOTAL_H))
    
    print(f"Generating video to {output_video}...")
    
    # Map frame_id to frame data for atom lookup
    frame_id_to_data = {fd["frame_id"]: fd for fd in frame_data}
    
    for i, fd in enumerate(tqdm(frame_data)):
        img_path = fd.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue
            
        # Get current frame ID and segment
        fid = fd["frame_id"]
        current_seg = frame_to_seg.get(fid)

        # Load and resize image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        
        # Create canvas
        canvas = np.zeros((TOTAL_H, TOTAL_W, 3), dtype=np.uint8)
        canvas[:, :IMG_W] = img
        
        # Draw gaze point
        gaze = fd.get("gaze")
        if gaze:
            gx, gy = int(gaze[0] * SCALE), int(gaze[1] * SCALE)
            cv2.circle(canvas, (gx, gy), 5, BLUE, -1)
            cv2.circle(canvas, (gx, gy), 50, (0, 0, 255), 2)
            
        # ================================================
        # DETERMINE WHICH OBJECTS TO HIGHLIGHT
        # ================================================
        focus_id = -1
        waver_ids = []
        
        if current_seg:
            # Parse segment focus: "category(ID)" -> extract ID
            focus_str = current_seg.get("focus", "")
            if focus_str and "(" in focus_str and ")" in focus_str:
                try:
                    focus_id = int(focus_str.split('(')[1].split(')')[0])
                except:
                    pass
            
            # Check if we're in a waver period for this frame
            wavers = current_seg.get("wavers", [])
            for w in wavers:
                waver_start = w.get("frame", 0)
                waver_duration = w.get("duration", 0)
                waver_end = waver_start + waver_duration
                
                if waver_start <= fid < waver_end:
                    # We're in this waver period
                    waver_str = w.get("waver_focus", "")
                    if waver_str and "(" in waver_str and ")" in waver_str:
                        try:
                            waver_id = int(waver_str.split('(')[1].split(')')[0])
                            waver_ids.append(waver_id)
                        except:
                            pass
        
        # ================================================
        # DRAW ALL OBJECTS WITH APPROPRIATE COLORS
        # ================================================
        all_objs = fd.get("objects", [])
        for obj in all_objs:
            oid = obj.get("id", -1)
            
            # Determine color and thickness based on object type
            if oid == focus_id:
                # FOCUS OBJECT -> THICK RED BOX
                color = DARK_RED
                thickness = 5
            elif oid in waver_ids:
                # WAVER OBJECT -> GREEN BOX
                color = GREEN
                thickness = 2
            else:
                # OTHER OBJECTS -> BLUE BOX
                color = BLUE
                thickness = 1
            
            # Draw bounding box
            ox, oy, w, h = obj.get("xywh", [0, 0, 0, 0])
            sx, sy, sw, sh = int(ox*SCALE), int(oy*SCALE), int(w*SCALE), int(h*SCALE)
            cv2.rectangle(canvas, (sx, sy), (sx+sw, sy+sh), color, thickness)
            
            # Draw category label
            cv2.putText(canvas, obj.get("category", "?"), (sx, sy-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Add atoms to current_seg for sidebar display
        if current_seg and fid in frame_id_to_data:
            current_seg_copy = current_seg.copy()
            current_seg_copy["atoms"] = frame_id_to_data[fid].get("atoms", [])
        else:
            current_seg_copy = current_seg

        # Draw sidebar
        draw_sidebar(canvas, fid, current_seg_copy, segments)
        
        out.write(canvas)
        
    out.release()
    print("Video saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, default="gaze_goals_verification.json")
    parser.add_argument("--output_video", type=str, default="gaze_segmentation_video.mp4")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"File {args.json_file} not found.")
    else:
        visualize_gaze_segments(args.json_file, args.output_video)
