import json
import cv2
import os
import argparse
import sys
from tqdm import tqdm

def visualize(json_file, output_video, max_frames=500):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    frame_data = data["frame_data"]
    
    if not frame_data:
        print("No frame data found.")
        return

    # sort by frame_id just in case
    frame_data.sort(key=lambda x: x["frame_id"])
    
    # Get first image to determine size
    first_img_path = frame_data[0]["image_path"]
    if not os.path.exists(first_img_path):
        print(f"Image output path not found: {first_img_path}")
        return
        
    img = cv2.imread(first_img_path)
    height, width, _ = img.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))
    
    print(f"Creating video {output_video} ({width}x{height})...")
    
    for i, fd in tqdm(enumerate(frame_data[:max_frames])):
        img_path = fd["image_path"]
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        
        # Draw Objects
        for obj in fd["objects"]:
            x, y, w, h = obj["xywh"]
            cat = obj["category"]
            oid = obj["id"]
            
            # Color based on category?
            color = (0, 255, 0) # Green
            if cat == "Player": color = (255, 0, 0) # Blue
            if "Enemy" in cat or "Shark" in cat or "Submarine" in cat: color = (0, 0, 255) # Red
            
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, 1)
            cv2.putText(img, f"{oid}:{cat}", (int(x), int(y)-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
        # Draw Gaze
        gaze = fd["gaze"]
        gx, gy = gaze
        # Draw crosshair
        cv2.circle(img, (int(gx), int(gy)), 3, (0, 255, 255), -1) # Yellow Dot
        # Fixation margin
        cv2.circle(img, (int(gx), int(gy)), 15, (0, 255, 255), 1) 
        
        # Draw Info
        fix_obj = fd["fixated_object"]
        fix_id = fd["fixated_object_id"]
        cv2.putText(img, f"Frame: {fd['frame_id']}", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f"Fixation: {fix_obj} ({fix_id})", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        out.write(img)
        
    out.release()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--out", type=str, default="debug_gaze.mp4")
    args = parser.parse_args()
    
    visualize(args.json, args.out)
