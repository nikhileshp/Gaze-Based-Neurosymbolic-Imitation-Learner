import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import os
import sys
from ocatari.core import OCAtari
import ocatari.vision.utils as utils

# Configuration
CSV_FILE = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/train.csv"
BASE_IMAGE_DIR = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/data/seaquest/gaze_data_tmp"
OUTPUT_IMAGE = "error_case_visualization.png"

# Monkeypatch match_objects to catch the error and return objects anyway
original_match_objects = utils.match_objects

def patched_match_objects(objects, objects_bb, max_obj, max_dist, ObjClass):
    if len(objects_bb) > max_obj:
        print(f"FOUND ERROR CASE: Detected {len(objects_bb)} objects of type {ObjClass.__name__} (Max: {max_obj})")
        # We want to visualize these objects.
        # objects_bb is a list of bounding boxes (x, y, w, h)
        # We can't easily return them through the normal flow because it expects to fill 'objects' list
        # which might have fixed size or logic.
        # But we can store them in a global variable or attribute to access later.
        global detected_overflow_objects
        detected_overflow_objects = objects_bb
        global overflow_class
        overflow_class = ObjClass.__name__
        
        # To allow the script to continue and finish this frame (so we can visualize), 
        # we can truncate the list to the max allowed.
        objects_bb = objects_bb[:max_obj]
        
    return original_match_objects(objects, objects_bb, max_obj, max_dist, ObjClass)

utils.match_objects = patched_match_objects

detected_overflow_objects = None
overflow_class = None

def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_FILE)
    
    print("Initializing OCAtari...")
    oc = OCAtari("Seaquest", mode="vision", render_mode="rgb_array")
    
    print("Searching for error case...")
    for idx, row in df.iterrows():
        traj_folder = row['trajectory']
        img_name = f"{row['frameid']}.png"
        img_path = os.path.join(BASE_IMAGE_DIR, traj_folder, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_array = np.array(image)
        except Exception:
            continue

        # Mock ALE
        if hasattr(oc, "_env") and hasattr(oc._env, "unwrapped") and hasattr(oc._env.unwrapped, "ale"):
            real_ale = oc._env.unwrapped.ale
            
            class MockALE:
                def __getattr__(self, name):
                    return getattr(real_ale, name)
                def getScreenRGB(self, *args):
                    return image_array
            
            oc._env.unwrapped.ale = MockALE()
            
            global detected_overflow_objects
            detected_overflow_objects = None
            
            try:
                oc.detect_objects()
            except Exception as e:
                # If it still crashes, we might have caught it in the patch
                pass
            finally:
                oc._env.unwrapped.ale = real_ale
                
            if detected_overflow_objects is not None:
                print(f"Visualizing error case from image: {img_path}")
                
                # Draw all detected objects (including the ones that caused overflow)
                draw = ImageDraw.Draw(image)
                
                # Draw normal objects detected by OCAtari
                for obj in oc.objects:
                    x, y = obj.xy
                    w, h = obj.wh
                    draw.rectangle([x, y, x+w, y+h], outline="green", width=2)
                    draw.text((x, y-10), obj.category, fill="green")
                
                # Draw the overflow objects (raw bounding boxes)
                # These are (x, y, w, h) tuples
                for bb in detected_overflow_objects:
                    x, y, w, h = bb
                    draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
                    draw.text((x, y+h), f"OVERFLOW {overflow_class}", fill="red")
                
                image.save(OUTPUT_IMAGE)
                print(f"Saved visualization to {OUTPUT_IMAGE}")
                break
                
        if idx > 1000: # Safety break if not found quickly
             print("Checked 1000 frames, no error found yet.")
             # break # Keep going, the user said there ARE warnings

if __name__ == "__main__":
    main()
