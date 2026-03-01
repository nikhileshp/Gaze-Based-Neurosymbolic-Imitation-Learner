from ocatari.core import OCAtari
from PIL import Image, ImageDraw
import numpy as np
import sys

IMAGE_PATH = "/home/nikhilesh/Projects/NeSY-Imitation-Learning/data/seaquest/gaze_data_tmp/237_RZ_9656617_Feb-08-14-12-21/RZ_9656617_13767.png"
OUTPUT_PATH = "detection_result.png"

def main():
    print(f"Loading image from {IMAGE_PATH}")
    try:
        image = Image.open(IMAGE_PATH).convert('RGB')
        image_array = np.array(image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("Initializing OCAtari...")
    oc = OCAtari("Seaquest", mode="vision", render_mode="rgb_array")
    
    # Mock ALE
    if hasattr(oc, "_env") and hasattr(oc._env, "unwrapped") and hasattr(oc._env.unwrapped, "ale"):
        real_ale = oc._env.unwrapped.ale
        
        class MockALE:
            def __getattr__(self, name):
                return getattr(real_ale, name)
                
            def getScreenRGB(self, *args):
                return image_array
        
        oc._env.unwrapped.ale = MockALE()
        
        try:
            print("Detecting objects...")
            oc.detect_objects()
        except ValueError as e:
            print(f"Warning: Detection limit reached: {e}")
        finally:
            oc._env.unwrapped.ale = real_ale
    else:
        print("Could not mock ALE structure.")
        return

    objects = oc.objects
    print(f"Detected {len(objects)} objects:")
    for obj in objects:
        print(f"- {obj}")

    # Visualize
    draw = ImageDraw.Draw(image)
    for obj in objects:
        # OCAtari objects usually have x, y, w, h or xywh
        # Let's check attributes
        x, y = obj.xy
        w, h = obj.wh
        
        # Draw rectangle
        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
        draw.text((x, y-10), obj.category, fill="red")

    image.save(OUTPUT_PATH)
    print(f"Saved visualization to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
