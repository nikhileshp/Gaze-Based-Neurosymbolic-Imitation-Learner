from ocatari.core import OCAtari
import numpy as np
from PIL import Image
import torch

def test_vision_extraction():
    # Create dummy image (210, 160, 3)
    image = np.zeros((210, 160, 3), dtype=np.uint8)
    
    # Initialize OCAtari in vision mode
    try:
        env = OCAtari("Seaquest", mode="vision", render_mode="rgb_array")
        env.reset()
        
        # OCAtari usually works by stepping the environment.
        # But we want to detect from a static image.
        # env.detect_objects(image_array) might exist?
        # Or we might need to rely on the internal detector.
        
        # Let's check available methods
        print("OCAtari initialized.")
        if hasattr(env, "detect_objects"):
            print("Has detect_objects method.")
        else:
            print("No detect_objects method found.")
            
        # Try to use the underlying vision method if possible
        # env._env.unwrapped.ale.setRAM(...) is for RAM mode.
        
    except Exception as e:
        print(f"Error initializing OCAtari: {e}")

if __name__ == "__main__":
    test_vision_extraction()
  