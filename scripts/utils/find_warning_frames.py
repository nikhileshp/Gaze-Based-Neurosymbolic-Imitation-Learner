from ocatari.core import OCAtari
import numpy as np
from PIL import Image, ImageDraw
import os
import traceback

def find_warning_frames():
    print("Initializing OCAtari with Seaquest...")
    try:
        env = OCAtari("Seaquest", mode="vision", render_mode="rgb_array")
        obs, _ = env.reset()
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return

    print("Starting simulation to find warning frames...")
    
    warning_frames_found = 0
    max_warning_frames = 2
    
    # Run for a sufficient number of steps to encounter complex scenes
    for i in range(10000):
        if i % 1000 == 0:
            print(f"Step {i}...")
        try:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Explicitly check for object count if no error was raised but we want to be sure
            # (Some versions might just print a warning and not raise)
            if hasattr(env, "objects") and len(env.objects) > 50: # Arbitrary high number check
                 print(f"High object count detected at step {i}: {len(env.objects)}")
            
            if terminated or truncated:
                env.reset()
                
        except ValueError as e:
            # This catches the specific error we expect when object limit is exceeded
            print(f"WARNING: FRAME {i} excessive missiles detected")
            
            # Save the frame
            filename = f"warning_frame_{warning_frames_found + 1}.png"
            try:
                print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")
                if obs.ndim == 3 and obs.shape[2] == 3:
                    img_data = obs.astype(np.uint8)
                else:
                    # Try to get image from render if obs is not it
                    print("Observation might not be an image. Trying env.render()")
                    img_data = env.render()
                    if img_data is None:
                         print("env.render() returned None.")
                         # Try to use the ale screen if available
                         if hasattr(env, "ale"):
                             img_data = env.ale.getScreenRGB()
                    
                if img_data is not None:
                    img = Image.fromarray(img_data)
                    
                    # Annotate objects
                    try:
                        draw = ImageDraw.Draw(img)
                        missile_count = 0
                        for obj in env.objects:
                            if type(obj).__name__ == "EnemyMissile":
                                missile_count += 1
                                x, y = obj.xy
                                w, h = obj.wh
                                draw.rectangle([x, y, x+w, y+h], outline="red", width=1)
                                draw.text((x, y-10), str(missile_count), fill="red")
                        print(f"Annotated {missile_count} EnemyMissile objects.")
                    except Exception as annot_err:
                        print(f"Error during annotation: {annot_err}")

                    img.save(filename)
                    print(f"Saved {filename}")
                    warning_frames_found += 1
                else:
                    print("Could not obtain image data.")
            except Exception as save_err:
                print(f"Could not save frame: {save_err}")
                traceback.print_exc()
            
            if warning_frames_found >= max_warning_frames:
                print("Found enough warning frames. Exiting.")
                break
                
            # Reset might be needed if the state is corrupted, but usually we can continue or reset
            env.reset()
            
        except Exception as e:
            print(f"Unexpected error at step {i}: {e}")
            traceback.print_exc()
            break
            
    env.close()

if __name__ == "__main__":
    find_warning_frames()
