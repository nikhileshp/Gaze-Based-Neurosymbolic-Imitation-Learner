import json
import numpy as np

try:
    with open("gaze_goals_verification.json") as f:
        data = json.load(f)
        
    count_diff = 0
    count_movement = 0
    max_diff = 0
    
    for seg in data:
        p_start = seg.get("player_loc_start")
        p_end = seg.get("player_loc_end")
        
        if p_start and p_end and p_start != p_end:
            count_diff += 1
            dist = np.sqrt((p_start[0]-p_end[0])**2 + (p_start[1]-p_end[1])**2)
            if dist > max_diff:
                max_diff = dist
                
        if seg.get("movement", "none") != "none":
            count_movement += 1
                
    print(f"Total Segments: {len(data)}")
    print(f"Segments with different start/end player loc: {count_diff}")
    print(f"Max distance moved in a segment: {max_diff}")
    print(f"Segments with movement != none: {count_movement}")
    
    # Check first few goals for format
    print("Sample goals:")
    for i in range(min(5, len(data))):
        print(f"  {data[i].get('goal')}")

except FileNotFoundError:
    print("File gaze_goals_verification.json not found.")
