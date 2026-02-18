import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import os

# Configuration
INPUT_FILE = "labeled_clustered_segments.json"
OUTPUT_FILE = "object_centric_intents.json"
PLOT_DIR = "out/clusters"

# Entity Merging Map
# Map specific object names to General Types
ENTITY_MAP = {
    "shark": "Enemy",
    "submarine": "Enemy",
    "fish": "Fish",
    "diver": "Diver",
    "oxygen_bar": "Oxygen",
    "surface": "Surface",
    "bottom": "Bottom",
    "player": "Player" # Self-focus?
}

def get_general_type(focus_str):
    """
    Parses 'shark(1)' -> 'Enemy'
    """
    # Remove parens: shark(1) -> shark
    base_name = focus_str.split('(')[0]
    
    # Map to general type
    return ENTITY_MAP.get(base_name, "Other")

def extract_trajectory_features(segment):
    """
    Extracts relative trajectory features for clustering.
    We need fixed-size features. 
    1. Start Distance vs End Distance (Approach/Retreat)
    2. Delta X, Delta Y (Direction)
    """
    # Get Player Positions
    p_start = np.array(segment.get("player_loc_start", [0,0]))
    p_end = np.array(segment.get("player_loc_end", [0,0]))
    
    # Get Gaze/Object Positions (Approximation of Object Location)
    # We assume 'gaze_loc' roughly tracks the object
    o_start = np.array(segment.get("gaze_loc_start", [0,0]))
    o_end = np.array(segment.get("gaze_loc_end", [0,0]))
    
    # Features
    # 1. Distance Change: (EndDist - StartDist)
    # Negative = Approach, Positive = Retreat
    dist_start = np.linalg.norm(p_start - o_start)
    dist_end = np.linalg.norm(p_end - o_end)
    dist_delta = dist_end - dist_start
    
    # 2. Movement Vector relative to Object
    # We want to know if we are moving towards/away/parallel
    # This is captured by dist_delta mainly.
    
    # 3. Vertical vs Horizontal Movement (Is it an evade up/down?)
    # Delta of Player
    p_delta = p_end - p_start
    
    return [dist_delta, p_delta[0], p_delta[1]]

def main():
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        
    segments = data['segments']
    print(f"Loaded {len(segments)} segments.")
    
    # 1. Group by Focus Type
    groups = defaultdict(list)
    
    for i, seg in enumerate(segments):
        focus = seg.get('focus', 'None')
        g_type = get_general_type(focus)
        
        # Filter "Other" or "None" if desired, but let's keep relevant ones
        if g_type in ["Enemy", "Diver", "Oxygen", "Surface", "Fish"]:
            groups[g_type].append(seg)
            
    print(f"\nGroups found: {list(groups.keys())}")
    for k, v in groups.items():
        print(f"  {k}: {len(v)} segments")

    # 2. Cluster each Group
    clustered_segments = []
    
    for g_type, group_segs in groups.items():
        print(f"\nClustering Group: {g_type} ({len(group_segs)} segments)...")
        
        # Extract Features
        X = []
        valid_indices = []
        for i, seg in enumerate(group_segs):
            try:
                feats = extract_trajectory_features(seg)
                X.append(feats)
                valid_indices.append(i)
            except:
                continue
                
        X = np.array(X)
        
        if len(X) < 2:
            print("  Skipping (too few samples)")
            continue
            
        # Hierarchical Clustering
        # We start with 2 clusters (Approach vs Retreat/Other)
        # Or let it find optimal? Let's try 3 to capture (Approach, Retreat, Static/Parallel)
        n_clusters = 3
        if len(X) < 3: n_clusters = len(X)
            
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
        labels = clustering.labels_
        
        # Analyze Clusters to Name them?
        # For now, just label {Type}_{ID}
        
        # Plot
        plt.figure(figsize=(10, 6))
        # Plot DistanceDelta vs VerticalMovement
        scatter = plt.scatter(X[:, 0], X[:, 2], c=labels, cmap='viridis', alpha=0.6)
        plt.xlabel("Distance Change (Neg=Approach, Pos=Retreat)")
        plt.ylabel("Vertical Movement (Neg=Up, Pos=Down)") # Note: Y IS INVERTED IN GAMES OFTEN, CHECK COORDS
        plt.title(f"Clustering traits for {g_type}")
        plt.grid(True)
        plt.savefig(f"{PLOT_DIR}/cluster_{g_type}.png")
        plt.close()
        
        # Assign Labels back to segments
        for idx, label in zip(valid_indices, labels):
            seg = group_segs[idx]
            seg['intent_id'] = int(label)
            seg['intent_label'] = f"{g_type}_{label}"
            clustered_segments.append(seg)
            
    # Save Result
    print(f"\nSaving {len(clustered_segments)} clustered segments to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({"segments": clustered_segments}, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
