import pandas as pd
import json
import argparse
import os

def preprocess(trajectory_path, json_path, output_path):
    print(f"Loading JSON from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    segments = data.get('segments', [])
    print(f"Loaded {len(segments)} segments.")

    print(f"Loading trajectory from {trajectory_path}...")
    # Manual parsing to handle unquoted commas in gaze_positions
    rows = []
    with open(trajectory_path, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                row = parts[:6]
                gaze = ",".join(parts[6:])
                row.append(gaze)
                rows.append(row)
    
    df = pd.DataFrame(rows, columns=header)
    print(f"Total frames in trajectory: {len(df)}")

    # Extract integer frame_id for matching with JSON
    # frame_id is like "RZ_2461867_1" -> we want 1
    def extract_id(fid_str):
        try:
            return int(fid_str.split('_')[-1])
        except (ValueError, IndexError):
            return -1

    df['int_frame_id'] = df['frame_id'].apply(extract_id)

    # Create a mapping from integer frame_id to gaze_positions string
    frame_to_gaze = {}
    for seg in segments:
        gaze_loc = seg.get('gaze_loc_start', seg.get('gaze_loc'))
        if gaze_loc is None:
            continue
        gaze_str = f"{gaze_loc[0]},{gaze_loc[1]}"
        for fid in range(seg['start_frame'], seg['end_frame'] + 1):
            frame_to_gaze[fid] = gaze_str

    print(f"Created map for {len(frame_to_gaze)} frames.")

    # Apply mapping using the integer frame ID
    df['gaze_positions'] = df['int_frame_id'].map(frame_to_gaze)
    
    # Filter out frames without a segment (where gaze_positions is now NaN)
    initial_len = len(df)
    df_out = df.dropna(subset=['gaze_positions']).copy()
    
    # Drop the temporary column
    df_out.drop(columns=['int_frame_id'], inplace=True)
    
    print(f"Filtered trajectory from {initial_len} to {len(df_out)} frames.")
    
    if len(df_out) == 0:
        print("Warning: No frames matched any segments. Output will be empty.")
    
    df_out.to_csv(output_path, index=False)
    print(f"Saved processed trajectory to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess trajectory gaze data using JSON segments.")
    parser.add_argument("--trajectory", type=str, required=True, help="Path to the trajectory TXT file.")
    parser.add_argument("--json", type=str, required=True, help="Path to the gaze goals JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file.")

    args = parser.parse_args()
    
    if not os.path.exists(args.trajectory):
        print(f"Error: Trajectory file {args.trajectory} not found.")
    elif not os.path.exists(args.json):
        print(f"Error: JSON file {args.json} not found.")
    else:
        preprocess(args.trajectory, args.json, args.output)
