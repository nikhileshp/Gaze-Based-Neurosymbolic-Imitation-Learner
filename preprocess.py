import os
import glob
import pandas as pd
import argparse

def parse_trajectory_file(txt_file, trajectory_number):
    """
    Manually parse the trajectory .txt file.
    Handles variable number of commas in the gaze_positions field.
    Filters out actions >= 6.
    Adds sequential trajectory_number.
    """
    try:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {txt_file}: {e}")
        return None
    
    if not lines:
        return None
        
    data = []
    # Header: frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions
    for i, line in enumerate(lines):
        if i == 0:
            continue # Skip header
            
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
            
        try:
            frame_id = parts[0]
            episode_id = parts[1]
            score = parts[2]
            duration = parts[3]
            unclipped_reward = parts[4]
            action = int(parts[5])
            
            # Action filtering
            if action >= 6:
                continue
                
            # Variable gaze positions
            gaze_positions = ",".join(parts[6:])
            
            data.append({
                'frame_id': frame_id,
                'episode_id': episode_id,
                'score': score,
                'duration': duration,
                'unclipped_reward': unclipped_reward,
                'action': action,
                'gaze_positions': gaze_positions,
                'trajectory_number': trajectory_number
            })
        except ValueError:
            continue
            
    if not data:
        return None
        
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Consolidate Atari trajectories into a single train.csv.")
    parser.add_argument("--input_folder", type=str, default="data/seaquest/trajectories", help="Path to input .txt files.")
    parser.add_argument("--output_file", type=str, default="data/seaquest/train.csv", help="Path to save combined CSV.")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        return

    txt_files = glob.glob(os.path.join(args.input_folder, "*.txt"))
    txt_files.sort() # Ensure deterministic order for trajectory numbers
        
    print(f"Found {len(txt_files)} trajectory file(s).")
    
    all_dfs = []
    for i, txt_file in enumerate(txt_files):
        traj_num = i + 1
        traj_filename = os.path.basename(txt_file)
        
        print(f"Processing ({traj_num}/{len(txt_files)}): {traj_filename}")
        
        df = parse_trajectory_file(txt_file, traj_num)
        
        if df is not None and not df.empty:
            all_dfs.append(df)
            
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save to CSV
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        combined_df.to_csv(args.output_file, index=False)
        print(f"\nSuccessfully saved {len(combined_df)} records to '{args.output_file}'.")
    else:
        print("\nNo valid data found to save.")

if __name__ == "__main__":
    main()
