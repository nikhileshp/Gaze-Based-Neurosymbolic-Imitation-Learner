import os
import glob
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Combine trajectory .txt files into a single CSV dataset.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to folder containing .txt trajectory files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output CSV file (e.g., train.csv).")
    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        return

    txt_files = glob.glob(os.path.join(args.input_folder, "*.txt"))
    print(f"Found {len(txt_files)} .txt files in '{args.input_folder}'.")

    if not txt_files:
        print("No .txt files found.")
        return

    dfs = []
    for txt_file in txt_files:
        try:
            # Manually parse the file because gaze_positions contains variable number of commas
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                continue
                
            # Header is expected to be: frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions
            # We will use hardcoded column names based on the known format to be safe
            columns = ['frame_id', 'episode_id', 'score', 'duration', 'unclipped_reward', 'action', 'gaze_positions']
            
            data = []
            for i, line in enumerate(lines):
                if i == 0:
                    continue # Skip header
                
                parts = line.strip().split(',')
                if len(parts) < 6:
                    # Skip malformed lines
                    continue
                    
                # Extract fixed fields
                frame_id = parts[0]
                episode_id = parts[1]
                score = parts[2]
                duration = parts[3]
                unclipped_reward = parts[4]
                action = parts[5]
                
                # Extract variable gaze positions (everything after action)
                # Join them back into a string
                gaze_positions = ",".join(parts[6:])
                
                data.append({
                    'frame_id': frame_id,
                    'episode_id': episode_id,
                    'score': score,
                    'duration': duration,
                    'unclipped_reward': unclipped_reward,
                    'action': action,
                    'gaze_positions': gaze_positions
                })
            
            if data:
                df = pd.DataFrame(data)
                dfs.append(df)
                print(f"Processed {len(df)} records from {os.path.basename(txt_file)}")
            
        except Exception as e:
            print(f"Error reading file {txt_file}: {e}")

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Ensure 'frame_id' is present (critical for ExpertDataset)
        if 'frame_id' not in combined_df.columns:
             print("Warning: 'frame_id' column missing from combined dataset!")
        
        # Fill missing values for robustness
        # combined_df.fillna("", inplace=True)
        
        # Save to CSV
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        combined_df.to_csv(args.output_file, index=False)
        print(f"Successfully saved combined dataset with {len(combined_df)} records to '{args.output_file}'.")
    else:
        print("No data frames to combine.")

if __name__ == "__main__":
    main()
