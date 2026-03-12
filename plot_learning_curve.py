import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_learning_curve(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Read the results
    df = pd.read_csv(csv_path)

    # Required columns
    if 'trajectory' not in df.columns or 'mean_reward' not in df.columns:
        print(f"Error: CSV must contain 'trajectory' and 'mean_reward' columns. Found: {df.columns.tolist()}")
        return

    plt.figure(figsize=(10, 6))

    if 'gaze' in df.columns:
        # Plot one line per gaze condition
        groups = {
            True:  ('With Gaze',    'tab:blue'),
            False: ('Without Gaze', 'tab:orange'),
        }
        for gaze_val, (label, color) in groups.items():
            # Support both string ('True'/'False') and boolean values
            mask = df['gaze'].astype(str).str.lower() == str(gaze_val).lower()
            g = df[mask].sort_values('trajectory')
            if g.empty:
                continue
            plt.plot(g['trajectory'], g['mean_reward'],
                     marker='o', linestyle='-', color=color, label=label)
            if 'std_reward' in df.columns:
                plt.fill_between(g['trajectory'],
                                 g['mean_reward'] - g['std_reward'],
                                 g['mean_reward'] + g['std_reward'],
                                 color=color, alpha=0.2)
        # Integer x-ticks based on the union of trajectories
        all_trajs = df['trajectory'].sort_values().unique()
        if len(all_trajs) <= 20:
            plt.xticks(all_trajs)
    else:
        # Original single-line behaviour
        plt.plot(df['trajectory'], df['mean_reward'],
                 marker='o', linestyle='-', color='tab:blue', label='Mean Reward')
        if 'std_reward' in df.columns:
            plt.fill_between(df['trajectory'],
                             df['mean_reward'] - df['std_reward'],
                             df['mean_reward'] + df['std_reward'],
                             color='tab:blue', alpha=0.2, label='Standard Deviation')
        if len(df['trajectory']) <= 20:
            plt.xticks(df['trajectory'])

    plt.title('Learning Curve (Iterative Trajectory Training)')
    plt.xlabel('Trajectory')
    plt.ylabel('Mean Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Learning curve plot saved to {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot learning curve from a results CSV file.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the results CSV file.")
    parser.add_argument("--output", type=str, default="learning_curve.png", help="Path to save the output plot (default: learning_curve.png).")
    args = parser.parse_args()
    
    plot_learning_curve(args.csv_path, args.output)

if __name__ == "__main__":
    main()
