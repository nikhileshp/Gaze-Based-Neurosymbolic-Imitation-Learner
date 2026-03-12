import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def load_csv(csv_path):
    """
    Handles CSVs where an extra unnamed/None column sits between
    'trajectory' and 'mean_reward' (5 data columns, 4 header columns).
    Falls back to a normal read if the column count already matches.
    """
    with open(csv_path, "r") as f:
        header = f.readline()
    n_header_cols = len(header.strip().split(","))

    with open(csv_path, "r") as f:
        f.readline()  # skip header
        first_data = f.readline()
    n_data_cols = len(first_data.strip().split(","))

    if n_data_cols > n_header_cols:
        col_names = ['trajectory', '_extra', 'mean_reward', 'std_reward', 'model']
        df = pd.read_csv(csv_path, names=col_names, skiprows=1)
        df = df.drop(columns=['_extra'], errors='ignore')
    else:
        df = pd.read_csv(csv_path)

    return df


def plot_learning_curve_models(csv_path, output_path, window_size=1):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = load_csv(csv_path)

    required = ['trajectory', 'mean_reward', 'model']
    for col in required:
        if col not in df.columns:
            print(f"Error: CSV must contain {required} columns. Found: {df.columns.tolist()}")
            return

    df['trajectory'] = pd.to_numeric(df['trajectory'], errors='coerce')
    df['mean_reward'] = pd.to_numeric(df['mean_reward'], errors='coerce')
    if 'std_reward' in df.columns:
        df['std_reward'] = pd.to_numeric(df['std_reward'], errors='coerce')

    plt.figure(figsize=(12, 7))

    models = df['model'].unique()
    print(models)
    # Custom color list: first is green, second is red. The rest are other distinct colors from tab10.
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, model_name in enumerate(models):
        mask = df['model'] == model_name
        g = df[mask].sort_values('trajectory')
        g_clean = g.dropna(subset=['trajectory', 'mean_reward'])
        if g_clean.empty:
            continue

        if not (g_clean['trajectory'] == 0).any():
            zero_row = pd.DataFrame({'trajectory': [0.0], 'mean_reward': [0.0]})
            if 'std_reward' in df.columns:
                zero_row['std_reward'] = [0.0]
            g_clean = pd.concat([zero_row, g_clean], ignore_index=True).sort_values('trajectory')

        color = colors[i % len(colors)]
        
        # Apply smoothing
        smoothed_reward = g_clean['mean_reward'].rolling(window=window_size, min_periods=1).mean()

        # Calculate percentages for x-axis
        max_traj = g_clean['trajectory'].max()
        if max_traj == 0:
            max_traj = 1
        traj_percent = (g_clean['trajectory'] / max_traj) * 100

        plt.plot(
            traj_percent, smoothed_reward,
            marker=None, linestyle='-', color=color, label=str(model_name)
        )

        if 'std_reward' in df.columns:
            std = g_clean['std_reward'].fillna(0)/5.0
            smoothed_std = std.rolling(window=window_size, min_periods=1).mean()
            plt.fill_between(
                traj_percent,
                smoothed_reward - smoothed_std,
                smoothed_reward + smoothed_std,
                color=color, alpha=0.2
            )

    plt.xticks(range(0, 101, 10), [f"{x}%" for x in range(0, 101, 10)])
    plt.xlim(0, 100)
    plt.ylim(0, 6500)


    plt.xlabel('Percentage of Dataset (%)')
    plt.ylabel('Mean Game Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Learning curve plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot learning curve for multiple models from a results CSV file."
    )
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to the results CSV file.")
    parser.add_argument("--output", type=str, default="learning_curve_models.png",
                        help="Path to save the output plot.")
    parser.add_argument("--window_size", type=int, default=1,
                        help="Window size for smoothing the learning curves (default: 1).")
    args = parser.parse_args()
    plot_learning_curve_models(args.csv_path, args.output, args.window_size)


if __name__ == "__main__":
    main()
