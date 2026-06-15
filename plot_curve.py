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
        col_names = ['percentage', 'mean_reward', 'std_reward', 'model']
        df = pd.read_csv(csv_path, names=col_names, skiprows=1)
    else:
        df = pd.read_csv(csv_path)
 
    return df
 
 
def plot_learning_curve_models(csv_path, output_path, window_size=1, hlines=None):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
 
    df = load_csv(csv_path)
 
    required = ['percentage', 'mean_reward', 'model']
    for col in required:
        if col not in df.columns:
            print(f"Error: CSV must contain {required} columns. Found: {df.columns.tolist()}")
            return
 
    df['percentage'] = pd.to_numeric(df['percentage'], errors='coerce')
    df['mean_reward'] = pd.to_numeric(df['mean_reward'], errors='coerce')
    if 'std_reward' in df.columns:
        df['std_reward'] = pd.to_numeric(df['std_reward'], errors='coerce')
 
    plt.figure(figsize=(12, 7))
    
    #divide std_reward by 5
    df['std_reward'] = df['std_reward'] / 5
    models = df['model'].unique()
    print(models)
    
    MODEL_COLORS = {
        'BC': 'blue',
        'AGIL': 'orange',
        'BC+Mask': 'purple',
        'NSFR': 'red',
        'GRAIL': 'green'
    }
    cmap = plt.get_cmap('tab10')
 
    for i, model_name in enumerate(models):
        mask = df['model'] == model_name
        g = df[mask].sort_values('percentage')
        g_clean = g.dropna(subset=['percentage', 'mean_reward'])
        if g_clean.empty:
            continue
 
        color = MODEL_COLORS.get(model_name, cmap(i % 10))
        
        # Apply smoothing
        smoothed_reward = g_clean['mean_reward'].rolling(window=window_size, min_periods=1).mean()
 
        # Calculate percentages for x-axis
        max_traj = g_clean['percentage'].max()
        if max_traj == 0:
            max_traj = 1
        traj_percent = (g_clean['percentage'] / max_traj) * 100
 
        plt.plot(
            traj_percent, smoothed_reward,
            marker=None, linestyle='-', color=color, label=str(model_name)
        )
 
        if 'std_reward' in df.columns:
            std = g_clean['std_reward'].fillna(0)
            smoothed_std = std.rolling(window=window_size, min_periods=1).mean()
            plt.fill_between(
                traj_percent,
                smoothed_reward - smoothed_std,
                smoothed_reward + smoothed_std,
                color=color, alpha=0.2
            )
 
    # Horizontal reference lines (methods with no learning curve, full-data only)
    if hlines:
        for label, value in hlines:
            color = MODEL_COLORS.get(label.split(' ')[0], 'gray')
            plt.axhline(y=value, linestyle='--', color=color, alpha=0.8, label=label)

    plt.xticks(range(10, 101, 10), [f"{x}%" for x in range(10, 101, 10)])
    plt.xlim(10, 100)
    plt.ylim(bottom=0)
 
    plt.title('Learning Curve by Model')
    plt.xlabel('Percentage of Dataset (%)')
    plt.ylabel('Mean Reward')
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
    parser.add_argument("--hlines", type=str, default=None,
                        help="Comma-sep horizontal reference lines 'Label:value', e.g. "
                             "'NSFR:25.38,GRAIL:25.38' (for full-data-only methods).")
    args = parser.parse_args()
    hlines = None
    if args.hlines:
        hlines = [(s.rsplit(':', 1)[0], float(s.rsplit(':', 1)[1]))
                  for s in args.hlines.split(',')]
    plot_learning_curve_models(args.csv_path, args.output, args.window_size, hlines)
 
 
if __name__ == "__main__":
    main()
 
 