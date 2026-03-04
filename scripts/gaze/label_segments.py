import json
import os

def main():
    input_file = 'clustered_segments.json'
    output_file = 'labeled_clustered_segments.json'

    # Manual labels derived from cluster descriptions
    CLUSTER_LABELS = {
        "0": "Player Navigation",
        "1": "Engaging Submarine",
        "2": "Tracking Enemy",
        "3": "Rescuing Diver",
        "4": "Saving Diver",
        "5": "Surfacing",
        "6": "Checking Oxygen",
        "7": "Attacking Shark",
        "8": "Dodging Missile"
    }

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    segments = data.get("segments", [])
    cluster_descriptions = data.get("cluster_descriptions", {})

    print(f"Found {len(segments)} segments and {len(cluster_descriptions)} cluster descriptions.")

    labeled_count = 0
    for segment in segments:
        cluster_id = str(segment.get("cluster_id"))
        
        # Determine label
        label = "Unknown"
        if cluster_id in CLUSTER_LABELS:
            label = CLUSTER_LABELS[cluster_id]
        else:
            print(f"Warning: Cluster ID {cluster_id} has no manual label.")

        segment["label"] = label
        labeled_count += 1

    print(f"Labeled {labeled_count} segments.")

    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()
