"""
Cluster gaze segments to discover high-level goal patterns.

Features:
- focus_object_type (categorical -> one-hot)
- number_of_frames (numeric)
- primitive_action_distribution (normalized vector)
- movement (categorical -> one-hot)
"""

import json
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter
import argparse

def extract_features(segments):
    """
    Extract features from segments for clustering.
    
    Features (21 total):
    - 2 spatial: gaze-player distance, player distance travelled
    - 1 duration: number of frames
    - 9 focus types: one-hot encoded
    - 6 primitive actions: normalized distribution
    - 3 movement types: one-hot encoded
    
    Returns:
        features: numpy array (n_segments, n_features)
        segment_indices: list of segment indices
        feature_names: list of feature column names
    """
    
    # Collect all unique primitive actions and focus types
    all_primitive_actions = set()
    all_focus_types = set()
    
    # Simplified movement categories (only 3)
    all_movements = ["towards_gaze", "away_from_gaze", "none"]
    
    for seg in segments:
        if "primitive_action_counts" in seg:
            all_primitive_actions.update(seg["primitive_action_counts"].keys())
        if "focus" in seg:
            # Extract focus type (before the parenthesis)
            focus = seg["focus"]
            if "(" in focus:
                focus_type = focus.split("(")[0]
            else:
                focus_type = focus
            all_focus_types.add(focus_type)
    
    all_primitive_actions = sorted(list(all_primitive_actions))
    all_focus_types = sorted(list(all_focus_types))
    
    print(f"Found {len(all_focus_types)} focus types: {all_focus_types}")
    print(f"Found {len(all_primitive_actions)} primitive actions: {all_primitive_actions}")
    print(f"Using {len(all_movements)} movement categories: {all_movements}")
    
    # Build feature vectors
    feature_vectors = []
    segment_indices = []
    feature_names = []
    
    # Feature name tracking
    # Spatial features first
    feature_names.append("gaze_player_distance")
    feature_names.append("player_distance_travelled")
    
    # Duration
    feature_names.append("num_frames")
    
    # Add focus type one-hot feature names
    for ft in all_focus_types:
        feature_names.append(f"focus_{ft}")
    
    # Add primitive action distribution feature names
    for action in all_primitive_actions:
        feature_names.append(f"action_{action}")
    
    # Add movement one-hot feature names
    for movement in all_movements:
        feature_names.append(f"movement_{movement}")
    
    for i, seg in enumerate(segments):
        features = []
        
        # 1. Spatial features (Distances only)
        # Player start position (needed for calculation)
        player_loc_start = seg.get("player_loc_start", [80, 105])
        player_x = player_loc_start[0] if player_loc_start else 80
        player_y = player_loc_start[1] if player_loc_start else 105
        
        # Gaze start position (needed for calculation)
        gaze_start = seg.get("gaze_loc_start", seg.get("gaze_loc", [80, 105]))
        gaze_x = gaze_start[0] if gaze_start else 80
        gaze_y = gaze_start[1] if gaze_start else 105
        
        # Absolute distance from gaze to player
        distance = np.sqrt((gaze_x - player_x)**2 + (gaze_y - player_y)**2)
        features.append(distance)
        
        # Player distance travelled (start to end)
        player_loc_end = seg.get("player_loc_end", player_loc_start)
        player_end_x = player_loc_end[0] if player_loc_end else player_x
        player_end_y = player_loc_end[1] if player_loc_end else player_y
        player_travelled = np.sqrt((player_end_x - player_x)**2 + (player_end_y - player_y)**2)
        features.append(player_travelled)
        
        # 2. Number of frames
        num_frames = seg["end_frame"] - seg["start_frame"] + 1
        features.append(num_frames)
        
        # 2. Focus object type (one-hot)
        focus = seg.get("focus", "Explore")
        if "(" in focus:
            focus_type = focus.split("(")[0]
        else:
            focus_type = focus
        
        focus_one_hot = [1 if ft == focus_type else 0 for ft in all_focus_types]
        features.extend(focus_one_hot)
        
        # 3. Primitive action distribution (normalized)
        action_counts = seg.get("primitive_action_counts", {})
        total_actions = sum(action_counts.values())
        
        if total_actions > 0:
            action_dist = [action_counts.get(action, 0) / total_actions 
                          for action in all_primitive_actions]
        else:
            action_dist = [0] * len(all_primitive_actions)
        
        features.extend(action_dist)
        
        # 4. Movement type (simplified to 3 categories, one-hot)
        raw_movement = seg.get("movement", "none")
        
        # Simplify movement categories
        if "towards" in raw_movement:
            movement = "towards_gaze"
        elif "away" in raw_movement:
            movement = "away_from_gaze"
        else:
            movement = "none"
        
        movement_one_hot = [1 if mv == movement else 0 for mv in all_movements]
        features.extend(movement_one_hot)
        
        feature_vectors.append(features)
        segment_indices.append(i)
    
    return np.array(feature_vectors), segment_indices, feature_names

def cluster_segments(features, n_clusters=5, method="kmeans"):
    """
    Cluster segments using specified method.
    
    Args:
        features: numpy array (n_segments, n_features)
        n_clusters: number of clusters
        method: "kmeans" or "hierarchical"
    
    Returns:
        cluster_labels: array of cluster assignments
        clusterer: fitted clustering model
    """
    
    # Standardize features (important for mixed feature types)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(features_scaled)
        
        # Compute silhouette score
        if n_clusters > 1:
            silhouette = silhouette_score(features_scaled, cluster_labels)
            print(f"\nSilhouette Score: {silhouette:.3f}")
    
    elif method == "hierarchical":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(features_scaled)
        
        if n_clusters > 1:
            silhouette = silhouette_score(features_scaled, cluster_labels)
            print(f"\nSilhouette Score: {silhouette:.3f}")
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    return cluster_labels, clusterer, scaler

def analyze_clusters(segments, cluster_labels, n_clusters, features=None):
    """
    Analyze and print cluster characteristics.
    Returns a dictionary of cluster descriptions.
    """
    
    print(f"\n{'='*80}")
    print(f"CLUSTER ANALYSIS ({n_clusters} clusters)")
    print(f"{'='*80}\n")
    
    cluster_descriptions = {}
    
    for cluster_id in range(n_clusters):
        cluster_segments = [seg for i, seg in enumerate(segments) if cluster_labels[i] == cluster_id]
        
        if not cluster_segments:
            continue
            
        description = {
            "size": len(cluster_segments),
            "pct": 100 * len(cluster_segments) / len(segments)
        }
        
        print(f"\n{'─'*80}")
        print(f"CLUSTER {cluster_id} ({len(cluster_segments)} segments)")
        print(f"{'─'*80}")
        
        # Spatial statistics
        if features is not None:
            # Get features for this cluster
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_feats = features[cluster_indices]
            
            print("\nSpatial Statistics (Mean ± SD):")
            
            # Distance Gaze-Player
            dist_mean = np.mean(cluster_feats[:, 0])
            dist_std = np.std(cluster_feats[:, 0])
            print(f"  Gaze-Player Dist:    {dist_mean:5.1f} ± {dist_std:4.1f}")
            
            # Distance Travelled
            trav_mean = np.mean(cluster_feats[:, 1])
            trav_std = np.std(cluster_feats[:, 1])
            
            # Movement Angle
            ang_mean = np.mean(cluster_feats[:, 6]) if cluster_feats.shape[1] > 6 else 0
            ang_std = np.std(cluster_feats[:, 6]) if cluster_feats.shape[1] > 6 else 0
            print(f"  Movement Cos Angle:  {ang_mean:5.2f} ± {ang_std:4.2f}")
            
            description["spatial"] = {
                "gaze_player_dist_mean": float(dist_mean),
                "gaze_player_dist_std": float(dist_std),
                "player_travelled_mean": float(trav_mean),
                "player_travelled_std": float(trav_std),
                "movement_cos_angle_mean": float(ang_mean),
                "movement_cos_angle_std": float(ang_std)
            }
        
        # Focus types
        focus_types = []
        for seg in cluster_segments:
            focus = seg.get("focus", "Explore")
            if "(" in focus:
                focus_type = focus.split("(")[0]
            else:
                focus_type = focus
            focus_types.append(focus_type)
        
        focus_counter = Counter(focus_types)
        print("\nMost Common Focus Types:")
        top_focus = []
        for focus, count in focus_counter.most_common(5):
            pct = 100 * count / len(cluster_segments)
            print(f"  {focus:20s}: {count:4d} ({pct:5.1f}%)")
            top_focus.append({"focus": focus, "count": count, "pct": pct})
        description["top_focus"] = top_focus
        
        # Movement types
        movements = [seg.get("movement", "none") for seg in cluster_segments]
        movement_counter = Counter(movements)
        print("\nMost Common Movements:")
        top_movements = []
        for mv, count in movement_counter.most_common(5):
            pct = 100 * count / len(cluster_segments)
            print(f"  {mv:20s}: {count:4d} ({pct:5.1f}%)")
            top_movements.append({"movement": mv, "count": count, "pct": pct})
        description["top_movements"] = top_movements
        
        # Primitive actions
        all_actions = []
        for seg in cluster_segments:
            action_counts = seg.get("primitive_action_counts", {})
            for action, count in action_counts.items():
                all_actions.extend([action] * count)
        
        action_counter = Counter(all_actions)
        print("\nMost Common Primitive Actions:")
        top_actions = []
        for action, count in action_counter.most_common(5):
            pct = 100 * count / len(all_actions) if all_actions else 0
            print(f"  {action:20s}: {count:4d} ({pct:5.1f}%)")
            top_actions.append({"action": action, "count": count, "pct": pct})
        description["top_actions"] = top_actions
        
        # Duration stats
        durations = [seg["end_frame"] - seg["start_frame"] + 1 for seg in cluster_segments]
        print(f"\nDuration Statistics:")
        print(f"  Mean:   {np.mean(durations):6.1f} frames")
        print(f"  Median: {np.median(durations):6.1f} frames")
        print(f"  Min:    {np.min(durations):6.0f} frames")
        print(f"  Max:    {np.max(durations):6.0f} frames")
        
        description["duration"] = {
            "mean": float(np.mean(durations)),
            "median": float(np.median(durations)),
            "min": float(np.min(durations)),
            "max": float(np.max(durations))
        }
        
        cluster_descriptions[str(cluster_id)] = description
        
    return cluster_descriptions

def find_optimal_k(features, max_k=20):
    """
    Use elbow method and silhouette analysis to find optimal number of clusters.
    """
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    inertias = []
    silhouettes = []
    k_range = range(2, min(max_k + 1, len(features)))
    
    print("\n" + "="*60)
    print("SILHOUETTE SCORE ANALYSIS")
    print("="*60)
    print(f"{'k':<5} {'Silhouette Score':<20} {'Inertia':<15}")
    print("-"*60)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        inertia = kmeans.inertia_
        silhouette = silhouette_score(features_scaled, labels)
        
        inertias.append(inertia)
        silhouettes.append(silhouette)
        
        # Print with visual indicator for good scores
        indicator = ""
        if silhouette > 0.50:
            indicator = " *** GOOD"
        elif silhouette > 0.25:
            indicator = " *"
        
        print(f"{k:<5} {silhouette:<20.4f} {inertia:<15.2f}{indicator}")
    
    print("-"*60)
    
    # Plot elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True)
    
    ax2.plot(k_range, silhouettes, 'ro-')
    ax2.axhline(y=0.50, color='g', linestyle='--', label='Good threshold (0.50)')
    ax2.axhline(y=0.25, color='orange', linestyle='--', label='Weak threshold (0.25)')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=150)
    print(f"\n✓ Saved cluster analysis plot to: cluster_analysis.png")
    
    # Recommend k values
    best_k_silhouette = k_range[np.argmax(silhouettes)]
    best_silhouette = max(silhouettes)
    
    print(f"\n" + "="*60)
    print(f"RECOMMENDATIONS")
    print("="*60)
    print(f"Best k by silhouette score: {best_k_silhouette} (score: {best_silhouette:.4f})")
    
    # Find top 3 k values by silhouette
    sorted_indices = np.argsort(silhouettes)[::-1][:3]
    print(f"\nTop 3 k values by silhouette score:")
    for i, idx in enumerate(sorted_indices, 1):
        k = k_range[idx]
        score = silhouettes[idx]
        print(f"  {i}. k={k} (score: {score:.4f})")

def plot_dendrogram(features, n_clusters=None, method='ward'):
    """
    Plot dendrogram for hierarchical clustering.
    
    Args:
        features: feature matrix
        n_clusters: if provided, draw a horizontal line at the cut height
        method: linkage method ('ward', 'complete', 'average', 'single')
    """
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute linkage matrix
    print(f"\nComputing hierarchical clustering with method='{method}'...")
    Z = linkage(features_scaled, method=method)
    
    # Plot dendrogram
    plt.figure(figsize=(15, 8))
    
    if n_clusters:
        # Calculate cut height for n_clusters
        # The linkage matrix Z has shape (n-1, 4)
        # Z[i,2] is the distance at which clusters were merged
        # To get n_clusters, we cut at the (n-n_clusters)th merge
        cut_height = Z[-(n_clusters-1), 2] if n_clusters > 1 else Z[-1, 2] * 1.1
        
        dendrogram(Z, 
                   truncate_mode='lastp',  # Show only last p merged clusters
                   p=50,  # Show last 50 merges for readability
                   leaf_font_size=10,
                   color_threshold=cut_height)
        
        # Draw horizontal line at cut height
        plt.axhline(y=cut_height, color='r', linestyle='--', 
                   label=f'Cut for {n_clusters} clusters')
        plt.legend()
    else:
        dendrogram(Z,
                   truncate_mode='lastp',
                   p=50,
                   leaf_font_size=10)
    
    plt.title(f'Hierarchical Clustering Dendrogram (method={method})')
    plt.xlabel('Cluster Index (or Sample Count in parentheses)')
    plt.ylabel('Distance')
    plt.tight_layout()
    
    output_file = f'dendrogram_k{n_clusters if n_clusters else "full"}.png'
    plt.savefig(output_file, dpi=150)
    print(f"✓ Saved dendrogram to: {output_file}")
    plt.close()

def save_clustered_segments(segments, cluster_labels, output_file, cluster_descriptions=None):
    """
    Save segments with cluster assignments and descriptions.
    """
    
    for i, seg in enumerate(segments):
        seg["cluster_id"] = int(cluster_labels[i])
    
    output_data = {"segments": segments}
    if cluster_descriptions:
        output_data["cluster_descriptions"] = cluster_descriptions
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved clustered segments to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster gaze segments to discover goal patterns")
    parser.add_argument("--json_file", type=str, default="gaze_segments_54_RZ.json",
                       help="Input JSON file with segments")
    parser.add_argument("--n_clusters", type=int, default=None,
                       help="Number of clusters (if None, will find optimal)")
    parser.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "hierarchical"],
                       help="Clustering method")
    parser.add_argument("--output", type=str, default="clustered_segments.json",
                       help="Output file for clustered segments")
    parser.add_argument("--find_k", action="store_true",
                       help="Find optimal k using elbow/silhouette method")
    parser.add_argument("--show_dendrogram", action="store_true",
                       help="Show dendrogram for hierarchical clustering")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading segments from {args.json_file}...")
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    segments = data["segments"]
    print(f"Loaded {len(segments)} segments")
    
    # Extract features
    print("\nExtracting features...")
    features, segment_indices, feature_names = extract_features(segments)
    print(f"Feature matrix shape: {features.shape}")
    
    # Find optimal k if requested
    if args.find_k:
        print("\nFinding optimal number of clusters...")
        find_optimal_k(features, max_k=40)
        print("\nRun again with --n_clusters <k> to cluster with chosen k")
    else:
        # Cluster
        n_clusters = args.n_clusters if args.n_clusters else 5
        print(f"\nClustering into {n_clusters} clusters using {args.method}...")
        
        # Show dendrogram if requested and using hierarchical
        if args.show_dendrogram and args.method == "hierarchical":
            plot_dendrogram(features, n_clusters=n_clusters, method='ward')
        
        cluster_labels, clusterer, scaler = cluster_segments(features, n_clusters, args.method)
        
        # Analyze
        cluster_descriptions = analyze_clusters(segments, cluster_labels, n_clusters, features)
        
        # Save
        save_clustered_segments(segments, cluster_labels, args.output, cluster_descriptions)
        
        # Print cluster distribution
        print(f"\n{'='*80}")
        print("CLUSTER SIZE DISTRIBUTION")
        print(f"{'='*80}")
        cluster_counts = Counter(cluster_labels)
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]
            pct = 100 * count / len(segments)
            print(f"Cluster {cluster_id}: {count:4d} segments ({pct:5.1f}%)")
