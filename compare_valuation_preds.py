import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from baselines.models.linear_models import Encoder, weight_init

def get_args():
    parser = argparse.ArgumentParser(description="Compare Valuation CNN predictions against ground truth")
    parser.add_argument("--dataset", type=str, default="nsfr/seaquest/initial_atoms.pkl")
    parser.add_argument("--model_dir", type=str, default="models/valuation_cnn")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold to binarize probability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # GABRIL Architecture Params (must match train_valuation_cnn.py)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--num_residual_layers", type=int, default=2)
    parser.add_argument("--num_residual_hiddens", type=int, default=32)
    parser.add_argument("--z_dim", type=int, default=256)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'rb') as f:
        data = pickle.load(f)

    observations = data['observations']
    atom_probs = data['atom_probs']
    atom_names = data['atom_names']
    num_atoms = len(atom_names)
    
    if not isinstance(observations, torch.Tensor): observations = torch.from_numpy(observations)
    if not isinstance(atom_probs, torch.Tensor): atom_probs = torch.from_numpy(atom_probs)
    
    # Normalize images
    if observations.ndim == 3:
        observations = observations.float().unsqueeze(1) / 255.0
    elif observations.ndim == 4:
        observations = observations.float() / 255.0
        
    in_channels = observations.shape[1]
    
    # Validation split to ensure we test on unseen data
    dataset_size = len(observations)
    np.random.seed(args.seed)
    indices = np.random.permutation(dataset_size)
    split = int(0.95 * dataset_size)
    val_idx = indices[split:]
    
    # Take a random subset of the validation set to display
    sample_indices = np.random.choice(val_idx, args.samples, replace=False)
    
    obs_samples = observations[sample_indices].to(device)
    target_samples = atom_probs[sample_indices].to(device)

    print("Loading models...")
    # Initialize Models
    encoder = Encoder(in_channels, args.embedding_dim, args.num_hiddens, 
                      args.num_residual_layers, args.num_residual_hiddens).to(device)
    
    encoder_out_dim = 8 * 8 * args.embedding_dim
    pre_actor = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(encoder_out_dim, args.z_dim),
        nn.ReLU()
    ).to(device)
    
    atom_predictor = nn.Sequential(
        nn.Linear(args.z_dim, args.z_dim),
        nn.ReLU(),
        nn.Linear(args.z_dim, num_atoms),
        nn.Sigmoid() 
    ).to(device)

    # Load Weights
    try:
        encoder.load_state_dict(torch.load(os.path.join(args.model_dir, "best_encoder.pth"), map_location=device))
        pre_actor.load_state_dict(torch.load(os.path.join(args.model_dir, "best_pre_actor.pth"), map_location=device))
        atom_predictor.load_state_dict(torch.load(os.path.join(args.model_dir, "best_atom_predictor.pth"), map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    encoder.eval()
    pre_actor.eval()
    atom_predictor.eval()

    print(f"\\n--- Comparing {args.samples} Samples ---")
    
    with torch.no_grad():
        z = encoder(obs_samples)
        dense_z = pre_actor(z)
        pred_samples = atom_predictor(dense_z)
    
    # Move to CPU for display
    target_samples = target_samples.cpu().numpy()
    pred_samples = pred_samples.cpu().numpy()

    for i in range(args.samples):
        print(f"\\n================= Sample {i+1} =================")
        print(f"{'Atom Name':<50} | {'Ground Truth':<15} | {'Prediction':<15} | {'Match?'}")
        print("-" * 95)
        
        matches = 0
        total_active_or_pred = 0
        
        for j, name in enumerate(atom_names):
            gt_prob = target_samples[i][j]
            pred_prob = pred_samples[i][j]
            
            gt_bin = gt_prob > args.threshold
            pred_bin = pred_prob > args.threshold
            
            # Print only if either ground truth OR prediction thinks the atom is true
            # (otherwise it would print 280+ mostly false atoms)
            if gt_bin or pred_bin:
                match_str = "YES" if gt_bin == pred_bin else "NO"
                print(f"{name:<50} | {gt_prob:>15.4f} | {pred_prob:>15.4f} | {match_str}")
                total_active_or_pred += 1
                if gt_bin == pred_bin:
                    matches += 1
                    
        print("-" * 95)
        if total_active_or_pred > 0:
            print(f"Active/Predicted Atom Match Rate: {matches}/{total_active_or_pred} ({(matches/total_active_or_pred)*100:.1f}%)")
        else:
            print("No atoms active in ground truth or prediction.")

if __name__ == "__main__":
    main()
