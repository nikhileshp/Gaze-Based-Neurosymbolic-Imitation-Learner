import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from baselines.models.linear_models import Encoder, weight_init

def get_args():
    parser = argparse.ArgumentParser(description="Train CNN to predict Neurosymbolic Valuation Atoms natively")
    parser.add_argument("--dataset", type=str, default="nsfr/seaquest/initial_atoms.pkl")
    parser.add_argument("--out_dir", type=str, default="models/valuation_cnn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # GABRIL Architecture Params
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--num_residual_layers", type=int, default=2)
    parser.add_argument("--num_residual_hiddens", type=int, default=32)
    parser.add_argument("--z_dim", type=int, default=256)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, 'rb') as f:
        data = pickle.load(f)

    # Dictionary keys from generate_valuation_atoms.py
    # atom_names: list of str
    # observations: (N, 84, 84) uint8
    # atom_probs: (N, num_atoms) float32
    
    observations = data['observations']
    atom_probs = data['atom_probs']
    atom_names = data['atom_names']
    num_atoms = len(atom_names)
    
    if not isinstance(observations, torch.Tensor): observations = torch.from_numpy(observations)
    if not isinstance(atom_probs, torch.Tensor): atom_probs = torch.from_numpy(atom_probs)
    
    # Preprocess images to float [0, 1] mapped as (B, C, H, W)
    print("Normalizing images...")
    if observations.ndim == 3:
        observations = observations.float().unsqueeze(1) / 255.0  # -> (N, 1, 84, 84)
    elif observations.ndim == 4:
        observations = observations.float() / 255.0  # -> (N, 4, 84, 84)
        
    in_channels = observations.shape[1]
    
    # Split into Train / Validation (95/5)
    dataset_size = len(observations)
    indices = np.random.permutation(dataset_size)
    split = int(0.95 * dataset_size)
    
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_ds = TensorDataset(observations[train_idx], atom_probs[train_idx])
    val_ds = TensorDataset(observations[val_idx], atom_probs[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset mapped: {len(train_ds)} train, {len(val_ds)} val elements.")
    print(f"Goal: Predict {num_atoms} independent logic probabilities directly from 84x84 Grayscale.")

    # ==========================
    # Model Architecture
    # ==========================
    # 1. Feature Extractor (GABRIL Encoder)
    encoder = Encoder(in_channels, args.embedding_dim, args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens).to(device)
    
    # 2. Flatten and map downwards into dense subspace
    encoder_out_dim = 8 * 8 * args.embedding_dim  # 4096
    pre_actor = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(encoder_out_dim, args.z_dim),
        nn.ReLU()
    ).to(device)
    pre_actor.apply(weight_init)
    
    # 3. Final Prediction Head for Independent atoms
    # We use Sigmoid tightly bounding the atom logic boundaries [0, 1].
    atom_predictor = nn.Sequential(
        nn.Linear(args.z_dim, args.z_dim),
        nn.ReLU(),
        nn.Linear(args.z_dim, num_atoms),
        nn.Sigmoid() 
    ).to(device)
    atom_predictor.apply(weight_init)

    # Optimizer & Loss
    params = list(encoder.parameters()) + list(pre_actor.parameters()) + list(atom_predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    # Using BCELoss because atoms probabilities are multi-label independent, NOT mutually exclusive
    criterion = nn.BCELoss()

    # ==========================
    # Training Loop
    # ==========================
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        encoder.train()
        pre_actor.train()
        atom_predictor.train()
        
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_imgs, batch_targets in pbar:
            batch_imgs = batch_imgs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            z = encoder(batch_imgs)
            dense_z = pre_actor(z)
            preds = atom_predictor(dense_z)
            
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_imgs.size(0)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_ds)
        
        # Validation Form
        encoder.eval()
        pre_actor.eval()
        atom_predictor.eval()
        
        val_loss = 0.0
        # Calculate strict logic accuracy using threshold 0.5 parity
        total_atoms = 0
        correct_atoms = 0
        
        with torch.no_grad():
            for batch_imgs, batch_targets in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_targets = batch_targets.to(device)
                
                z = encoder(batch_imgs)
                dense_z = pre_actor(z)
                preds = atom_predictor(dense_z)
                
                v_loss = criterion(preds, batch_targets)
                val_loss += v_loss.item() * batch_imgs.size(0)
                
                # Binarize threshold
                pred_bin = (preds > 0.5).float()
                target_bin = (batch_targets > 0.5).float()
                
                correct_atoms += (pred_bin == target_bin).sum().item()
                total_atoms += batch_targets.numel()
                
        val_loss /= len(val_ds)
        val_acc = correct_atoms / total_atoms
        
        print(f"-> Epoch {epoch+1} | Train BCELoss: {train_loss:.4f} | Val BCELoss: {val_loss:.4f} | Val Atom Acc (Threshold 0.5): {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("  [✓] New best validation loss! Saving checkpoints...")
            torch.save(encoder.state_dict(), f"{args.out_dir}/best_encoder.pth")
            torch.save(pre_actor.state_dict(), f"{args.out_dir}/best_pre_actor.pth")
            torch.save(atom_predictor.state_dict(), f"{args.out_dir}/best_atom_predictor.pth")

    print("\nTraining Complete.")
    print(f"Best Models saved in: {args.out_dir}")
    print(f"Final Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
