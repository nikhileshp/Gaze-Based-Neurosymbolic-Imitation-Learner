import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from baselines.models.linear_models import Encoder

def get_validation_loader(dataset_path, in_channels, batch_size=256, seed=42):
    # Load identical seed to get the exact same validation 5% split
    np.random.seed(seed)
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
        
    observations = data['observations']
    atom_probs = data['atom_probs']
    num_atoms = len(data['atom_names'])
    
    if not isinstance(observations, torch.Tensor): observations = torch.from_numpy(observations)
    if not isinstance(atom_probs, torch.Tensor): atom_probs = torch.from_numpy(atom_probs)
    
    if in_channels == 1:
        observations = observations.float().unsqueeze(1) / 255.0
    else:
        observations = observations.float() / 255.0
        
    dataset_size = len(observations)
    indices = np.random.permutation(dataset_size)
    split = int(0.95 * dataset_size)
    val_idx = indices[split:]
    
    val_ds = TensorDataset(observations[val_idx], atom_probs[val_idx])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return val_loader, num_atoms

def evaluate_model(model_dir, val_loader, in_channels, num_atoms, device="cuda"):
    dev = torch.device(device)
    
    # Init architecture
    encoder = Encoder(in_channels, 64, 128, 2, 32).to(dev)
    pre_actor = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(8 * 8 * 64, 256),
        nn.ReLU()
    ).to(dev)
    atom_predictor = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, num_atoms),
        nn.Sigmoid() 
    ).to(dev)
    
    # Load weights
    encoder.load_state_dict(torch.load(f"{model_dir}/best_encoder.pth", map_location=dev))
    pre_actor.load_state_dict(torch.load(f"{model_dir}/best_pre_actor.pth", map_location=dev))
    atom_predictor.load_state_dict(torch.load(f"{model_dir}/best_atom_predictor.pth", map_location=dev))
    
    encoder.eval()
    pre_actor.eval()
    atom_predictor.eval()
    
    criterion = nn.BCELoss()
    val_loss = 0.0
    correct_atoms = 0
    total_atoms = 0
    
    with torch.no_grad():
        for batch_imgs, batch_targets in val_loader:
            batch_imgs = batch_imgs.to(dev)
            batch_targets = batch_targets.to(dev)
            
            z = encoder(batch_imgs)
            dense_z = pre_actor(z)
            preds = atom_predictor(dense_z)
            
            loss = criterion(preds, batch_targets)
            val_loss += loss.item() * batch_imgs.size(0)
            
            pred_bin = (preds > 0.5).float()
            target_bin = (batch_targets > 0.5).float()
            
            correct_atoms += (pred_bin == target_bin).sum().item()
            total_atoms += batch_targets.numel()
            
    avg_loss = val_loss / len(val_loader.dataset)
    acc = correct_atoms / total_atoms
    
    return avg_loss, acc

def main():
    print("="*50)
    print("Comparing Valuation Predictor Architectures")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Evaluate 1-Frame Model
    print("\\n[1] Evaluating 1-Frame Static Model...")
    loader_1, n_atoms = get_validation_loader("nsfr/seaquest/initial_atoms.pkl", in_channels=1)
    loss_1, acc_1 = evaluate_model("models/valuation_cnn/100_epochs", loader_1, 1, n_atoms, device)
    print(f"  -> Validation BCELoss: {loss_1:.4f}")
    print(f"  -> Validation Accuracy: {acc_1*100:.2f}%")
    
    # Evaluate 4-Frame Model
    print("\\n[2] Evaluating 4-Frame Temporal Model...")
    loader_4, _ = get_validation_loader("nsfr/seaquest/initial_atoms_4frame.pkl", in_channels=4)
    loss_4, acc_4 = evaluate_model("models/valuation_cnn_4/100_epochs", loader_4, 4, n_atoms, device)
    print(f"  -> Validation BCELoss: {loss_4:.4f}")
    print(f"  -> Validation Accuracy: {acc_4*100:.2f}%")
    
    print("\\n" + "="*50)
    print("WINNER ALGORITHM:")
    if acc_4 > acc_1:
         print(f"4-Frame Temporal Stack outperforms by {+(acc_4 - acc_1)*100:.2f}% accuracy!")
    else:
         print(f"1-Frame Static Frame outperforms by {+(acc_1 - acc_4)*100:.2f}% accuracy!")

if __name__ == "__main__":
    main()
