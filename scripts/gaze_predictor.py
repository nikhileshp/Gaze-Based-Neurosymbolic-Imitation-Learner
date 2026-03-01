import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
try:
    from data_utils import load_gaze_predictor_data, load_fact_gaze_predictor_data
except ImportError:
    from scripts.data_utils import load_gaze_predictor_data, load_fact_gaze_predictor_data

def my_softmax(x):
    """
    Softmax activation function over spatial dimensions [H, W].
    Expected input shape: (B, C, H, W).
    """
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)
    x_soft = F.softmax(x_flat, dim=2)
    return x_soft.view(B, C, H, W)

def my_kld(y_true, y_pred):
    """
    Compute the KL-divergence between spatial heatmaps.
    Shapes: (B, C, H, W)
    """
    epsilon = 1e-10
    y_true = torch.clamp(y_true, min=epsilon, max=1.0)
    y_pred = torch.clamp(y_pred, min=epsilon, max=1.0)
    return torch.sum(y_true * torch.log(y_true / y_pred), dim=[1, 2, 3]).mean()

class HumanGazeNet(nn.Module):
    def __init__(self, in_channels=4, dropout=0.0):
        super(HumanGazeNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.drop3 = nn.Dropout2d(p=dropout)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.drop_d1 = nn.Dropout2d(p=dropout)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0)
        self.bn_d2 = nn.BatchNorm2d(32)
        self.drop_d2 = nn.Dropout2d(p=dropout)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4, padding=0)

    def forward(self, x):
        x = self.drop1(self.bn1(F.relu(self.conv1(x))))
        x = self.drop2(self.bn2(F.relu(self.conv2(x))))
        x = self.drop3(self.bn3(F.relu(self.conv3(x))))
        x = self.drop_d1(self.bn_d1(F.relu(self.deconv1(x))))
        x = self.drop_d2(self.bn_d2(F.relu(self.deconv2(x))))
        out = my_softmax(self.deconv3(x))
        return out

class FactGazeNet(nn.Module):
    def __init__(self, num_facts=284, frame_stack=4, dropout=0.0):
        super(FactGazeNet, self).__init__()
        input_dim = num_facts * frame_stack
        
        # Expand 1D facts to spatial bottleneck (64 channels of 7x7)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 64 * 7 * 7),
            nn.ReLU()
        )
        
        # Upsample 7x7 -> 14x14 -> 28x28 -> 84x84
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=3, padding=0)
        )

    def forward(self, x):
        # x shape: (B, input_dim)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 64, 7, 7)
        out = self.decoder(x)
        return my_softmax(out)

class Human_Gaze_Predictor:
    def __init__(self, game_name):
        self.game_name = game_name 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def init_model(self, gaze_model_file=None, lr=1.0, rho=0.95, weight_decay=0.0, use_facts=False):
        self.k = 4
        if use_facts:
            self.model = FactGazeNet(num_facts=284, frame_stack=self.k).to(self.device)
        else:
            self.model = HumanGazeNet(in_channels=self.k).to(self.device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr, rho=rho, eps=1e-08, weight_decay=weight_decay)
        self.criterion = my_kld
        
        if gaze_model_file:
            print(f"Loading model weights from {gaze_model_file}")
            try:
                self.model.load_state_dict(torch.load(gaze_model_file, map_location=self.device))
                print("Loaded successfully.")
            except Exception as e:
                print(f"Failed to load PyTorch weights ({e}). Cannot load Keras .hdf5 directly.")
                
    def train_model(self, imgs, masks_tensor, valid_indices, epochs=10, batch_size=64):
        """
        imgs: Numpy array of shape (Batch, Height, Width, Channels) (NHWC)
        masks_tensor: Ground truth torch tensor of shape (Total_Frames, 84, 84)
        valid_indices: The original indices of the frames in imgs mapping to masks_tensor
        """
        print(f"Starting training for {epochs} epochs...")
        self.model.train()
        
        # Convert NHWC -> NCHW for PyTorch, or keep 2D if using facts
        if imgs.ndim == 4:
            imgs_tensor = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2)
        elif imgs.ndim == 3:
            imgs_tensor = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1)
        else:
            # Fact data is 2D (Batch, k * num_atoms)
            imgs_tensor = imgs if isinstance(imgs, torch.Tensor) else torch.tensor(imgs, dtype=torch.float32)
            
        dataset_len = len(imgs_tensor)
        
        # Initialize logging
        if hasattr(self, 'log_csv') and self.log_csv:
            import pandas as pd
            # Create or overwrite the log file with headers
            pd.DataFrame(columns=['epoch', 'loss']).to_csv(self.log_csv, index=False)
            print(f"Logging training metrics to {self.log_csv}")
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Shuffle indices for the epoch
            permutation = torch.randperm(dataset_len)
            
            from tqdm import tqdm
            pbar = tqdm(range(0, dataset_len, batch_size), desc=f"Epoch {epoch+1}/{epochs}")
            for i in pbar:
                batch_indices = permutation[i:i+batch_size]
                
                # Get image batch
                batch_imgs = imgs_tensor[batch_indices].to(self.device)
                
                # Get corresponding mask indices from the original CSV 
                original_mask_indices = valid_indices[batch_indices]
                batch_masks = masks_tensor[original_mask_indices].unsqueeze(1).to(self.device) # (Batch, 1, 84, 84)
                
                self.optimizer.zero_grad()
                preds = self.model(batch_imgs)
                
                loss = self.criterion(batch_masks, preds)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(batch_indices)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            avg_loss = total_loss / dataset_len
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
            
            # Log metrics
            if hasattr(self, 'log_csv') and self.log_csv:
                import pandas as pd
                df = pd.DataFrame([{'epoch': epoch + 1, 'loss': avg_loss}])
                df.to_csv(self.log_csv, mode='a', header=False, index=False)
            
        save_path = f"models/gaze_predictor/{self.game_name}_{'fact' if hasattr(self.model, 'fc') else 'visual'}_gaze_predictor_limit_2.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
  
    def predict_and_save(self, imgs):
        """
        imgs: Numpy array of shape (Batch, Height, Width, Channels) (NHWC)
        Returns predictions and saves them.
        """
        print("Predicting results...")
        self.model.eval()
        
        # Convert NHWC -> NCHW for PyTorch, or keep 2D if using facts
        if imgs.ndim == 4:
            imgs_tensor = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2)
        elif imgs.ndim == 3:
            imgs_tensor = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1)
        else:
            imgs_tensor = imgs if isinstance(imgs, torch.Tensor) else torch.tensor(imgs, dtype=torch.float32)
        
        batch_size = 64
        preds_list = []
        
        with torch.no_grad():
            from tqdm import tqdm
            for i in tqdm(range(0, len(imgs_tensor), batch_size)):
                batch = imgs_tensor[i:i+batch_size].to(self.device)
                out = self.model(batch)
                preds_list.append(out.cpu().numpy())
                
        # Shape back to (N, 1, H, W)
        self.preds = np.concatenate(preds_list, axis=0)
        print("Predicted.")
    
        print("Writing predicted gaze heatmap (train) into the npz file...")
        # Output is (N, C, H, W) where C=1, save (N, H, W)
        np.savez_compressed("human_gaze_" + self.game_name, heatmap=self.preds[:, 0, :, :])
        print("Done. Output is:")
        print(" %s" % "human_gaze_" + self.game_name + '.npz')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or generate human gaze heatmaps.")
    # ----- new .pt-based flow -----
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to .pt dataset file (from convert_trajectories_to_pt.py). '
                             'When provided, this takes priority over --trajectories_dir / --labels_csv.')
    parser.add_argument('--use_facts', action='store_true',
                        help='If flag is set, Train FactGazeNet instead of HumanGazeNet directly from fact vectors.')
    parser.add_argument('--facts_dataset', type=str, default=None,
                        help='Path to .pkl containing atom_probs (only used with --use_facts).')
    parser.add_argument('--frame_stack', type=int, default=4,
                        help='Number of frames to stack as one input sample (default: 4).')
    # ----- legacy CSV-based flow -----
    parser.add_argument('--trajectories_dir', '-t', type=str, default=None,
                        help='Path to the trajectories directory (legacy CSV flow).')
    parser.add_argument('--labels_csv', '-l', type=str, default=None,
                        help='Path to the labels CSV file (legacy CSV flow).')
    parser.add_argument('--gaze_masks', type=str, default='data/seaquest/gaze_masks.pt',
                        help='Ground truth gaze masks .pt file (legacy flow only).')
    # ----- shared args -----
    parser.add_argument('--game_name', '-g', type=str, required=True,
                        help='Game name prefix for the saved checkpoint (e.g. seaquest).')
    parser.add_argument('--model_weights', '-m', type=str, default=None,
                        help='Path to a .pth checkpoint to resume training from.')
    parser.add_argument('--train', action='store_true',
                        help='Train the model (required for legacy CSV flow; always trains in .pt flow).')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--rho', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--log_csv', type=str, default=None,
                        help='CSV path to log (epoch, avg_loss) pairs.')
    args = parser.parse_args()

    gp = Human_Gaze_Predictor(args.game_name)
    gp.init_model(args.model_weights, lr=args.lr, rho=args.rho, weight_decay=args.weight_decay, use_facts=args.use_facts)
    gp.log_csv = args.log_csv

    # ── .pt dataset flow ─────────────────────────────────────────────────────
    if args.use_facts:
        if not args.facts_dataset:
            parser.error('--facts_dataset required when using --use_facts')
        # Here gaze_masks is used as the target labels tensor
        facts_stacked, gaze_masks, valid_indices = load_fact_gaze_predictor_data(
            args.facts_dataset, args.gaze_masks, frame_stack=args.frame_stack, device='cpu'
        )
        gp.train_model(facts_stacked, gaze_masks, valid_indices,
                       epochs=args.epochs, batch_size=args.batch_size)
    elif args.dataset:
        imgs_nhwc, gaze_masks, valid_indices = load_gaze_predictor_data(
            args.dataset, frame_stack=args.frame_stack, device='cpu'
        )

        gp.train_model(imgs_nhwc, gaze_masks, valid_indices,
                       epochs=args.epochs, batch_size=args.batch_size)

    # ── Legacy CSV flow ───────────────────────────────────────────────────────
    else:
        from load_data import Dataset
        if not args.trajectories_dir or not args.labels_csv:
            parser.error('--trajectories_dir and --labels_csv are required when --dataset is not set.')

        d = Dataset(args.trajectories_dir, args.labels_csv)
        d.generate_data_for_gaze_prediction()

        if args.train:
            print(f"Loading ground truth masks from {args.gaze_masks}...")
            masks_tensor = torch.load(args.gaze_masks, map_location='cpu')
            valid_indices = d.original_indices[3:]
            gp.train_model(d.gaze_imgs, masks_tensor, valid_indices,
                           epochs=args.epochs, batch_size=args.batch_size)
        else:
            gp.predict_and_save(d.gaze_imgs)