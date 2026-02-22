
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys 

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

class Human_Gaze_Predictor:
    def __init__(self, game_name):
        self.game_name = game_name 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def init_model(self, gaze_model_file=None, lr=1.0, rho=0.95, weight_decay=0.0):
        self.k = 4
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
        
        # Convert NHWC -> NCHW for PyTorch
        if imgs.ndim == 4:
            imgs_tensor = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2)
        elif imgs.ndim == 3:
            imgs_tensor = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1)
            
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
            
        save_path = f"{self.game_name}_gaze_predictor_2.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
  
    def predict_and_save(self, imgs):
        """
        imgs: Numpy array of shape (Batch, Height, Width, Channels) (NHWC)
        Returns predictions and saves them.
        """
        print("Predicting results...")
        self.model.eval()
        
        # Convert NHWC -> NCHW for PyTorch
        if imgs.ndim == 4:
            imgs_tensor = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2)
        elif imgs.ndim == 3:
            imgs_tensor = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1)
        
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
    from load_data import *
    
    parser = argparse.ArgumentParser(description="Generate human gaze heatmaps from trained PyTorch predictor model.")
    parser.add_argument('--trajectories_dir', '-t', type=str, required=True, help="Path to the trajectories directory.")
    parser.add_argument('--labels_csv', '-l', type=str, required=True, help="Path to the labels CSV file.")
    parser.add_argument('--game_name', '-g', type=str, required=True, help="Name of the game (e.g., seaquest).")
    parser.add_argument('--model_weights', '-m', type=str, required=False, default=None, help="Path to the pre-trained gaze model .pth file.")
    parser.add_argument('--train', action='store_true', help="Set this flag to train the model instead of generating predictions.")
    parser.add_argument('--gaze_masks', type=str, default="data/seaquest/gaze_masks.pt", help="Path to the ground truth gaze masks .pt file (required for training).")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train.")
    parser.add_argument('--lr', type=float, default=1.0, help="Learning rate for Adadelta optimizer (default: 1.0).")
    parser.add_argument('--rho', type=float, default=0.95, help="Rho parameter for Adadelta optimizer (default: 0.95).")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay for regularization (default: 0.0).")
    parser.add_argument('--log_csv', type=str, default=None, help="Path to save training learning curve metrics (Epoch vs Loss) as CSV.")
    args = parser.parse_args()

    d = Dataset(args.trajectories_dir, args.labels_csv)
    # For gaze prediction
    d.generate_data_for_gaze_prediction()
    
    gp = Human_Gaze_Predictor(args.game_name) #game name
    gp.init_model(args.model_weights, lr=args.lr, rho=args.rho, weight_decay=args.weight_decay) #gaze model .pth file
    gp.log_csv = args.log_csv
    
    if args.train:
        print(f"Loading ground truth masks from {args.gaze_masks}...")
        masks_tensor = torch.load(args.gaze_masks, map_location='cpu') # Shape (Num_Frames, 84, 84)
        
        # d.gaze_imgs corresponds to training frames starting after pastK=3
        valid_indices = d.original_indices[3:]
        
        gp.train_model(d.gaze_imgs, masks_tensor, valid_indices, epochs=args.epochs)
    else:
        gp.predict_and_save(d.gaze_imgs)