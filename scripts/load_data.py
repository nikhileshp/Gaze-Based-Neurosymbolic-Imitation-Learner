
import sys, os, re, threading, time, copy
import numpy as np
import tarfile
import cv2


def preprocess(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 84
    height = 84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame / 255.0

class Dataset:
  def __init__(self, tar_fname, label_fname):
    t1=time.time()
    print ("Reading all training data into memory...")

    # Read action labels from csv file
    frame_ids, episode_ids, lbls, original_indices = [], [], [], [] 
    import pandas as pd
    try:
        df = pd.read_csv(label_fname)
        if 'frame_id' in df.columns and 'action' in df.columns and 'episode_id' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row['action']):
                    frame_ids.append(str(row['frame_id']))
                    episode_ids.append(str(row['episode_id']))
                    lbls.append(int(row['action']))
                    original_indices.append(idx)
        else:
            print("Error: train.csv does not contain 'frame_id', 'episode_id', or 'action' columns.")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading {label_fname}: {e}")
        sys.exit(1)
        
    self.train_lbl = np.asarray(lbls, dtype=np.int32)
    self.train_size = len(self.train_lbl)
    self.frame_ids = np.asarray(frame_ids)
    self.episode_ids = np.asarray(episode_ids)
    self.original_indices = np.asarray(original_indices, dtype=np.int32)
    print(self.train_size)

    # Read training images directly from directories
    imgs = [None] * self.train_size
    
    # We assume 'tar_fname' is actually the base trajectories directory here.
    base_dir = tar_fname
    if not os.path.exists(base_dir):
        print(f"Warning: Base directory {base_dir} does not exist.")
        
    # Pre-index all available .png files in the base directory to fast-lookup
    import glob
    print("Indexing available PNG files in trajectories directory...")
    all_pngs = glob.glob(os.path.join(base_dir, "*", "*.png"))
    # build a dictionary: basename (without .png) -> full path
    png_map = {}
    for p in all_pngs:
        basename = os.path.basename(p)
        frame_name = os.path.splitext(basename)[0]
        png_map[frame_name] = p
        
    print(f"Reading images from {base_dir}...")
    from tqdm import tqdm
    for i in tqdm(range(self.train_size)):
        frame_id = self.frame_ids[i]
        
        if frame_id in png_map:
            png_fname = png_map[frame_id]
            try:
                img = np.float32(cv2.imread(png_fname))
                if img is None:
                    continue # Try not to crash if image is corrupt, though we could
                img = preprocess(img)
                imgs[i] = copy.deepcopy(img)
            except Exception as e:
                print(f"Failed to read/preprocess {png_fname}: {e}")
        else:
            # Can't find it, might be an issue.
            print(f"Warning: {frame_id}.png not found in {base_dir}")

    self.train_imgs = np.asarray(imgs)
    print ("Time spent to read training data: %.1fs" % (time.time()-t1))

  def standardize(self):
    self.mean = np.mean(self.train_imgs, axis=(0,1,2))
    self.train_imgs -= self.mean # done in-place --- "x-=mean" is faster than "x=x-mean"

  def load_predicted_gaze_heatmap(self, train_npz):
    train_npz = np.load(train_npz)
    self.train_GHmap = train_npz['heatmap']
    # npz file from pastK models has pastK-fewer data, so we need to know use value of pastK
    pastK = 3
    self.train_imgs = self.train_imgs[pastK:]
    self.train_lbl = self.train_lbl[pastK:]

  def reshape_heatmap_for_cgl(self, heatmap_shape):
    # predicted human gaze was in 84 x 84, needs to be reshaped for cgl
    #heatmap_shape: output feature map size of the conv layer 
    import cv2
    self.temp = np.zeros((len(self.train_GHmap), heatmap_shape, heatmap_shape))
    for i in range(len(self.train_GHmap)):
        self.temp[i] = cv2.resize(self.train_GHmap[i], (heatmap_shape, heatmap_shape), interpolation=cv2.INTER_AREA)
    self.train_GHmap = self.temp

  def generate_data_for_gaze_prediction(self):
    self.gaze_imgs = [None] * (self.train_size - 3)
    #stack every four frames to make an observation (84,84,4)
    for i in range(3, self.train_size):
        stacked_obs = np.zeros((84, 84, 4))
        stacked_obs[:, :, 0] = self.train_imgs[i-3]
        stacked_obs[:, :, 1] = self.train_imgs[i-2]
        stacked_obs[:, :, 2] = self.train_imgs[i-1]
        stacked_obs[:, :, 3] = self.train_imgs[i]
        self.gaze_imgs[i-3] = copy.deepcopy(stacked_obs)

    self.gaze_imgs = np.asarray(self.gaze_imgs)
    print("Shape of the data for gaze prediction: ", self.gaze_imgs.shape)

if __name__ == "__main__":
    d = Dataset(sys.argv[1], sys.argv[2]) #tarfile (images), txtfile (labels)
    # For gaze prediction
    d.generate_data_for_gaze_prediction()
    # For training imitation learning algorithms (cgl, agil)
    d.load_predicted_gaze_heatmap(sys.argv[3]) #npz file (predicted gaze heatmap)
    d.standardize() #for training imitation learning only, gaze model has its own mean files