
from train_il import ExpertDataset, BASE_IMAGE_DIR, CSV_FILE
from torch.utils.data import DataLoader
import time
import torch
from nudge.env import NudgeBaseEnv

# We need to wrap custom env init because it's not picklable?
# OCAtari env might not be picklable.

def test_workers():
    print("Initializing main process...")
    # Using 'spawn' which is safer for CUDA and some C++ extensions
    # torch.multiprocessing.set_start_method('spawn', force=True)
    
    # We need a real env to init dataset?
    # ExpertDataset needs 'nudge_env' for rules.
    # NudgeEnv is picklable?
    env = NudgeBaseEnv.from_name("seaquest", mode='logic')
    
    # Init dataset (limit to small number for quick test, but enough to trigger workers)
    dataset = ExpertDataset("seaquest", ["pred1"], limit=10, nudge_env=env)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test num_workers=2
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    
    print("Starting iteration with num_workers=2...")
    start = time.time()
    for batch_idx, (states, actions) in enumerate(loader):
        print(f"Batch {batch_idx}: {states.shape}, {actions.shape}")
        if batch_idx >= 2: break
        
    print(f"Success! Time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    test_workers()
