
from train_il import ExpertDataset, BASE_IMAGE_DIR, CSV_FILE
from ocatari.core import OCAtari
import os
from unittest.mock import MagicMock

# Mock OCAtari to avoid heavy initialization if possible, 
# although __init__ does init it.
# We can just let it init, it takes a few seconds.

class MockNudgeEnv:
    def __init__(self):
        self.pred2action = {}
    
    def convert_state(self, objects):
        return [0]*10, [0]*10
        
    def reset(self):
        return ([0]*10, [0]*10)

def test_dataset_init():
    print("Testing ExpertDataset initialization...")
    # Mock environment
    env = MockNudgeEnv()
    
    # Init dataset
    # We use a small limit to speed up if df loading is slow
    dataset = ExpertDataset("seaquest", ["pred1"], limit=5, nudge_env=env)
    
    print(f"Map size: {len(dataset.traj_folder_map)}")
    assert len(dataset.traj_folder_map) > 0, "Traj folder map should not be empty"
    
    # Test __getitem__
    print("Testing __getitem__...")
    dataset.__getitem__(0)
    print("Successfully loaded item 0")
    
    # Verify map usage
    # We can inspect if it populated the cache further if needed
    
if __name__ == "__main__":
    test_dataset_init()
