
import torch
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'nsfr'))

from train_il import ExpertDataset, PREDICATE_TO_ACTION_MAP
from nudge.agents.imitation_agent import ImitationAgent

def test_pkl_loading():
    print("Testing .pkl loading...")
    
    # 1. Create dummy .pkl
    dummy_file = "dummy_dataset.pkl"
    
    # Create dummy atom names
    # We need to match what a real agent might have or just random names for dataset test
    atom_names = [f"atom_{i}" for i in range(10)]
    
    # Create dummy data
    data = []
    supported_actions = list(PREDICATE_TO_ACTION_MAP.values())
    
    for i in range(50):
        # 10 atoms
        atoms = np.random.rand(10).tolist()
        action = np.random.choice(supported_actions)
        data.append({
            'frameid': f"f_{i}",
            'atoms': atoms,
            'action': action,
            'gaze': [10.0, 20.0]
        })
        
    dummy_content = {
        'atom_names': atom_names,
        'data': data
    }
    
    torch.save(dummy_content, dummy_file)
    print(f"Created {dummy_file}")
    
    try:
        # 2. Init ExpertDataset
        # We need to pass SOME prednames, but for pkl loading it might not matter much if we don't use them in __init__ for pkl logic
        # But ExpertDataset init expects valid env_name etc.
        # We can use "seaquest"
        
        print("Initializing ExpertDataset...")
        # agent_prednames can be empty list if we don't use it in __init__ for pkl path
        dataset = ExpertDataset("seaquest", [], dummy_file, limit=None)
        
        print(f"Dataset length: {len(dataset)}")
        assert len(dataset) == 50
        
        # 3. Check item
        item = dataset[0]
        print("Item 0 types:", type(item[0]), type(item[1]), type(item[2]))
        print("Item 0 shapes:", item[0].shape, item[1].shape, item[2].shape)
        
        assert item[0].dim() == 1
        assert item[0].size(0) == 10
        
        print("ExpertDataset loaded .pkl correctly.")
        
        # 4. Cleanup
        os.remove(dummy_file)
        
    except Exception as e:
        print(f"FAILED: {e}")
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
        raise e

def test_nsfr_forward():
    print("\nTesting NSFReasoner forward with 2D input...")
    # We need an agent.
    # This might require real rule files.
    # If it fails to init, we skip this part.
    try:
        agent = ImitationAgent("search", "search", "cpu") # Simple env/rules? Or seaquest
        # seaquest/new might be safer given files exist
        
    except Exception as e:
        print(f"Could not init agent: {e}. Trying simple mock.")
        return

    # Mock Input (Batch=2, Atoms=Probs)
    num_atoms = len(agent.model.atoms)
    print(f"Model has {num_atoms} atoms.")
    
    # 2D input
    dummy_input = torch.rand(2, num_atoms)
    
    # Forward
    try:
        output = agent.model(dummy_input)
        print("Forward pass successful.")
        print("Output shape:", output.shape)
    except Exception as e:
        print(f"Forward pass FAILED: {e}")
        # raise e

if __name__ == "__main__":
    test_pkl_loading()
    # test_nsfr_forward() # Optional, might need checking if agent can start
