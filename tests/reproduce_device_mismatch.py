import torch
import sys
import os

# Add nsfr to path
sys.path.append(os.path.join(os.getcwd(), 'nsfr'))

from nudge.agents.imitation_agent import ImitationAgent

def reproduce():
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot reproduce device mismatch (cuda vs cpu).")
        return

    device = "cuda"
    print(f"Initializing ImitationAgent on {device}...")
    agent = ImitationAgent("seaquest", "new", device=device)
    
    n_objects = 48
    n_features = 7
    state = torch.zeros((1, n_objects, n_features), dtype=torch.float32).to(device)
    
    # Define an enemy
    state[0, 1] = torch.tensor([1, 50, 100, 10, 10, 0, 0], device=device)
    
    # Create Synthetic Gazemap on CPU
    print("Creating Gazemap on CPU...")
    gazemap = torch.zeros((1, 84, 84), dtype=torch.float32) # Default is CPU
    gazemap[0, 40:45, 25:30] = 1.0
    
    print("\nRunning inference with Gazemap (CPU) and States (CUDA)...")
    try:
        _ = agent.model(state, gaze=gazemap)
        print("SUCCESS: No device mismatch error.")
    except Exception as e:
        print(f"CAUGHT ERROR: {e}")

if __name__ == "__main__":
    reproduce()
