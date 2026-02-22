import torch
import sys
import os
import numpy as np

# Add nsfr to path
sys.path.append(os.path.join(os.getcwd(), 'nsfr'))

from nudge.agents.imitation_agent import ImitationAgent

def run_comparison():
    device = "cpu"
    print("Initializing ImitationAgent...")
    # Initialize agent (seaquest, rules=new)
    # We use cpu for simplicity and direct tensor inspection
    agent = ImitationAgent("seaquest", "new", device=device)
    
    # Create Dummy State
    # Shape: (Batch, N_Objects, N_Features)
    # N_Objects = 48, N_Features = 7
    # Features: [vis, x, y, w, h, ori, type_id]
    
    n_objects = 48
    n_features = 7
    state = torch.zeros((1, n_objects, n_features), dtype=torch.float32)
    
    # Define Objects
    # Object 0: Player (Type 3) at (50, 50)
    # Note: Using index 0 for Player might not match strict env offsets but works for type-based predicates
    state[0, 0] = torch.tensor([1, 50, 50, 10, 10, 0, 3])
    
    # Object 1: Enemy (Type 0) at (50, 100) (Below Player)
    state[0, 1] = torch.tensor([1, 50, 100, 10, 10, 0, 0])
    
    # Object 2: Diver (Type 1) at (100, 50) (Right of Player)
    state[0, 2] = torch.tensor([1, 100, 50, 10, 10, 0, 1])
    
    print("\nState defined:")
    print("Object 0 (Player): (50, 50)")
    print("Object 1 (Enemy):  (50, 100)")
    print("Object 2 (Diver):  (100, 50)")
    
    # Create Synthetic Gazemap
    # Shape: (Batch, 84, 84)
    gazemap = torch.zeros((1, 84, 84), dtype=torch.float32)
    
    # Focus on Enemy at (50, 100)
    # Scale coordinates to 84x84
    # Frame size: 160x210
    gx = int(50 * (84/160))
    gy = int(100 * (84/210))
    
    print(f"\nCreating Gazemap focused on Enemy at ({gx}, {gy}) in 84x84 grid...")
    
    # Create a 5x5 region of high attention
    r = 5
    gazemap[0, max(0, gy-r):min(84, gy+r), max(0, gx-r):min(84, gx+r)] = 1.0
    
    # Run WITHOUT Gazemap
    print("\nRunning inference WITHOUT Gazemap...")
    # Passing None as gaze
    _ = agent.model(state, gaze=None)
    v0_no_gaze = agent.model.V_0.clone().detach() # (1, N_Atoms)
    
    # Run WITH Gazemap
    print("Running inference WITH Gazemap...")
    _ = agent.model(state, gaze=gazemap)
    v0_gaze = agent.model.V_0.clone().detach() # (1, N_Atoms)
    
    # Find relevant atoms
    # We want to check 'visible_enemy(obj1)' and 'visible_diver(obj2)'
    # Note: The atom string representation depends on how NSFR names objects.
    # Usually 'visible_enemy(obj1)' if we use offsets, OR 
    # 'visible_enemy(o_1)' depending on constant naming.
    # Let's inspect all atoms that show difference.
    
    atoms = agent.model.atoms
    diff_count = 0
    print("\nComparing Valuation Vectors (V_0):")
    print(f"{'Atom':<60} | {'No Gaze':<10} | {'With Gaze':<10} | {'Diff'}")
    print("-" * 100)
    
    for i, atom in enumerate(atoms):
        val_no = v0_no_gaze[0, i].item()
        val_with = v0_gaze[0, i].item()
        
        if abs(val_no - val_with) > 1e-4:
            print(f"{str(atom):<60} | {val_no:.4f}     | {val_with:.4f}     | {val_with - val_no:.4f}")
            diff_count += 1
            
    if diff_count == 0:
        print("No differences found! The gazemap had no effect on V_0.")
    else:
        print(f"\nFound {diff_count} atoms with different valuations.")

if __name__ == "__main__":
    run_comparison()
