
from nudge.agents.imitation_agent import ImitationAgent
import torch
import pandas as pd

def check_atoms():
    # Load agent
    print("Loading agent...")
    env_name = "seaquest"
    rules = "new" # Assuming 'new' rules based on user command
    device = "cpu"
    
    agent = ImitationAgent(env_name, rules, device)
    atoms = agent.model.atoms
    print(f"Model has {len(atoms)} atoms.")
    
    # Load CSV
    csv_file = "data/seaquest/train.csv"
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file, nrows=5)
    
    row = df.iloc[2] # Pick a row
    rel_str = row['relationships']
    print(f"\nCSV Relationship String: {rel_str}")
    
    # Parse CSV predicates
    # Format: "pred1 , pred2(arg) , "
    csv_preds = [x.strip() for x in rel_str.split(',') if x.strip()]
    
    print(f"\nParsed {len(csv_preds)} predicates from CSV.")
    
    # Check for matches
    print("\nChecking matches...")
    atom_strs = [str(a) for a in atoms]
    
    match_count = 0
    for pred in csv_preds:
        if pred in atom_strs:
            match_count += 1
        else:
            print(f"MISSING: '{pred}' not found in model atoms.")
            # Partial match check?
            # e.g. CSV has 'enemyFacingLeft(enemy_submarine_0)'
            # Atom might be 'enemyFacingLeft(X)'? No, atoms are ground atoms.
            
    print(f"\nMatched {match_count}/{len(csv_preds)} predicates.")
    
    if match_count == len(csv_preds):
        print("SUCCESS: All CSV predicates match model atoms!")
    else:
        print("FAILURE: Mismatch detected.")
        print("Sample model atoms:")
        for a in atom_strs[:10]:
            print(f"  {a}")

if __name__ == "__main__":
    check_atoms()
