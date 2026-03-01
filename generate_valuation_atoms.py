import os
import argparse
import torch
from tqdm import tqdm
import pickle
import numpy as np

from nudge.env import NudgeBaseEnv
from nudge.agents.imitation_agent import ImitationAgent

def get_args():
    parser = argparse.ArgumentParser(description="Generate Ground Truth Valuation Atoms")
    parser.add_argument("--dataset", type=str, default="data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt")
    parser.add_argument("--output", type=str, default="nsfr/seaquest/initial_atoms.pkl")
    parser.add_argument("--env", type=str, default="seaquest")
    parser.add_argument("--rules", type=str, default="new")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_gaze", action="store_true", help="Apply gaze masks to evaluations")
    parser.add_argument("--gaze_threshold", type=float, default=50.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Filter out ambiguous frames (e.g. 0.8 keeps only highly confident atoms)")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)

    print(f"Loading dataset from {args.dataset}...")
    data = torch.load(args.dataset, map_location='cpu', weights_only=False)
    
    logic_states = data['logic_state']  # (N, num_obj, features)
    actions = data['actions']           # (N,)
    observations = data['observations'] # (N, H, W)
    gaze_images = data.get('gaze_image', None)
    
    # Filter valid actions (0-5)
    valid_mask = actions <= 5
    logic_states = logic_states[valid_mask]
    observations = observations[valid_mask]
    actions = actions[valid_mask]
    if gaze_images is not None:
        gaze_images = gaze_images[valid_mask]

    N = len(logic_states)
    print(f"Total valid samples: {N}")

    print(f"Initializing NSFR Engine on {device}...")
    # Instantiate the agent to access the compiled NSFR logic engine (FactsConverter)
    agent_gaze_thresh = args.gaze_threshold if args.use_gaze else None
    agent = ImitationAgent(args.env, args.rules, str(device), gaze_threshold=agent_gaze_thresh)
    model = agent.model
    model.eval()
    
    atom_names = [str(atom) for atom in model.atoms]
    print(f"Extracting {len(atom_names)} atoms per frame...")

    all_atom_probs = []
    
    # Process in batches 
    with torch.no_grad():
        for i in tqdm(range(0, N, args.batch_size), desc="Computing Valuations"):
            batch_logic = logic_states[i : i + args.batch_size]
            if not isinstance(batch_logic, torch.Tensor):
                batch_logic = torch.from_numpy(batch_logic)
            batch_logic = batch_logic.float().to(device)

            batch_gaze = None
            if args.use_gaze and gaze_images is not None:
                bg = gaze_images[i : i + args.batch_size]
                if not isinstance(bg, torch.Tensor):
                    bg = torch.from_numpy(bg)
                batch_gaze = bg.float().unsqueeze(1).to(device)
            
            # Run the object states through the symbolic Fact Converter
            atoms_vals = model.fc(batch_logic, model.atoms, model.bk, gaze=batch_gaze)
            
            # Store probabilities
            all_atom_probs.append(atoms_vals.cpu().numpy())

    all_atom_probs = np.concatenate(all_atom_probs, axis=0) # (N, num_atoms)

    # Filtering logic for high-confidence atoms if specified
    if args.confidence_threshold > 0.0:
        # Confidence logic: frames where the valuation probabilities are highly polarized (close to 0 or 1).
        # Score = |prob - 0.5| * 2. High = 1.0 (prob=0 or 1), Low = 0.0 (prob=0.5)
        # We can just check if the average confidence of atoms is high, or just save all and let the DataLoader filter.
        print(f"Filtering out frames where average atom confidence is < {args.confidence_threshold}")
        confidence_scores = np.abs(all_atom_probs - 0.5) * 2.0
        mean_conf = confidence_scores.mean(axis=1)
        
        keep_mask = mean_conf >= args.confidence_threshold
        all_atom_probs = all_atom_probs[keep_mask]
        observations = observations[keep_mask]
        actions = actions[keep_mask]
        print(f"Filtered dataset from {N} to {len(all_atom_probs)} highly confident frames.")
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Saving compiled atoms to {args.output}...")
    
    # Save as a comprehensive dictionary so the CNN has everything it needs
    save_data = {
        'atom_names': atom_names,
        'atom_probs': all_atom_probs,          # (N_filtered, num_atoms) float32
        'observations': observations,          # (N_filtered, H, W) uint8
        'actions': actions                     # (N_filtered,) long
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(save_data, f)
        
    print("Done! You can now load this .pkl and map 'observations' directly to 'atom_probs' via BCELoss.")

if __name__ == "__main__":
    main()
