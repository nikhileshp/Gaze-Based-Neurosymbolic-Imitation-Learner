import torch
import argparse
from nsfr.utils.common import load_module
from nsfr.fol.language import Language

def main():
    parser = argparse.ArgumentParser(description="Check valuation vector contents")
    parser.add_argument("--valuation_path", type=str, required=True, help="Path to valuations_*.pt")
    parser.add_argument("--env", type=str, required=True, help="Environment name (e.g., seaquest, asterix)")
    parser.add_argument("--ep_id", type=int, default=0, help="Episode ID to check")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index within episode")
    parser.add_argument("--threshold", type=float, default=0.5, help="Only show atoms with value > threshold")
    args = parser.parse_args()

    # 1. Load valuations
    print(f"Loading valuations from {args.valuation_path}...")
    vals = torch.load(args.valuation_path, map_location='cpu', weights_only=False)
    
    if args.ep_id not in vals:
        print(f"Error: Episode {args.ep_id} not found. Available episodes: {list(vals.keys())[:10]}...")
        return
        
    ep_vals = vals[args.ep_id]
    if args.frame_idx >= len(ep_vals):
        print(f"Error: Frame {args.frame_idx} out of range (max {len(ep_vals)-1})")
        return
        
    frame_val = ep_vals[args.frame_idx] # This is a tensor of shape (num_atoms,)
    
    # 2. Get Atoms from Language
    print(f"Loading language for {args.env}...")
    # This part depends on how atoms are constructed. 
    # Usually they are built in the NSFR model.
    # We can try to guess the path to preds.txt
    preds_path = f"core/envs/{args.env}/logic/best/preds.txt"
    try:
        from nsfr.fol.logic import NeuralPredicate
        # We need a dummy Language object to get the ground atoms
        # This is a bit complex without the full NSFR setup, but we can try:
        # Alternatively, we can just print indices if we don't have the labels easily.
        print("Mapping indices to atoms (this requires the same Language setup as training)...")
        # For a truly robust tool, we'd need to re-init the NSFR model.
    except ImportError:
        pass

    print(f"\nTop valuations for Episode {args.ep_id}, Frame {args.frame_idx}:")
    print(f"{'Index':<6} | {'Value':<8}")
    print("-" * 20)
    
    # Sort by value descending
    vals_sorted, indices = torch.sort(frame_val, descending=True)
    
    for i in range(len(indices)):
        val = vals_sorted[i].item()
        if val < args.threshold:
            break
        idx = indices[i].item()
        print(f"{idx:<6} | {val:<8.4f}")

    print("\nTo see labels, you need to match these indices with the ground atoms list from the model.")

if __name__ == "__main__":
    main()
