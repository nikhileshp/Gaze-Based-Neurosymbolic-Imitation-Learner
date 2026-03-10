import torch
import argparse
import os
import sys

# Add the project root to sys.path so we can import core and nsfr
sys.path.append(os.getcwd())

from core.nsfr.utils.logic import get_lang

def main():
    parser = argparse.ArgumentParser(description="Check valuation vector contents with atom labels")
    parser.add_argument("--valuation_path", type=str, required=True, help="Path to valuations_*.pt")
    parser.add_argument("--env", type=str, required=True, help="Environment name (e.g., seaquest, asterix)")
    parser.add_argument("--rules", type=str, default="best", help="Rule set to use for language (default: best)")
    parser.add_argument("--ep_id", type=int, default=0, help="Episode ID to check")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index within episode")
    parser.add_argument("--threshold", type=float, default=0.5, help="Only show atoms with value > threshold")
    args = parser.parse_args()

    # 1. Load language and generate atoms
    # We must use the same paths as get_nsfr_model in common.py
    lark_path = 'core/nsfr/lark/exp.lark'
    lang_base_path = f"core/envs/{args.env}/logic/"
    
    print(f"Loading language for {args.env} (rules: {args.rules})...")
    try:
        # get_lang(lark_path, lang_base_path, dataset, use_limited_consts=False)
        # Note: the 'dataset' argument in get_lang corresponds to the folder name under logic/
        lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, args.rules)
    except Exception as e:
        print(f"Error loading language: {e}")
        return

    # 2. Load valuations
    if not os.path.exists(args.valuation_path):
        print(f"Error: Valuation file not found at {args.valuation_path}")
        return

    print(f"Loading valuations from {args.valuation_path}...")
    vals = torch.load(args.valuation_path, map_location='cpu', weights_only=False)
    
    if args.ep_id not in vals:
        available_eps = list(vals.keys())
        print(f"Error: Episode {args.ep_id} not found. Available episodes: {available_eps[:10]}...")
        return
        
    ep_vals = vals[args.ep_id]
    if args.frame_idx >= len(ep_vals):
        print(f"Error: Frame {args.frame_idx} out of range (max {len(ep_vals)-1})")
        return
        
    frame_val = ep_vals[args.frame_idx] # This is a tensor of shape (num_atoms,)
    
    if len(frame_val) != len(atoms):
        print(f"Warning: Valuation vector size ({len(frame_val)}) does not match atom count ({len(atoms)})!")
        print("This usually means the rule set or environment doesn't match the stored valuations.")

    # 3. Print valuations with labels
    print(f"\nTop valuations for Episode {args.ep_id}, Frame {args.frame_idx}:")
    print(f"{'Value':<8} | {'Index':<6} | {'Atom'}")
    print("-" * 40)
    
    # Sort by value descending
    vals_sorted, indices = torch.sort(frame_val, descending=True)
    
    for i in range(len(indices)):
        val = vals_sorted[i].item()
        if val < args.threshold:
            break
        idx = indices[i].item()
        
        atom_label = str(atoms[idx]) if idx < len(atoms) else "[UNKNOWN ATOM]"
        print(f"{val:<8.4f} | {idx:<6} | {atom_label}")

if __name__ == "__main__":
    main()
