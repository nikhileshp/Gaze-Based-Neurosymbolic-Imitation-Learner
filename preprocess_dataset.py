
import os
import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from functools import partial
import multiprocessing
import numpy as np
from PIL import Image


# Import necessary classes from train_il to allow pickling
from train_il import ExpertDataset, BASE_IMAGE_DIR, CSV_FILE, PREDICATE_TO_ACTION_MAP
from nudge.env import NudgeBaseEnv

def process_chunk(chunk_indices, dataset_args):
    """
    Process a chunk of indices and return a list of (frameid, logic_state_str).
    logic_state_str will be a string representation of the atoms/logic state.
    
    Actually, we want to save the atoms.
    ExpertDataset returns (logic_state_tensor, action).
    logic_state_tensor is (num_objects, num_features).
    
    We want to save this tensor. 
    Can we save it as a numpy array in .npy format? Or just a giant tensor file?
    
    Or can we save the atom strings?
    The user wants "frame -> input predicate atoms mapping".
    The model takes a tensor of shape (N, F).
    It converts this tensor to valuation V_0 using FactsConverter.
    
    If we pre-compute the valuation V_0 (probability of each atom), we can save that.
    V_0 shape is (num_atoms).
    
    Wait, the user said: "CSV relationships do not correspond to the input predicates... Need to create a new csv for frame -> input predicate atoms mapping"
    
    If we save the logic state tensor (objects), the model still needs to run FactsConverter.
    If we save the atoms (V_0), we bypass FactsConverter.
    
    Let's save the atoms (V_0).
    
    To does this, we need access to the agent/model to run FactsConverter.
    Or we can just instantiate FactsConverter.
    """
    env_name, rules = dataset_args
    # Re-init env and agent in worker?
    # Or just re-init FactsConverter.
    
    # We need to load the env to get the obs (objects)
    # Then converting objects to logic state tensor is fast.
    # Then running FactsConverter is the part we want to skip? 
    # Or is OCAtari the slow part? OCAtari is the slow part.
    
    # So if we just save the logic_state_tensor (objects), we skip OCAtari.
    # That is likely sufficient for speedup. FactsConverter is fast (tensor operations).
    
    # However, saving (N, F) tensor for every frame in a CSV is hard.
    # We can save it as a pickle or .pt file per frame? Or one big file.
    # If we want a CSV, we can save the atoms as a string representation if we run FC.
    
    # Let's stick to generating a mapping of frameid -> atoms (as string) if possible, 
    # OR just frameid -> file_path_to_tensor.
    
    # The user asked for "frame -> input predicate atoms mapping".
    # This suggests they might want the symbolic representation.
    
    # Let's initialize the dataset in the worker
    # We need to filter the df to just the chunk
    
    env = NudgeBaseEnv.from_name(env_name, mode='logic')
    # We assume rules are needed for atom generation if we go that far
    
    # We can't easily pass the huge DF, so we just pass indices and let worker load/subset?
    # Or pass the subset of DF.
    
    return []


def worker_fn(args_tuple, env_name):
    worker_id, indices = args_tuple
    
    # Set num threads to 1 to avoid thread contention
    torch.set_num_threads(1)
    
    # Load DF
    df = pd.read_csv(CSV_FILE)
    subset_df = df.iloc[indices]
    
    # Init Env
    env = NudgeBaseEnv.from_name(env_name, mode='logic')
    
    # Init Agent
    from nudge.agents.imitation_agent import ImitationAgent
    rules = "new" 
    device = "cpu"
    try:
        agent = ImitationAgent(env_name, rules, device)
        model = agent.model
        model.eval()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return []

    # Init Dataset
    dataset = ExpertDataset(env_name, [], limit=None, nudge_env=env)
    dataset.df = subset_df
    # Ensure trajectory map is built or pass it? 
    # ExpertDataset will build it again for this process. It's fast (cached).
    
    results = []
    
    # We want to save: frameid -> atoms (as list of probs)
    # The model atoms are fixed. We should save the model.atoms list ONCE in main process too.
    
    # Use tqdm with position for worker progress
    # leave=True so it persists (or False to clear)
    iterator = tqdm(range(len(subset_df)), desc=f"Worker {worker_id}", position=worker_id, leave=True)
    
    for i in iterator:
        try:
            row = subset_df.iloc[i]
            fid = row['frame_id']
            
            # Get logic state tensor (objects) using dataset
            # We use dataset[i] but it returns converted tensors
            # We need to be careful about index 'i'. dataset[i] uses iloc[i] on its internal df.
            # Our dataset.df IS subset_df, so i=0..len(subset_df) works.
            
            logic_state, action, gaze = dataset[i]
            
            # Forward pass through FactsConverter
            batch_state = logic_state.unsqueeze(0)
            with torch.no_grad():
                atoms_vals = model.fc(batch_state, model.atoms, model.bk)
             
            # Store
            atom_probs = atoms_vals[0].numpy().tolist()
            
            # We compress: only store active atoms? 
            # Or just store the full vector (254 floats). 
            # 254 floats is ~1KB. 80k frames = 80MB. Acceptable.
            
            record = {
                'frame_id': fid, 
                'action': int(action.item()),
                'atoms': atom_probs
            }
            results.append(record)
            
        except Exception as e:
            # Check if fid is defined, otherwise use 'unknown'
            current_fid = locals().get('fid', 'unknown')
            # Use tqdm.write to avoid interfering with progress bars
            tqdm.write(f"Error processing frame {current_fid}: {e}")
            pass
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="seaquest")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="data/seaquest/train_atoms.pkl")
    args = parser.parse_args()
    
    if not os.path.exists(CSV_FILE):
        print(f"CSV {CSV_FILE} not found.")
        return
        
    df = pd.read_csv(CSV_FILE)
    n_samples = len(df)
    print(f"Processing {n_samples} samples with {args.num_workers} workers...")
    
    # Save atom names first?
    # We can invoke agent once to get atom names and save metadata
    print("Extracting atom names...")
    from nudge.agents.imitation_agent import ImitationAgent
    dummy_agent = ImitationAgent(args.env, "new", "cpu")
    atom_names = [str(a) for a in dummy_agent.model.atoms]
    print(f"Model has {len(atom_names)} atoms.")
    
    indices = np.arange(n_samples)
    chunk_size = int(np.ceil(n_samples / args.num_workers))
    chunks = [indices[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]
    
    # Add worker IDs to chunks
    chunks_with_ids = list(enumerate(chunks))
    
    ctx = multiprocessing.get_context('spawn')
    
    # Initialize partial function with env_name fixed
    worker_partial = partial(worker_fn, env_name=args.env)

    with ctx.Pool(args.num_workers) as pool:
        results_nested = []
        # Use imap to get results as they complete
        # We also create a global progress bar, but set its position offset by num_workers
        total_chunks = len(chunks_with_ids)
        for res in tqdm(pool.imap(worker_partial, chunks_with_ids), total=total_chunks, desc="Total Progress", position=args.num_workers):
            results_nested.append(res)
            
    results = [item for sublist in results_nested for item in sublist]
    print(f"Generated {len(results)} records.")
    
    final_data = {
        'atom_names': atom_names,
        'data': results
    }
    
    print(f"Saving to {args.output}...")
    torch.save(final_data, args.output)
    print("Done.")

if __name__ == "__main__":
    main()
        