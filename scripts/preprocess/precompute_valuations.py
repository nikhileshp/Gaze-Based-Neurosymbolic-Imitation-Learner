import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.utils.utils import PtDataset
from nsfr.agents.imitation_agent import ImitationAgent
from nsfr.env import NSFRBaseEnv
import argparse



DATASET_PATH = 'data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="seaquest", help="Environment name")
    parser.add_argument("--rules", type=str, default="claude_extensive", help="Ruleset name")
    parser.add_argument("--dataset", type=str, default=None, help="Path to .pt dataset file (from convert_trajectories_to_pt.py)")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save pre-computed valuations")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--use_gaze", type=bool, default=False, help="Use gaze (GRAIL) or not (NSFR)")
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to load from .pt dataset")
    args = parser.parse_args()
    
    if args.use_gaze:
        raise NotImplementedError("GRAIL (gaze based valuations) not implemented yet")
        # OUTPUT_PATH = f"trained_models/{args.env}/grail/valuations_{args.rules}.pt"
    else:
        OUTPUT_PATH = f"trained_models/{args.env}/nsfr/valuations_{args.rules}.pt"

    if args.dataset is None:
        args.dataset = DATASET_PATH
    if args.output_path is None:
        args.output_path = OUTPUT_PATH
    if args.device is None:
        args.device = DEVICE
    if args.num_episodes is None:
        args.num_episodes = 28

    print(f"Loading dataset from {DATASET_PATH} ...")
    dataset = PtDataset(DATASET_PATH, num_episodes=28)
    # Make sure dataset returns ep_nums and step_idxs
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    print("Initializing logic model...")
    env = NSFRBaseEnv.from_name('seaquest', mode='logic')
    agent = ImitationAgent('seaquest', 'new', device=DEVICE)
    agent.model.eval()

    valuations = {}
    print("Pre-computing valuations...")
    for batch in tqdm(dataloader):
        states, actions, gazes, ep_nums, step_idxs = batch
        states = states.to(DEVICE)
        
        with torch.no_grad():
            vT = agent.model.fc(states, agent.model.atoms, agent.model.bk)
            vT = vT.cpu()
        
        for i in range(len(ep_nums)):
            ep_id = ep_nums[i].item()
            s_idx = step_idxs[i].item()
            
            if ep_id not in valuations:
                valuations[ep_id] = {}
            valuations[ep_id][s_idx] = vT[i]

    # Convert inner dicts to lists sorted by step_idx
    final_valuations = {}
    for ep_id, steps in valuations.items():
        max_step = max(steps.keys())
        val_list = [torch.zeros(len(agent.model.atoms)) for _ in range(max_step + 1)]
        for s_idx, v in steps.items():
            val_list[s_idx] = v
        final_valuations[ep_id] = torch.stack(val_list)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"Saving pre-computed valuations to {OUTPUT_PATH} ...")
    torch.save(final_valuations, OUTPUT_PATH)
    print("Done!")

if __name__ == '__main__':
    main()
