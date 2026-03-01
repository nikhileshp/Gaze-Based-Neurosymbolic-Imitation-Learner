import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.data_utils import PtDataset
from nudge.agents.imitation_agent import ImitationAgent
from nudge.env import NudgeBaseEnv

DATASET_PATH = 'data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt'
OUTPUT_PATH = 'models/nsfr/seaquest/_no_gaze/valuation.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print(f"Loading dataset from {DATASET_PATH} ...")
    dataset = PtDataset(DATASET_PATH, num_episodes=28)
    # Make sure dataset returns ep_nums and step_idxs
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    print("Initializing logic model...")
    env = NudgeBaseEnv.from_name('seaquest', mode='logic')
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
