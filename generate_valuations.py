import torch
import os
from tqdm import tqdm
from scripts.data_utils import PtDataset
from nudge.agents.imitation_agent import ImitationAgent
import argparse

def generate_valuations(model_path, data_path, output_path, device_name, num_episodes=None, sort_by=None):
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Load Agent
    print(f"Loading agent from {model_path}...", flush=True)
    agent = ImitationAgent("seaquest", "new", device)
    agent.load(model_path)
    agent.model.eval()
    print("Agent loaded.", flush=True)

    # Load Dataset
    print(f"Loading dataset from {data_path} (sort_by={sort_by}, num_episodes={num_episodes})...", flush=True)
    dataset = PtDataset(data_path, use_gaze=False, num_episodes=num_episodes, sort_by=sort_by)
    print(f"Dataset loaded. Length: {len(dataset)}", flush=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    valuations_by_episode = {}

    print("Generating valuations...", flush=True)

    with torch.no_grad():
        for logic_states, actions, gazes, ep_nums, step_idxs in tqdm(dataloader):
            logic_states = logic_states.to(device)
            # Pass through model to get V_T
            agent.model(logic_states, None)
            vT = agent.model.V_T.cpu() # (B, num_atoms)
            
            for i in range(len(ep_nums)):
                ep_id = ep_nums[i].item()
                if ep_id not in valuations_by_episode:
                    valuations_by_episode[ep_id] = []
                valuations_by_episode[ep_id].append(vT[i])

    # Convert lists to stacked tensors
    print("Stacking tensors...")
    for ep_id in valuations_by_episode:
        valuations_by_episode[ep_id] = torch.stack(valuations_by_episode[ep_id])

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(valuations_by_episode, output_path)
    print(f"Valuations saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="out/imitation/atoms/seaquest_new_il_no_gaze.pth")
    parser.add_argument("--data_path", type=str, default="data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt")
    parser.add_argument("--output_path", type=str, default="models/nsfr/seaquest/_no_gaze/valuation.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to process")
    parser.add_argument("--sort_by", type=str, choices=["length", "reward_per_step"], default=None, help="Sort episodes by length or reward_per_step")
    args = parser.parse_args()

    generate_valuations(args.model_path, args.data_path, args.output_path, args.device, args.num_episodes, args.sort_by)
