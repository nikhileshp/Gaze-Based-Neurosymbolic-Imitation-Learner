import os
import argparse
import torch
import numpy as np
from nudge.agents.imitation_agent import ImitationAgent
from nudge.env import NudgeBaseEnv
from nudge.utils import make_deterministic

def evaluate(agent, env, num_episodes=5):
    """
    Evaluates the agent in the environment for a set number of episodes.
    Returns the list of total rewards for each episode.
    """
    agent.model.eval()
    episode_rewards = []

        
    for i in range(num_episodes):
        try:
            state = env.reset(seed=i + 42)
        except TypeError:
            print("Warning: env.reset() does not accept seed. Results may vary.")
            state = env.reset()
        done = False
        total_reward = 0
        while not done:
            # state is (logic_state, neural_state)
            logic_state, _ = state
            logic_state_tensor = torch.tensor(logic_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            
            # Select action index
            action_idx = agent.act(logic_state_tensor)
            
            # Map action index to predicate name
            prednames = agent.model.get_prednames()
            predicate = prednames[action_idx]
            
            # Step environment
            state, reward, done = env.step(predicate)
            total_reward += reward
            
        episode_rewards.append(total_reward)
        print(f"Episode {i+1}: Reward = {total_reward}")
        
    return episode_rewards

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Imitation Learning model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model (.pth)")
    parser.add_argument("--env", type=str, default="seaquest", help="Environment name")
    parser.add_argument("--rules", type=str, default="new", help="Ruleset name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--use_gaze", action="store_true", help="Use gaze data logic in model")
    parser.add_argument("--gaze_threshold", type=float, default=20.0, help="Gaze threshold if use_gaze is set")
    
    args = parser.parse_args()

    # Setup device
    device_name = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Set seed
    make_deterministic(args.seed)

    # Initialize Environment
    print(f"Initializing Environment: {args.env}...")
    env = NudgeBaseEnv.from_name(args.env, mode='logic')

    # Initialize Agent
    print(f"Initializing Agent for rules: {args.rules}...")
    gaze_threshold = args.gaze_threshold if args.use_gaze else None
    agent = ImitationAgent(args.env, args.rules, device, gaze_threshold=gaze_threshold)
    
    # Load Model
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
        
    print(f"Loading model from {args.model_path}...")
    agent.load(args.model_path)

    # Run Evaluation
    print(f"Starting evaluation for {args.episodes} episodes...")
    rewards = evaluate(agent, env, num_episodes=args.episodes)

    # Calculate Statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print("\n" + "="*30)
    print(f"Evaluation Results for {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Deviation: {std_reward:.2f}")
    print("="*30)

if __name__ == "__main__":
    main()
