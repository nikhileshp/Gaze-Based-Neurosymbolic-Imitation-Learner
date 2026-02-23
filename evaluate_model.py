import os
import argparse
import torch
import numpy as np
from collections import deque
import cv2

from nudge.agents.imitation_agent import ImitationAgent
from nudge.env import NudgeBaseEnv
from nudge.utils import make_deterministic

# Import gaze predictor if available
try:
    from scripts.gaze_predictor import Human_Gaze_Predictor
except ImportError:
    Human_Gaze_Predictor = None

def preprocess_frame(frame):
    """Convert raw 210x160x3 RGB frame to 84x84 grayscale frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0

def evaluate(agent, env, num_episodes=5, seed=42, gaze_predictor=None, log_interval=100, valuation_interval=50):
    """
    Evaluates the agent in the environment for a set number of episodes.
    Returns the list of total rewards for each episode.
    """
    agent.model.eval()
    episode_rewards = []
    if seed is not None:
        make_deterministic(seed)
        
    for i in range(num_episodes):
        try:
            if seed is not None:
                state = env.reset(seed=seed + i)
            else:
                state = env.reset()
        except TypeError:
            print("Warning: env.reset() does not accept seed. Results may vary.")
            state = env.reset()
            
        done = False
        total_reward = 0
        step_count = 0
        
        # Initialize the gaze frame buffer if gaze predictor is active
        frame_buffer = None
        if gaze_predictor is not None:
            frame_buffer = deque(maxlen=4)
            # Fetch the first frame and duplicate it 4 times to fill the initial buffer
            initial_rgb = env.get_rgb_frame()
            initial_gray = preprocess_frame(initial_rgb)
            for _ in range(4):
                frame_buffer.append(initial_gray)
                
        while not done:
            # Generate the gaze heatmap if predictor is available
            gaze_tensor = None
            if gaze_predictor is not None:
                # Shape buffer into (1, 4, 84, 84) NHWC -> NCHW handled inside predict_and_save style or directly here
                # Convert deque to numpy array of shape (84, 84, 4)
                img_stack = np.stack(frame_buffer, axis=-1)
                
                # Convert to tensor (1, 4, 84, 84)
                input_tensor = torch.tensor(img_stack, dtype=torch.float32, device=gaze_predictor.device)
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) # (1, 4, H, W)
                
                with torch.no_grad():
                    gaze_pred = gaze_predictor.model(input_tensor) # Outputs (1, 1, 84, 84) spatial softmax
                    
                gaze_tensor = gaze_pred.squeeze(0) # Reduce to (1, 84, 84) for the Valuation Module
            
            # state is (logic_state, neural_state)
            logic_state, _ = state
            logic_state_tensor = torch.tensor(logic_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            
            # Select action index
            # agent.act passes gaze down to the NSFR model if provided
            action_idx = agent.act(logic_state_tensor, gaze=gaze_tensor)
            
            # Print top atom valuations every valuation_interval steps
            if valuation_interval > 0 and step_count % valuation_interval == 0 and step_count > 0:
                if hasattr(agent.model, 'V_0') and agent.model.V_0 is not None:
                    v0 = agent.model.V_0.squeeze(0).detach().cpu()
                    atoms = agent.model.atoms
                    pairs = sorted(zip(atoms, v0.tolist()), key=lambda x: x[1], reverse=True)
                    visible_pairs = [(a, v) for a, v in pairs if str(a).startswith("visible_") and v > 0.01]
                    print(f"  --- visible_ Valuations at Step {step_count} ---")
                    for atom, val in visible_pairs:
                        print(f"    {val:.3f}  {atom}")
            
            # Map action index to predicate name
            prednames = agent.model.get_prednames()
            predicate = prednames[action_idx]
            
            # Step environment
            state, reward, done = env.step(predicate)
            total_reward += reward
            step_count += 1
            if log_interval > 0 and step_count % log_interval == 0:
                print(f"  Episode {i+1} | Step {step_count} | Cumulative Reward: {total_reward:.1f}")
            
            # Update gaze frame buffer
            if gaze_predictor is not None and not done:
                next_rgb = env.get_rgb_frame()
                next_gray = preprocess_frame(next_rgb)
                frame_buffer.append(next_gray)
            
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
    parser.add_argument("--log_interval", type=int, default=0, help="Print cumulative reward every N steps (0 to disable)")
    parser.add_argument("--valuation_interval", type=int, default=0, help="Print top atom valuations every N steps (0 to disable)")
    parser.add_argument("--use_gaze", action="store_true", help="Use gaze data logic in model")
    parser.add_argument("--gaze_threshold", type=float, default=20.0, help="Gaze threshold if use_gaze is set")
    parser.add_argument("--use_gazemap", action="store_true", help="Pipe live 84x84 gaze predictions into logic agent during testing")
    parser.add_argument("--gaze_model_path", type=str, default="seaquest_gaze_predictor_2.pth", help="Path to the .pth gaze predictor weights")
    
    args = parser.parse_args()

    # Setup device
    device_name = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")
    
    # Initialize Gaze Predictor 
    gaze_predictor = None
    if args.use_gazemap:
        if Human_Gaze_Predictor is None:
            print("Error: Could not import Human_Gaze_Predictor. Ensure gaze_predictor.py is in the scripts/ folder.")
            return
            
        print(f"Initializing Gaze Predictor from {args.gaze_model_path}...")
        gaze_predictor = Human_Gaze_Predictor(args.env)
        # Using default padding/stride/rho/lr for init_model
        gaze_predictor.init_model(args.gaze_model_path)
        gaze_predictor.model.eval()

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
    rewards = evaluate(agent, env, num_episodes=args.episodes, gaze_predictor=gaze_predictor, seed=args.seed,
                       log_interval=args.log_interval, valuation_interval=args.valuation_interval)

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
