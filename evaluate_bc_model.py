import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque

# Import GABRIL CNN mapping
from baselines.models.linear_models import Encoder

PRIMITIVE_ACTIONS = {0: 'noop', 1: 'fire', 2: 'up', 3: 'right', 4: 'left', 5: 'down'}

def load_bc_model(run_dir, gaze_method="None", device="cuda", ckpt_prefix="best_"):
    """
    Loads the BC/AGIL CNN models from a specified directory.
    
    Args:
        run_dir (str): Path containing the encoder.pth, pre_actor.pth, etc.
        gaze_method (str): "None", "AGIL", or "Mask".
        device (str): "cuda" or "cpu".
        ckpt_prefix (str): Prefix of the checkpoints (default: "best_").
        
    Returns:
        tuple: (encoder, pre_actor, actor, encoder_agil)
    """
    dev = torch.device(device)
    
    # Default model params from train_bc_pt.py
    embedding_dim = 64
    num_hiddens = 128
    num_residual_layers = 2
    num_residual_hiddens = 32
    z_dim = 256
    action_dim = 18 # Fallback default
    encoder_out_dim = 8 * 8 * embedding_dim  # 4096
    
    actor_ckpt_path = None

    # Determine exact checkpoint prefix and path
    if run_dir and os.path.isdir(run_dir):
        if ckpt_prefix == "best_" and not os.path.exists(f"{run_dir}/best_actor.pth"):
            print(f"Warning: best_actor.pth not found in {run_dir}. Searching for latest epoch...")
            ep_dirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and d.startswith("ep")]
            if ep_dirs:
                ep_dirs.sort(key=lambda x: int(x[2:]))
                latest_ep_dir = ep_dirs[-1]
                print(f"Found latest epoch directory: {latest_ep_dir}")
                run_dir = os.path.join(run_dir, latest_ep_dir)
                ckpt_prefix = f"{latest_ep_dir}_"
            else:
                print("No epoch subdirectories found either.")
                
        actor_ckpt_path = f"{run_dir}/{ckpt_prefix}actor.pth"
        if os.path.exists(actor_ckpt_path):
            try:
                # Peek at the model weights to get the action dimension
                dummy_ckpt = torch.load(actor_ckpt_path, map_location='cpu', weights_only=False)
                if '2.bias' in dummy_ckpt:
                    action_dim = dummy_ckpt['2.bias'].shape[0]
                    print(f"Inferred action_dim={action_dim} from weights.")
            except Exception as e:
                print(f"Error inferring action_dim from weights: {e}")

    # 1. Initialize Networks
    encoder = Encoder(4, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens).to(dev)
    
    pre_actor = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(encoder_out_dim, z_dim),
        nn.ReLU()
    ).to(dev)
    
    actor = nn.Sequential(
        nn.Linear(z_dim, z_dim), nn.ReLU(),
        nn.Linear(z_dim, action_dim)
    ).to(dev)

    encoder_agil = None
    if gaze_method == "AGIL":
        encoder_agil = Encoder(4, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens).to(dev)

    # 2. Load Weights if directory provided
    if actor_ckpt_path and os.path.exists(actor_ckpt_path):
        try:
            encoder.load_state_dict(torch.load(f"{run_dir}/{ckpt_prefix}encoder.pth", map_location=dev))
            pre_actor.load_state_dict(torch.load(f"{run_dir}/{ckpt_prefix}pre_actor.pth", map_location=dev))
            actor.load_state_dict(torch.load(actor_ckpt_path, map_location=dev))
            
            if encoder_agil is not None:
                encoder_agil.load_state_dict(torch.load(f"{run_dir}/{ckpt_prefix}encoder_agil.pth", map_location=dev))
        except FileNotFoundError as e:
            print(f"Error loading weights: {e}")
            print("Returning randomly initialized models.")
    else:
        print(f"Warning: {run_dir} not found or no actor checkpoint resolved. Returning randomly initialized models.")

    # 3. Set to Evaluation Mode
    encoder.eval()
    pre_actor.eval()
    actor.eval()
    if encoder_agil is not None:
        encoder_agil.eval()
        
    return encoder, pre_actor, actor, encoder_agil


def evaluate_bc_model(env, run_dir, gaze_method="None", num_episodes=10, seed=42, device="cuda", use_gazemap=False, gaze_model_path="seaquest_gaze_predictor_2.pth", ckpt_prefix="best_"):
    """
    Loads a pretrained BC/AGIL baseline and runs it in the provided environment.
    
    Args:
        env (gym.Env / NudgeBaseEnv): The environment instance to step through.
        run_dir (str): Path containing the pretrained .pth files.
        gaze_method (str): Method used ("None", "AGIL", "Mask").
        num_episodes (int): Number of episodes to evaluate.
        seed (int): Random seed for environments.
        device (str): Compute device ("cuda" or "cpu").
        use_gazemap (bool): If True, instantiate live Human_Gaze_Predictor to supply heatmaps.
        gaze_model_path (str): Path to the gaze predictor model.
        ckpt_prefix (str): Prefix of the checkpoints (default: "best_").
        
    Returns:
        list: Total rewards for each episode.
    """
    dev = torch.device(device)
    encoder, pre_actor, actor, encoder_agil = load_bc_model(run_dir, gaze_method, device, ckpt_prefix=ckpt_prefix)
    
    gaze_predictor = None
    if (use_gazemap or gaze_method in ['ViSaRL', 'Mask', 'AGIL']) and gaze_method != "None":
        try:
            from scripts.gaze_predictor import Human_Gaze_Predictor
            print(f"Initializing Test-Time Gaze Predictor from {gaze_model_path}...")
            # We pass 'seaquest' string directly
            gaze_predictor = Human_Gaze_Predictor("seaquest")
            gaze_predictor.init_model(gaze_model_path)
            gaze_predictor.model.eval()
        except ImportError:
            print("Warning: Could not import Human_Gaze_Predictor! Gaze will drop to 0.0.")
            
    rewards = []
    
    # Map for nudge env from atari base actions (0-17)
    valid_actions = {0: 'noop', 1: 'fire', 2: 'up', 3: 'right', 4: 'left', 5: 'down', 6: 'upright', 7: 'upleft', 8: 'downright', 9: 'downleft', 10: 'upfire', 11: 'rightfire', 12: 'leftfire', 13: 'downfire', 14: 'uprightfire', 15: 'upleftfire', 16: 'downrightfire', 17: 'downleftfire'}
    
    for i in range(num_episodes):
        try:
            state = env.reset(seed=seed + i)
        except TypeError:
            state = env.reset()
            
        done = False
        total_r = 0.0
        
        # Initialize Gaze temporal buffer AND frame buffer
        frame_buffer = deque(maxlen=4)
        raw_frame = env.get_rgb_frame() if hasattr(env, 'get_rgb_frame') else (env.render() if hasattr(env, 'render') else state)
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
        for _ in range(4): frame_buffer.append(gray)
        
        from tqdm import tqdm
        pbar = tqdm(desc=f"Episode {i+1}")
        step_count = 0
        action_counts = {}
        
        while not done and step_count < 10000:
            step_count += 1
            # We need the raw RGB frame for the CNN
            if hasattr(env, 'get_rgb_frame'):
                raw_frame = env.get_rgb_frame()
            else:
                raw_frame = env.render() if hasattr(env, 'render') else state
                
            # Grayscale -> 84x84 -> normalize [0, 1]
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
            
            img_stack = np.stack(frame_buffer, axis=-1) # (84, 84, 4)
            xx = torch.tensor(img_stack, dtype=torch.float32, device=dev).permute(2, 0, 1).unsqueeze(0) # (1, 4, 84, 84)
            
            gg = torch.zeros(1, 1, 84, 84, device=dev)
            if gaze_predictor is not None:
                # Gaze predictor still takes the (1, 4, 84, 84) stack
                input_tensor = torch.tensor(img_stack, dtype=torch.float32, device=gaze_predictor.device)
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) # (1, 4, 84, 84)
                with torch.no_grad():
                    gaze_pred = gaze_predictor.model(input_tensor)
                gg = gaze_pred.to(dev) # Output is (1, 1, 84, 84)
            
            with torch.no_grad():
                if gaze_method == 'Mask':
                    xx_in = xx * gg
                else:
                    xx_in = xx
                    
                z = encoder(xx_in)
                if gaze_method == 'AGIL' and encoder_agil is not None:
                    z = (z + encoder_agil(xx * gg)) / 2
                    
                logits = actor(pre_actor(z))
                action_idx = logits.argmax(dim=1).item()
                
            # Map network output back to valid env action name for logging
            action_str = valid_actions.get(action_idx, "noop")
            action_counts[action_str] = action_counts.get(action_str, 0) + 1
            
            # Step the environment with the raw integer action
            step_result = env.step(action_idx, is_mapped=True)
            if len(step_result) == 3:
                state, reward, done = step_result
            else:
                state, reward, done, truncated, info = step_result
                if truncated: done = True
                
            total_r += reward
            
            # Update frame buffer for gaze with the new environment state
            if gaze_predictor is not None and not done:
                next_raw = env.get_rgb_frame() if hasattr(env, 'get_rgb_frame') else (env.render() if hasattr(env, 'render') else state)
                next_gray = cv2.cvtColor(next_raw, cv2.COLOR_RGB2GRAY)
                next_gray = cv2.resize(next_gray, (84, 84), interpolation=cv2.INTER_AREA) / 255.0
                frame_buffer.append(next_gray)
            
            pbar.update(1)
        pbar.close()
            
        rewards.append(total_r)
        print(f"Episode {i+1}/{num_episodes} - Reward: {total_r} - Steps: {step_count}")
        print(f"  Action Distribution: {action_counts}")
        
    return rewards

# Example Usage Block (can be executed directly)
if __name__ == "__main__":
    import argparse
    from nudge.env import NudgeBaseEnv
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to folder containing BC checkpoints")
    parser.add_argument("--gaze_method", type=str, default="None", choices=["None", "AGIL", "Mask"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_gazemap", action="store_true", help="Pipe live 84x84 gaze predictions into logic agent during testing")
    parser.add_argument("--gaze_model_path", type=str, default="seaquest_gaze_predictor_2.pth")
    parser.add_argument("--ckpt_prefix", type=str, default="best_")
    args = parser.parse_args()
    
    print(f"Initializing Nudge Seaquest environment for {args.episodes} episodes...")
    test_env = NudgeBaseEnv.from_name("seaquest", mode='logic')
    
    print(f"Loading {args.gaze_method} BC Model from: {args.run_dir} with prefix {args.ckpt_prefix}")
    eval_rewards = evaluate_bc_model(test_env, args.run_dir, gaze_method=args.gaze_method, 
                                     num_episodes=args.episodes, seed=args.seed, device=args.device,
                                     use_gazemap=args.use_gazemap, gaze_model_path=args.gaze_model_path,
                                     ckpt_prefix=args.ckpt_prefix)
    
    print(f"\\nFinal Evaluation over {args.episodes} episodes:")
    print(f"Mean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
