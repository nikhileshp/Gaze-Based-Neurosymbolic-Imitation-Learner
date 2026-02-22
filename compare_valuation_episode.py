import os
import argparse
import torch
import numpy as np
import cv2
from collections import deque

from nudge.agents.imitation_agent import ImitationAgent
from nudge.env import NudgeBaseEnv
from scripts.gaze_predictor import Human_Gaze_Predictor
from nudge.utils import make_deterministic

def preprocess_frame(frame):
    """Convert raw 210x160x3 RGB frame to 84x84 grayscale frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized

def run_comparison():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the agent .pth file")
    parser.add_argument("--gaze_model_path", type=str, required=True, help="Path to the gaze predictor .pth file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Initializing Environment: seaquest...")
    env = NudgeBaseEnv.from_name("seaquest", mode="logic", render_oc_overlay=False)
    make_deterministic(42)

    print(f"Initializing Human_Gaze_Predictor from {args.gaze_model_path}...")
    gaze_predictor = Human_Gaze_Predictor("seaquest")
    gaze_predictor.init_model(args.gaze_model_path)
    gaze_predictor.model.eval()

    print(f"Initializing ImitationAgent from {args.model_path}...")
    agent = ImitationAgent("seaquest", "new", device=device)
    agent.load(args.model_path)
    agent.model.eval()

    state = env.reset(seed=42)

    frame_buffer = deque(maxlen=4)
    initial_rgb = env.get_rgb_frame()
    initial_gray = preprocess_frame(initial_rgb)
    for _ in range(4):
        frame_buffer.append(initial_gray)

    done = False
    total_frames = 0
    diff_frames = 0

    print("\nStarting episode 1...")
    
    while not done:
        # 1. Prepare Gaze Tensor
        img_stack = np.stack(frame_buffer, axis=-1)
        input_tensor = torch.tensor(img_stack, dtype=torch.float32, device=gaze_predictor.device)
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) # (1, 4, 84, 84)
        
        with torch.no_grad():
            gaze_pred = gaze_predictor.model(input_tensor)
        gaze_tensor = gaze_pred.squeeze(0) # Keep as (1, 84, 84) for Heatmap shape detection

        # 2. Extract Logic Tensor
        logic_state, _ = state
        logic_state_tensor = torch.tensor(logic_state, dtype=torch.float32, device=agent.device).unsqueeze(0)

        # 3. Predict WITHOUT gaze
        with torch.no_grad():
            _ = agent.model(logic_state_tensor, gaze=None)
            v0_no_gaze = agent.model.V_0.squeeze(0).clone().detach()

        # 4. Predict WITH gaze
        with torch.no_grad():
            _ = agent.model(logic_state_tensor, gaze=gaze_tensor)
            v0_gaze = agent.model.V_0.squeeze(0).clone().detach()

        # 5. Compare Valuation Vectors
        diff = torch.abs(v0_no_gaze - v0_gaze)
        max_diff = diff.max().item()

        if max_diff > 1e-4:
            if diff_frames == 0:
                print(f"DEBUG [Frame {total_frames}]: First difference detected! Max diff: {max_diff:.4f}")
                # Print which atoms differ
                for i in range(len(diff)):
                    if diff[i].item() > 1e-4:
                        print(f"  {agent.model.atoms[i]}: {v0_no_gaze[i].item():.3f} vs {v0_gaze[i].item():.3f}")
            diff_frames += 1
        elif total_frames == 10:
            print(f"DEBUG [Frame {total_frames}]: Still no difference.")
            print(f"Gaze tensor max: {gaze_tensor.max().item():.3f}, min: {gaze_tensor.min().item():.3f}")
            print(f"V0 sum without gaze: {v0_no_gaze.sum().item():.3f}")
            print(f"V0 sum with gaze: {v0_gaze.sum().item():.3f}")
            
            # Print logic_state_tensor[0, 0] (Player) and [0,1] (Enemy) visibility
            print(f"Player visibility flag: {logic_state_tensor[0, 0, 0].item()}")
            print(f"Enemy1 visibility flag: {logic_state_tensor[0, 1, 0].item()}")
            print(f"Enemy2 visibility flag: {logic_state_tensor[0, 2, 0].item()}")

        # 6. Step the environment
        # Use the gaze model's valuation to test realistic stepping scenarios
        action_idx = agent.act(logic_state_tensor, gaze=gaze_tensor)
        prednames = agent.model.get_prednames()
        predicate = prednames[action_idx]
        
        state, _, done = env.step(predicate)

        if not done:
            next_rgb = env.get_rgb_frame()
            next_gray = preprocess_frame(next_rgb)
            frame_buffer.append(next_gray)
            
        total_frames += 1

    print(f"\n--- Results ---")
    print(f"Episode complete. Total frames processed: {total_frames}")
    print(f"Frames modifying V_0 valuation (with vs without gazemap): {diff_frames} ({(diff_frames/max(1, total_frames))*100:.2f}%)")

if __name__ == "__main__":
    run_comparison()
