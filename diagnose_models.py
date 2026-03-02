import torch
import numpy as np
from nudge.agents.imitation_agent import ImitationAgent
from nudge.env import NudgeBaseEnv
import os

def compare_models(model_path1, model_path2, env_name="seaquest", rules="new", num_states=10):
    device = torch.device("cpu")
    
    print(f"Loading Model 1: {model_path1}")
    agent1 = ImitationAgent(env_name, rules, device)
    agent1.load(model_path1)
    agent1.model.eval()
    
    print(f"Loading Model 2: {model_path2}")
    agent2 = ImitationAgent(env_name, rules, device)
    agent2.load(model_path2)
    agent2.model.eval()
    
    # Check weight differences
    weights1 = agent1.model.state_dict()
    weights2 = agent2.model.state_dict()
    
    diff_count = 0
    total_params = 0
    for key in weights1:
        if key in weights2:
            diff = torch.abs(weights1[key] - weights2[key]).sum().item()
            if diff > 1e-6:
                diff_count += 1
            total_params += weights1[key].numel()
    
    print(f"Number of parameter tensors that differ: {diff_count} / {len(weights1)}")
    
    # Compare predictions
    env = NudgeBaseEnv.from_name(env_name, mode='logic')
    
    print(f"\nComparing predictions for {num_states} states...")
    
    mismatches = 0
    for i in range(num_states):
        state = env.reset(seed=i+42)
        logic_state, _ = state
        logic_state_tensor = torch.tensor(logic_state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            probs1 = agent1.model(logic_state_tensor)
            probs2 = agent2.model(logic_state_tensor)
            
            action1 = torch.argmax(probs1, dim=1).item()
            action2 = torch.argmax(probs2, dim=1).item()
            
            prednames = agent1.model.get_prednames()
            
            if action1 != action2:
                print(f"State {i}: Action Mismatch! Model1={prednames[action1]}, Model2={prednames[action2]}")
                print(f"  Probs1: {probs1[0, action1]:.4f}, Probs2: {probs2[0, action2]:.4f}")
                mismatches += 1
            else:
                # Even if actions are same, check if probs are exactly same
                prob_diff = torch.abs(probs1 - probs2).sum().item()
                if prob_diff > 1e-6:
                    print(f"State {i}: Same action ({prednames[action1]}), but probs differ by {prob_diff:.6f}")
                else:
                    print(f"State {i}: Same action ({prednames[action1]}), and probs are IDENTICAL.")
                    
    print(f"\nTotal mismatches in {num_states} states: {mismatches}")

if __name__ == "__main__":
    m1 = "out/imitation/atoms/seaquest_new_il_no_gaze.pth"
    m2 = "out/imitation/atoms/imitationseaquest_new_il_with_gazemap_values.pth"
    
    if os.path.exists(m1) and os.path.exists(m2):
        compare_models(m1, m2)
    else:
        print("Model files not found. Check paths.")
