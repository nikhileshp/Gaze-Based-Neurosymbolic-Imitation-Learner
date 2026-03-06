import time
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from nsfr.agents.imitation_agent import ImitationAgent
from nsfr.env import NSFRBaseEnv
from scripts.evaluation.evaluate_model import evaluate_parallel, evaluate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing initialization...")
    # Just need an agent to pass in, don't actually need weights to match for timing test
    agent = ImitationAgent('seaquest', 'new', device)
    
    env = NSFRBaseEnv.from_name('seaquest', mode='logic')
    
    print("\n--- Testing Sequential Evaluation ---")
    start = time.time()
    rewards = evaluate(agent, env, num_episodes=5, max_steps=50, seed=42)
    seq_time = time.time() - start
    print(f"Sequential time for 5 episodes: {seq_time:.2f}s")
    
    print("\n--- Testing Parallel Evaluation ---")
    start = time.time()
    rewards_p = evaluate_parallel(agent, env_name='seaquest', num_episodes=5, num_workers=5, max_steps=50, seed=42, use_gaze=False)
    par_time = time.time() - start
    print(f"Parallel time for 5 episodes (5 workers): {par_time:.2f}s")
    
if __name__ == "__main__":
    main()
