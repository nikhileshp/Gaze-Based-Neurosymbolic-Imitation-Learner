import torch
from nudge.env import NudgeBaseEnv
from nudge.agents.imitation_agent import ImitationAgent
import time

print("Starting isolated evaluation loop test...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = ImitationAgent("seaquest", "new", device)
    agent.load("models/nsfr/seaquest/no_gaze/1_ep/best.pth")
    agent.model.eval()
    
    env = NudgeBaseEnv.from_name("seaquest", mode='logic')
    state = env.reset(seed=42)
    
    print("Agent and Env initialized. Starting loop...")
    
    for step in range(10):
        start = time.time()
        print(f"Step {step}...")
        
        logic_state, _ = state
        logic_state_tensor = torch.as_tensor(logic_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        
        print("  Calling agent.act()...")
        predicate = agent.act(logic_state_tensor)
        print(f"  Action chosen: {predicate}")
        
        print("  Calling env.step()...")
        state, reward, done = env.step(predicate)
        print(f"  Step returned. Done={done}. Time={time.time() - start:.2f}s")
        if done:
            break
            
    print("Loop test complete.")
except Exception as e:
    print(f"Error: {e}")
