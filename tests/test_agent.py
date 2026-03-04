import torch
from nudge.agents.imitation_agent import ImitationAgent
print("Starting agent test...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Initializing ImitationAgent Seaquest...")
    agent = ImitationAgent("seaquest", "new", device)
    print("Agent initialized.")
    
    model_path = "models/nsfr/seaquest/no_gaze/1_ep/best.pth"
    import os
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        agent.load(model_path)
        print("Model loaded.")
    else:
        print(f"Model not found at {model_path}")

except Exception as e:
    print(f"Error during agent test: {e}")
    import traceback
    traceback.print_exc()
print("Test complete")
