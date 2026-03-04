import torch
from nudge.env import NudgeBaseEnv
print("Starting environment test...")
try:
    print("Initializing environment Seaquest...")
    env = NudgeBaseEnv.from_name("seaquest", mode='logic')
    print("Environment initialized.")
    
    print("Resetting environment...")
    state = env.reset(seed=42)
    print("Environment reset successful.")
    
    print("Taking a step with NOOP...")
    # Map NOOP
    noop_idx = env.pred2action['noop']
    state, reward, done = env.step("noop_0", is_mapped=False)
    print(f"Step successful. Reward: {reward}, Done: {done}")
    
    env.close()
    print("Environment closed.")
except Exception as e:
    print(f"Error during environment test: {e}")
print("Test complete")
