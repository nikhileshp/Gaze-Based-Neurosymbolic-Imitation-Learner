from nudge.env import NudgeBaseEnv
import torch

try:
    env = NudgeBaseEnv.from_name('seaquest', mode='logic')
    print('pred2action:', env.pred2action)
    print('action_space:', env.action_space)
except Exception as e:
    # Print more details if it fails
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
