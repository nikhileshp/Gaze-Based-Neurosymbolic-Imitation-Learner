import gymnasium as gym
import ale_py

env = gym.make("ALE/Seaquest-v5")
print(f"Action meanings: {env.unwrapped.get_action_meanings()}")
env.close()
