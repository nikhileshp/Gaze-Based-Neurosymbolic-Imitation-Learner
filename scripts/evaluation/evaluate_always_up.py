"""Evaluate a fixed 'always UP' policy on Freeway for N episodes."""
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nsfr.env import NSFRBaseEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rewards = []
    for ep in range(args.episodes):
        env = NSFRBaseEnv.from_name('freeway', mode='logic')
        seed = args.seed + 1000 * ep
        env.reset(seed=seed)
        total_reward = 0.0
        for _ in range(args.max_steps):
            _, reward, done = env.step('up')
            total_reward += reward
            if done:
                break
        env.close()
        rewards.append(total_reward)
        print(f"  Episode {ep+1}/{args.episodes}: Reward = {total_reward}")

    rewards = np.array(rewards)
    print(f"\nAlways-UP Policy over {args.episodes} episodes:")
    print(f"  Mean = {rewards.mean():.2f}  Std = {rewards.std():.2f}")
    print(f"  Min  = {rewards.min():.2f}  Max = {rewards.max():.2f}")


if __name__ == '__main__':
    main()
