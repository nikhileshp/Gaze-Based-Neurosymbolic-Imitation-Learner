"""Evaluate the three freeway policies on the real Freeway env (full episodes).

Policies (all on the `default_xclose` ruleset, up:-. unconditional):
  1. max + nll  — baseline (always-up)
  2. max + bce  — BCE pins up to base rate, calibrated noop overrides
  3. linear + ce — unnormalized learnable readout + softmax cross-entropy

Each policy's latest checkpoint is auto-resolved, loaded into an ImitationAgent
built with the matching action_head, and rolled out for N full episodes with the
same gabril-compatible settings training uses. Reports mean/std/min/max reward.

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 SDL_VIDEODRIVER=dummy WANDB_MODE=disabled \
    ~/miniconda3/envs/grail/bin/python scripts/evaluation/eval_three_policies.py [N_EPISODES]
"""
import sys
import glob
import os

import torch

from core.nsfr.agents.imitation_agent import ImitationAgent
from scripts.evaluation.evaluate_model import evaluate_parallel

ENV = "freeway"
RULES = "default_xclose"
N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MAX_STEPS = 20000          # large => full Freeway episodes (game ends on its own timer)
SEED = 12345               # held-out seed (training used seed 42) => "test" rollouts

# (display name, loss tag in path, action_head)
POLICIES = [
    ("max + nll   (baseline)", "nll", "max"),
    ("max + bce", "bce", "max"),
    ("linear + ce", "ce", "linear"),
]


def latest_checkpoint(loss_tag: str) -> str | None:
    """Highest-epoch .pth in the most recent run dir for this loss tag."""
    base = f"trained_models/{ENV}/nsfr/{RULES}_rules_*_lr_{loss_tag}_*/full_ep/*"
    run_dirs = sorted(glob.glob(base))
    if not run_dirs:
        return None
    for run_dir in reversed(run_dirs):                       # most recent first
        ckpts = glob.glob(os.path.join(run_dir, "epoch_*.pth"))
        if not ckpts:
            continue
        ckpts.sort(key=lambda p: int(p.split("epoch_")[-1].split(".")[0]))
        return ckpts[-1]
    return None


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  episodes={N_EPISODES}  max_steps={MAX_STEPS}  seed={SEED}\n")

    results = []
    for name, loss_tag, head in POLICIES:
        ckpt = latest_checkpoint(loss_tag)
        if ckpt is None:
            print(f"[{name}] no checkpoint found — skipping")
            results.append((name, None))
            continue
        print(f"[{name}] loading {ckpt}  (action_head={head})")
        agent = ImitationAgent(ENV, RULES, device, action_head=head)
        agent.load(ckpt)
        rewards = evaluate_parallel(
            agent, env_name=ENV, num_episodes=N_EPISODES, seed=SEED,
            max_steps=MAX_STEPS, use_gaze=False, gabril_compat=True,
            train_run=False, verbose=False,
        )
        rewards = list(rewards)
        import statistics as st
        mean = sum(rewards) / len(rewards)
        std = st.pstdev(rewards) if len(rewards) > 1 else 0.0
        results.append((name, (mean, std, min(rewards), max(rewards), rewards)))
        print(f"[{name}] mean={mean:.2f}  std={std:.2f}  min={min(rewards):.0f}  max={max(rewards):.0f}\n")

    print("\n==================== SUMMARY (Freeway, full episodes) ====================")
    print(f"{'policy':<26} {'mean':>7} {'std':>6} {'min':>5} {'max':>5}")
    for name, r in results:
        if r is None:
            print(f"{name:<26}  (no checkpoint)")
        else:
            mean, std, lo, hi, _ = r
            print(f"{name:<26} {mean:>7.2f} {std:>6.2f} {lo:>5.0f} {hi:>5.0f}")


if __name__ == "__main__":
    main()
