"""Evaluate a set of freeway policies (ruleset x loss x action_head) on full episodes.

Each argv entry is "rules,loss,head" (head in {max,linear}). The latest checkpoint for
that (rules, loss) run is auto-resolved, loaded into an ImitationAgent with the matching
head, and rolled out for N full episodes. Prints a mean/std reward table.

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 SDL_VIDEODRIVER=dummy WANDB_MODE=disabled \
    python scripts/evaluation/eval_ablation.py conditional_xclose,nll,max conditional_approach,nll,max ...
"""
import sys
import glob
import os
import statistics as st

import torch

from core.nsfr.agents.imitation_agent import ImitationAgent
from scripts.evaluation.evaluate_model import evaluate_parallel

ENV = "freeway"
N_EPISODES = 20
MAX_STEPS = 20000          # full Freeway episodes
SEED = 12345               # held-out seed


def latest_checkpoint(rules: str, loss_tag: str) -> str | None:
    # exact `_lr_{loss}_` segment so 'nll' doesn't match other tags; newest run by mtime
    base = f"trained_models/{ENV}/nsfr/{rules}_rules_*_lr_{loss_tag}_td_*/full_ep/*"
    for run_dir in sorted(glob.glob(base), key=os.path.getmtime, reverse=True):
        ckpts = glob.glob(os.path.join(run_dir, "epoch_*.pth"))
        if ckpts:
            ckpts.sort(key=lambda p: int(p.split("epoch_")[-1].split(".")[0]))
            return ckpts[-1]
    return None


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = [s.split(",") for s in sys.argv[1:]]
    print(f"device={device}  episodes={N_EPISODES}  seed={SEED}\n")

    rows = []
    for rules, loss_tag, head in specs:
        name = f"{rules} ({head}+{loss_tag})"
        ckpt = latest_checkpoint(rules, loss_tag)
        if ckpt is None:
            print(f"[{name}] no checkpoint — skipping"); rows.append((name, None)); continue
        print(f"[{name}] {ckpt}")
        agent = ImitationAgent(ENV, rules, device, action_head=head)
        agent.load(ckpt)
        rewards = list(evaluate_parallel(
            agent, env_name=ENV, num_episodes=N_EPISODES, seed=SEED,
            max_steps=MAX_STEPS, use_gaze=False, gabril_compat=True,
            train_run=False, verbose=False,
        ))
        mean = sum(rewards) / len(rewards)
        std = st.pstdev(rewards) if len(rewards) > 1 else 0.0
        rows.append((name, (mean, std, min(rewards), max(rewards))))
        print(f"[{name}] mean={mean:.2f} std={std:.2f} min={min(rewards):.0f} max={max(rewards):.0f}\n")

    print("\n==================== ABLATION SUMMARY (Freeway, full episodes) ====================")
    print(f"{'policy':<30} {'mean':>7} {'std':>6} {'min':>5} {'max':>5}")
    for name, r in rows:
        if r is None:
            print(f"{name:<30}  (no checkpoint)")
        else:
            mean, std, lo, hi = r
            print(f"{name:<30} {mean:>7.2f} {std:>6.2f} {lo:>5.0f} {hi:>5.0f}")


if __name__ == "__main__":
    main()
