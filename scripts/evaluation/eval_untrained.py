"""Eval UNTRAINED freeway policies on full episodes (W at block-diag init).

For rulesets with one clause per action head, the clause weight matrix W barely moves
during training (each head's softmax is over a single clause), so the untrained policy
is a faithful estimate of the trained one — useful for a fast game-reward comparison.

Run:  python scripts/evaluation/eval_untrained.py default,max conditional_xclose,max conditional_approach,max
"""
import sys
import statistics as st

import torch

from core.nsfr.agents.imitation_agent import ImitationAgent
from scripts.evaluation.evaluate_model import evaluate_parallel

ENV = "freeway"
N = 20
MAX_STEPS = 20000
SEED = 12345


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = [s.split(",") for s in sys.argv[1:]] or [["default", "max"], ["conditional_xclose", "max"], ["conditional_approach", "max"]]
    print(f"device={dev}  episodes={N}  seed={SEED}  (UNTRAINED policies)\n", flush=True)
    rows = []
    for rules, head in specs:
        a = ImitationAgent(ENV, rules, dev, action_head=head)   # no load => untrained
        rw = list(evaluate_parallel(a, env_name=ENV, num_episodes=N, seed=SEED,
                                    max_steps=MAX_STEPS, use_gaze=False, gabril_compat=True,
                                    train_run=False, verbose=False))
        m = sum(rw) / len(rw); s = st.pstdev(rw)
        rows.append((rules, m, s, min(rw), max(rw)))
        print(f"{rules:12s} mean={m:.2f} std={s:.2f} min={min(rw):.0f} max={max(rw):.0f}", flush=True)
    print("\n=== UNTRAINED game-reward (Freeway, 20 full episodes, seed 12345) ===", flush=True)
    for rules, m, s, lo, hi in rows:
        print(f"  {rules:12s} {m:6.2f} +/- {s:4.2f}   [{lo:.0f},{hi:.0f}]", flush=True)
    print("EVAL_UNTRAINED DONE", flush=True)


if __name__ == "__main__":
    main()
