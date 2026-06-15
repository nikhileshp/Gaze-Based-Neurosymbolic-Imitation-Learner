"""One-shot diagnostic: why does freeway NSFR plateau at the majority-class baseline?

Builds the ImitationAgent for the default ruleset, runs a forward pass on a real
batch from the training .pt, and reports:
  - mean valuation per rule (head predicate),
  - mean action-score per primitive action (after max-aggregation),
  - the predicted-action distribution (argmax) vs the ground-truth distribution.

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 SDL_VIDEODRIVER=dummy \
    ~/miniconda3/envs/grail/bin/python scripts/diagnostics/probe_nsfr_freeway.py
"""
import sys
import collections

import torch

from core.nsfr.agents.imitation_agent import ImitationAgent
from core.utils.utils import PtDataset

DATASET = "data/freeway/Freeway_logicfmt5_obj12.pt"
ENV = "freeway"
RULES = sys.argv[1] if len(sys.argv) > 1 else "default"
N = 4096


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    agent = ImitationAgent(ENV, RULES, device, target_diagonal=0.99, random_init=False)
    model = agent.unwrapped_model
    prednames = model.get_prednames()
    print(f"prednames = {prednames}")
    print(f"primitive_action_map = {agent.primitive_action_map}")

    ds = PtDataset(DATASET, use_gaze=False, num_episodes=None,
                   max_action=agent.num_actions - 1, env_name=ENV)
    n = min(N, len(ds))
    states = torch.stack([ds.logic[i] for i in range(n)]).to(device)
    actions = torch.stack([ds.actions[i] for i in range(n)]).to(device)
    print(f"batch: states={tuple(states.shape)} actions={tuple(actions.shape)}")

    with torch.no_grad():
        probs, action_scores = agent.predict(states, None)

    print("\n=== mean rule valuation (probs columns) ===")
    mp = probs.mean(0)
    mx = probs.max(0).values
    for i, p in enumerate(prednames):
        print(f"  {p:28s} mean={mp[i].item():.4f}  max={mx[i].item():.4f}")

    print("\n=== mean action score (after max-agg over rules) ===")
    inv = {v: k for k, v in agent.primitive_action_map.items()}
    msc = action_scores.mean(0)
    for a in range(agent.num_actions):
        print(f"  action {a} ({inv.get(a, a):>5}): mean_score={msc[a].item():.4f}")

    pred = action_scores.argmax(1)
    print("\n=== predicted vs ground-truth action distribution ===")
    pc = collections.Counter(pred.tolist())
    gc = collections.Counter(actions.tolist())
    for a in range(agent.num_actions):
        print(f"  action {a} ({inv.get(a, a):>5}): "
              f"pred={pc.get(a, 0)/n:.4f}  truth={gc.get(a, 0)/n:.4f}")
    acc = (pred == actions).float().mean().item()
    print(f"\naccuracy on this batch = {acc:.4f}")
    # how often is up's score the single max AND ties resolved to up?
    up_idx = agent.primitive_action_map.get("up")
    if up_idx is not None:
        up_is_max = (action_scores.argmax(1) == up_idx).float().mean().item()
        up_score_ge_all = (action_scores[:, up_idx:up_idx+1] >= action_scores).all(1).float().mean().item()
        print(f"argmax==up fraction          = {up_is_max:.4f}")
        print(f"up_score >= every other score = {up_score_ge_all:.4f}")


if __name__ == "__main__":
    sys.exit(main())
