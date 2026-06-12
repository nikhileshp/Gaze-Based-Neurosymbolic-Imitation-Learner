"""Headless sanity check for the Space Invaders input representation.

Runs the SI env for a few steps and prints, per frame, the populated logic-state
slots (slot, category, x, y, w, h) and the neural predicates that fire (>0.5).
A fast cross-check independent of the pygame/VNC GUI.

Run with the full-stack env:
    /home/nick/miniconda3/envs/grail/bin/python scripts/diagnostics/si_check_state.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "core"))
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from nsfr.env import NSFRBaseEnv  # noqa: E402
from nsfr.agents.imitation_agent import ImitationAgent  # noqa: E402

TYPE_NAMES = {0: "player", 1: "alien", 2: "shield", 3: "bullet", 4: "satellite"}


def _populated(logic: torch.Tensor):
    rows = []
    for i in range(logic.shape[0]):
        if logic[i, 0] == 1:
            rows.append((i, TYPE_NAMES.get(int(logic[i, 5]), "?"),
                         int(logic[i, 1]), int(logic[i, 2]),
                         int(logic[i, 3]), int(logic[i, 4])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default="space_invaders_root")
    ap.add_argument("--oc_mode", default="vision", choices=["ram", "vision"])
    ap.add_argument("--warmup", type=int, default=120, help="steps before reporting")
    ap.add_argument("--frames", type=int, default=3, help="frames to report")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = NSFRBaseEnv.from_name("space_invaders", mode="logic",
                                render_oc_overlay=False, oc_mode=args.oc_mode)
    agent = ImitationAgent("space_invaders", args.rules, device="cpu")
    agent.model.eval()
    atoms = agent.model.atoms

    logic, _ = env.reset(seed=args.seed)
    for _ in range(args.warmup):
        (logic, _), _, done = env.step("noop")
        if done:
            logic, _ = env.reset(seed=args.seed)

    for f in range(args.frames):
        print(f"\n===== frame {f} =====")
        rows = _populated(logic)
        print(f"populated slots: {len(rows)}")
        for r in rows:
            print("  slot %2d %-9s x=%3d y=%3d w=%2d h=%2d" % r)
        zs = logic.unsqueeze(0).float()
        with torch.no_grad():
            vT = agent.model.fc(zs, agent.model.atoms, agent.model.bk)[0]
        fired = [(str(atoms[i]), float(vT[i])) for i in range(len(atoms))
                 if float(vT[i]) > 0.5
                 and not str(atoms[i]).startswith(("noop", "fire", "move", "in_image", "."))]
        fired.sort(key=lambda kv: -kv[1])
        print(f"neural predicates firing (>0.5): {len(fired)}")
        for name, v in fired[:30]:
            print(f"  {v:.2f}  {name}")
        (logic, _), _, done = env.step("noop")
        if done:
            logic, _ = env.reset(seed=args.seed)

    env.close()


if __name__ == "__main__":
    main()
