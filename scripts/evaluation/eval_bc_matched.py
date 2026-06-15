"""Re-evaluate the saved full-data BC checkpoint under the SAME protocol as the NSFR
rulesets (gabril_compat=True, seed 12345, 20 full episodes) for a fair head-to-head.
"""
import statistics as st

import torch

from nsfr.env import NSFRBaseEnv
from scripts.evaluation.evaluate_bc_model import evaluate_bc_model

RUN_DIR = "trained_models/freeway/bc_learning_curve_epoch_100_seed_42_lr_0.001/100p"
N = 20
SEED = 12345


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = NSFRBaseEnv.from_name("freeway", mode="logic")
    print(f"device={dev}  BC run_dir={RUN_DIR}  episodes={N}  seed={SEED}  gabril_compat=True", flush=True)
    rewards = list(evaluate_bc_model(
        env, RUN_DIR, gaze_method="None", num_episodes=N, seed=SEED,
        device=dev, ckpt_prefix="best_", stack=1, gabril_compat=True,
    ))
    m = sum(rewards) / len(rewards); s = st.pstdev(rewards)
    print(f"\n=== BC (full data) — MATCHED protocol (gabril, seed 12345, 20 ep) ===", flush=True)
    print(f"  BC  mean={m:.2f} +/- {s:.2f}  min={min(rewards):.0f} max={max(rewards):.0f}", flush=True)
    print(f"  rewards={[round(r,1) for r in rewards]}", flush=True)
    print("EVAL_BC_MATCHED DONE", flush=True)


if __name__ == "__main__":
    main()
