"""
evaluate_hackatari_freeway.py

Zero-shot evaluation of trained Freeway models across HackAtari modification modes.
Evaluates NSFR, GRAIL (neurosymbolic) and BC/AGIL (neural pixel-based) models.

NSFR/GRAIL use RAM-based object detection at eval time — visual mods don't break them.
BC/AGIL use raw pixel frames — visual mods (invisible, phantom) will degrade them.
This asymmetry is the key experimental result.

Usage:
    cd <project_root>
    python scripts/evaluation/evaluate_hackatari_freeway.py
    python scripts/evaluation/evaluate_hackatari_freeway.py --episodes 50 --output results/hackatari_freeway.csv
    python scripts/evaluation/evaluate_hackatari_freeway.py --modes vanilla speed_mode invisible_mode
"""

import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# HackAtari is a local install — try common locations if not on sys.path
try:
    from hackatari.core import HackAtari as _HackAtari  # noqa: F401 — just to validate
except ModuleNotFoundError:
    _candidate = os.path.join(os.path.expanduser("~"), "NUDGE", "HackAtari")
    if os.path.isdir(_candidate):
        sys.path.insert(0, _candidate)
    else:
        raise RuntimeError(
            "HackAtari not found. Install it or set NUDGE/HackAtari on sys.path. "
            f"Looked in: {_candidate}"
        )

from nsfr.env import NSFRBaseEnv
from nsfr.agents.imitation_agent import ImitationAgent
from nsfr.utils import make_deterministic
from core.utils.utils import preprocess_frame, get_gaze_model_path
from scripts.evaluation.evaluate_bc_model import load_bc_model


# ── HackAtari modes ──────────────────────────────────────────────────────────
# Key: mode name used in CLI and output CSV
# Value: list of modification strings passed to HackAtari
AVAILABLE_MODES = {
    "vanilla":              [],
    "speed_mode":           ["speed_mode"],
    "vary_car_speeds":      ["vary_car_speeds"],
    "align_all_cars":       ["align_all_cars"],
    "stop_all_cars_tunnel": ["stop_all_cars_tunnel"],
    "invisible_mode":       ["invisible_mode"],
    "phantom_mode":         ["phantom_mode"],
    "stop_nearest_5":       ["stop_nearest_5"],
}

DEFAULT_MODES = [
    "vanilla",
    "speed_mode",
    "vary_car_speeds",
    "align_all_cars",
    "stop_all_cars_tunnel",
    "invisible_mode",
    "phantom_mode",
]

# ── Model registry ────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NSFR_MODELS = [
    {
        "name": "NSFR",
        "model_path": os.path.join(
            _ROOT,
            "trained_models/freeway/nsfr"
            "/new_rules_0.01_lr_nll_td_0.99"
            "/full_ep/2026_06_10_17_48_45/epoch_1.pth"
        ),
        "rules": "new",
        "use_gaze": False,
        "unnormalized": False,
    },
    {
        "name": "GRAIL",
        "model_path": os.path.join(
            _ROOT,
            "trained_models/freeway/grail"
            "/new_rules_0.01_lr_nll_unnormalized_vis_only_td_0.99"
            "/full_ep/2026_06_11_00_03_42/epoch_1.pth"
        ),
        "rules": "new",
        "use_gaze": True,
        "unnormalized": True,
    },
]

BC_MODELS = [
    {
        "name": "BC",
        "run_dir": os.path.join(_ROOT, "trained_models/freeway/agil/all_ep"),
        "gaze_method": "None",
    },
    {
        "name": "AGIL",
        "run_dir": os.path.join(_ROOT, "trained_models/freeway/agil/all_ep"),
        "gaze_method": "AGIL",
    },
]


# ── HackAtari Freeway environment ─────────────────────────────────────────────

def make_hackatari_freeway_env(modifs: list, render_mode: str = "rgb_array"):
    """
    Build a Freeway NSFREnv backed by HackAtari instead of vanilla OCAtari.
    The returned env has the same interface as the standard NSFREnv (freeway):
      - .reset(seed)  → (logic_state, neural_state)
      - .step(action) → (logic_state, neural_state), reward, done
      - .get_rgb_frame() → np.ndarray (210, 160, 3)
    Object detection is always RAM-based (mode="ram"), so visual mods like
    invisible_mode or phantom_mode do not break symbolic state extraction.
    """
    from hackatari.core import HackAtari
    from contextlib import redirect_stdout, redirect_stderr
    from nsfr.utils.common import load_module

    env_path = os.path.join(_ROOT, "core/envs/freeway/env.py")
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            env_module = load_module(env_path)

    # Instantiate freeway NSFREnv then swap its inner .env for HackAtari
    nsfr_env = env_module.NSFREnv.__new__(env_module.NSFREnv)
    NSFRBaseEnv.__init__(nsfr_env, mode="logic")
    nsfr_env.name = "freeway"
    nsfr_env.pred2action = {"noop": 0, "up": 1, "down": 2}

    nsfr_env.env = HackAtari(
        env_name="ALE/Freeway-v5",
        mode="ram",
        obs_mode="ori",
        modifs=modifs,
        render_mode=render_mode,
        render_oc_overlay=False,
    )

    # Bind methods from NSFREnv onto the instance
    nsfr_env.reset = lambda seed=None, options=None: _freeway_reset(nsfr_env, seed, options)
    nsfr_env.step = lambda action, is_mapped=False: _freeway_step(nsfr_env, action, is_mapped)
    nsfr_env.get_rgb_frame = lambda: nsfr_env.env.render()
    nsfr_env.extract_logic_state = env_module.NSFREnv.extract_logic_state.__get__(nsfr_env)
    nsfr_env.extract_neural_state = env_module.NSFREnv.extract_neural_state.__get__(nsfr_env)
    nsfr_env.close = lambda: nsfr_env.env.close()
    nsfr_env.map_action = NSFRBaseEnv.map_action.__get__(nsfr_env)
    nsfr_env.convert_state = NSFRBaseEnv.convert_state.__get__(nsfr_env)

    return nsfr_env


def _freeway_reset(env, seed=None, options=None):
    env.env.reset(seed=seed, options=options)
    state = env.env.objects
    return env.convert_state(state)


def _freeway_step(env, action, is_mapped=False):
    if not is_mapped:
        action = env.map_action(action)
    _, reward, terminated, truncated, _ = env.env.step(action)
    done = terminated or truncated
    state = env.env.objects
    return env.convert_state(state), reward, done


# ── GABRIL-compatible wrapper ─────────────────────────────────────────────────

class _GABRILWrap:
    """Minimal sticky-action + noop wrapper applied on top of the HackAtari env."""

    def __init__(self, env, noop_max=30, seed=42):
        self._env = env
        self.noop_max = noop_max
        self._rng = np.random.default_rng(seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, seed=None, options=None):
        state = self._env.reset(seed=seed, options=options)
        self._rng = np.random.default_rng(seed if seed is not None else 42)
        n_noops = int(self._rng.integers(0, self.noop_max + 1))
        for _ in range(n_noops):
            state, _, done = self._env.step("noop")
            if done:
                state = self._env.reset(seed=seed, options=options)
        return state

    def step(self, action, is_mapped=False):
        return self._env.step(action, is_mapped=is_mapped)

    def get_rgb_frame(self):
        return self._env.get_rgb_frame()

    def close(self):
        self._env.close()


# ── NSFR / GRAIL sequential evaluator ────────────────────────────────────────

def _eval_nsfr(agent, env, num_episodes, seed, max_steps, gaze_model=None, device="cpu"):
    """Run the NSFR/GRAIL agent sequentially. Returns list of episode rewards."""
    agent.model.eval()
    rewards = []

    for ep in range(num_episodes):
        ep_seed = seed + 1000 * ep
        wrapped = _GABRILWrap(env, noop_max=30, seed=ep_seed)
        state = wrapped.reset(seed=ep_seed)
        done = False
        total_r = 0.0
        steps = 0

        frame_buf = None
        if gaze_model is not None:
            frame_buf = deque(maxlen=4)
            gray = preprocess_frame(wrapped.get_rgb_frame())
            for _ in range(4):
                frame_buf.append(gray)

        while not done and steps < max_steps:
            logic_state, _ = state
            ls_tensor = torch.as_tensor(
                logic_state, dtype=torch.float32
            ).unsqueeze(0).to(device)

            gaze_tensor = None
            if gaze_model is not None and frame_buf is not None:
                inp = torch.tensor(
                    np.stack(list(frame_buf), axis=0),
                    dtype=torch.float32, device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    gaze_tensor = gaze_model.predict_normalized(inp).squeeze(0)

            with torch.no_grad():
                action = agent.act(ls_tensor, gaze=gaze_tensor)

            state, reward, done = wrapped.step(action)
            total_r += reward
            steps += 1

            if gaze_model is not None and not done:
                frame_buf.append(preprocess_frame(wrapped.get_rgb_frame()))

        rewards.append(total_r)
        print(f"    ep {ep+1}/{num_episodes}: reward={total_r:.1f}")

    return rewards


# ── BC / AGIL sequential evaluator ───────────────────────────────────────────

def _eval_bc(env, run_dir, gaze_method, num_episodes, seed, device, gaze_model_path, max_steps=5000):
    """Run BC/AGIL pixel-based model sequentially. Returns list of episode rewards."""
    dev = torch.device(device)
    encoder, pre_actor, actor, encoder_agil = load_bc_model(
        run_dir, gaze_method, device, ckpt_prefix="best_", stack=4
    )

    gaze_predictor = None
    if gaze_method == "AGIL" and os.path.exists(gaze_model_path):
        try:
            from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
            gaze_predictor = Human_Gaze_Predictor("freeway")
            gaze_predictor.init_model(gaze_model_path)
            gaze_predictor.model.eval()
        except Exception as e:
            print(f"    Warning: could not load gaze predictor: {e}")

    rewards = []

    for ep in range(num_episodes):
        ep_seed = seed + 1000 * ep
        wrapped = _GABRILWrap(env, noop_max=30, seed=ep_seed)
        state = wrapped.reset(seed=ep_seed)
        done = False
        total_r = 0.0
        steps = 0

        frame_buf = deque(maxlen=4)
        gray = preprocess_frame(wrapped.get_rgb_frame())
        for _ in range(4):
            frame_buf.append(gray)

        while not done and steps < max_steps:
            img_stack = np.stack(list(frame_buf), axis=-1)  # (84, 84, 4)
            xx = torch.tensor(
                img_stack, dtype=torch.float32, device=dev
            ).permute(2, 0, 1).unsqueeze(0)  # (1, 4, 84, 84)

            gg = torch.zeros(1, 1, 84, 84, device=dev)
            if gaze_predictor is not None:
                inp4 = torch.tensor(img_stack, dtype=torch.float32, device=dev
                                    ).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    gg = gaze_predictor.predict_normalized(inp4).to(dev)

            with torch.no_grad():
                xx_in = xx * gg if gaze_method == "Mask" else xx
                z = encoder(xx_in)
                if gaze_method == "AGIL" and encoder_agil is not None:
                    z = (z + encoder_agil(xx * gg)) / 2
                logits = actor(pre_actor(z))
                action_idx = logits.argmax(dim=1).item()

            step_result = wrapped.step(action_idx, is_mapped=True)
            state, reward, done = step_result[0], step_result[1], step_result[2]
            total_r += reward
            steps += 1

            if not done:
                frame_buf.append(preprocess_frame(wrapped.get_rgb_frame()))

        rewards.append(total_r)
        print(f"    ep {ep+1}/{num_episodes}: reward={total_r:.1f}")

    return rewards


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Freeway models across HackAtari modification modes."
    )
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                        choices=list(AVAILABLE_MODES.keys()),
                        help="HackAtari modes to evaluate (default: all meaningful modes)")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Episodes per model per mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Max steps per episode")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str,
                        default=os.path.join(_ROOT, "results/hackatari_freeway_eval.csv"),
                        help="Output CSV path")
    parser.add_argument("--nsfr_only", action="store_true",
                        help="Skip BC/AGIL models (faster)")
    parser.add_argument("--bc_only", action="store_true",
                        help="Skip NSFR/GRAIL models")
    args = parser.parse_args()

    make_deterministic(args.seed)
    device = torch.device(args.device)
    gaze_model_path = get_gaze_model_path("freeway")

    # Validate model paths
    missing = []
    if not args.bc_only:
        for cfg in NSFR_MODELS:
            if not os.path.exists(cfg["model_path"]):
                missing.append(f"  NSFR model '{cfg['name']}': {cfg['model_path']}")
    if not args.nsfr_only:
        for cfg in BC_MODELS:
            if not os.path.exists(os.path.join(cfg["run_dir"], "best_actor.pth")):
                missing.append(f"  BC model '{cfg['name']}': {cfg['run_dir']}/best_actor.pth")
    if missing:
        print("WARNING: Some model paths not found — those models will be skipped:")
        for m in missing:
            print(m)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = []
    modes_to_run = args.modes

    print(f"\n{'='*60}")
    print(f"HackAtari Freeway Evaluation")
    print(f"  Modes:    {modes_to_run}")
    print(f"  Episodes: {args.episodes} per model per mode")
    print(f"  Device:   {args.device}")
    print(f"{'='*60}\n")

    for mode_name in modes_to_run:
        modifs = AVAILABLE_MODES[mode_name]
        print(f"\n{'─'*50}")
        print(f"Mode: {mode_name}  (modifs={modifs})")
        print(f"{'─'*50}")

        # ── NSFR / GRAIL ──────────────────────────────────────────────────────
        if not args.bc_only:
            for cfg in NSFR_MODELS:
                if not os.path.exists(cfg["model_path"]):
                    print(f"  [{cfg['name']}] skipped — model not found")
                    continue

                print(f"\n  [{cfg['name']}]  loading {os.path.basename(cfg['model_path'])}")
                agent = ImitationAgent(
                    "freeway",
                    cfg["rules"],
                    device,
                    gaze_threshold=50.0 if cfg["use_gaze"] else None,
                    unnormalized=cfg.get("unnormalized", False),
                )
                agent.load(cfg["model_path"])
                agent.model.eval()

                gaze_model = None
                if cfg["use_gaze"] and os.path.exists(gaze_model_path):
                    try:
                        from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
                        gaze_model = Human_Gaze_Predictor("freeway")
                        gaze_model.init_model(gaze_model_path)
                        gaze_model.model = gaze_model.model.to(device)
                        gaze_model.model.eval()
                    except Exception as e:
                        print(f"    Warning: could not load gaze predictor: {e}")

                env = make_hackatari_freeway_env(modifs)
                try:
                    rewards = _eval_nsfr(
                        agent, env,
                        num_episodes=args.episodes,
                        seed=args.seed,
                        max_steps=args.max_steps,
                        gaze_model=gaze_model,
                        device=device,
                    )
                finally:
                    env.close()

                if gaze_model is not None:
                    del gaze_model
                    torch.cuda.empty_cache()

                mean_r, std_r = np.mean(rewards), np.std(rewards)
                print(f"    → mean={mean_r:.2f}  std={std_r:.2f}  "
                      f"min={min(rewards):.0f}  max={max(rewards):.0f}")
                results.append({
                    "model":       cfg["name"],
                    "model_type":  "neurosymbolic",
                    "mode":        mode_name,
                    "mean_reward": mean_r,
                    "std_reward":  std_r,
                    "min_reward":  min(rewards),
                    "max_reward":  max(rewards),
                    "n_episodes":  len(rewards),
                })

        # ── BC / AGIL ─────────────────────────────────────────────────────────
        if not args.nsfr_only:
            for cfg in BC_MODELS:
                actor_path = os.path.join(cfg["run_dir"], "best_actor.pth")
                if not os.path.exists(actor_path):
                    print(f"  [{cfg['name']}] skipped — model not found")
                    continue

                print(f"\n  [{cfg['name']}]  gaze_method={cfg['gaze_method']}")
                env = make_hackatari_freeway_env(modifs)
                try:
                    rewards = _eval_bc(
                        env,
                        run_dir=cfg["run_dir"],
                        gaze_method=cfg["gaze_method"],
                        num_episodes=args.episodes,
                        seed=args.seed,
                        device=args.device,
                        gaze_model_path=gaze_model_path,
                        max_steps=args.max_steps,
                    )
                finally:
                    env.close()

                mean_r, std_r = np.mean(rewards), np.std(rewards)
                print(f"    → mean={mean_r:.2f}  std={std_r:.2f}  "
                      f"min={min(rewards):.0f}  max={max(rewards):.0f}")
                results.append({
                    "model":       cfg["name"],
                    "model_type":  "neural",
                    "mode":        mode_name,
                    "mean_reward": mean_r,
                    "std_reward":  std_r,
                    "min_reward":  min(rewards),
                    "max_reward":  max(rewards),
                    "n_episodes":  len(rewards),
                })

    # ── Save and display results ───────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Pivot table: models as rows, modes as columns
    if not df.empty:
        pivot = df.pivot_table(
            index="model", columns="mode", values="mean_reward", aggfunc="first"
        )
        # Sort columns to match DEFAULT_MODES order
        ordered_cols = [m for m in DEFAULT_MODES if m in pivot.columns]
        pivot = pivot[ordered_cols]
        print(f"\n{'='*60}")
        print("MEAN REWARD SUMMARY  (rows=models, cols=modes)")
        print(f"{'='*60}")
        print(pivot.to_string(float_format="{:.1f}".format))
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
