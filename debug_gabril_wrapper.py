"""
debug_gabril_wrapper.py
=======================
Diagnoses whether GABRILEnvWrapper sits correctly on NSFRBaseEnv.

Checks:
  1. Env wrapper chain (frame skip, reward clipping, life handling)
  2. ALE lives path
  3. Noop steps actually advance the env
  4. Sticky actions firing at ~25%
  5. Life-loss termination triggers correctly
  6. Bare vs wrapped reward comparison over N episodes

Run:
    python debug_gabril_wrapper.py --env seaquest --episodes 3
"""

import argparse
import numpy as np
import torch
from nsfr.env import NSFRBaseEnv

# ── inline the wrapper so this file is self-contained ────────────────────────
class GABRILEnvWrapper:
    def __init__(self, env, action_repeat_probability=0.25, noop_max=30,
                 terminal_on_life_loss=True, noop_action='noop', seed=42,
                 frame_skip=4):
        self._env                       = env
        self.action_repeat_probability  = action_repeat_probability
        self.noop_max                   = noop_max
        self.terminal_on_life_loss      = terminal_on_life_loss
        self.noop_action                = noop_action
        self.frame_skip                 = frame_skip
        self._rng                       = np.random.default_rng(seed)
        self._last_action               = noop_action
        self._lives                     = 0
        self._ale_available             = False

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_lives(self):
        try:
            lives = self._env.env.unwrapped.ale.lives()
            return lives
        except Exception as e:
            return None

    def reset(self, seed=None, options=None):
        state             = self._env.reset(seed=seed, options=options)
        self._last_action = self.noop_action
        self._rng         = np.random.default_rng(seed if seed is not None else 42)
        if self.noop_max > 0:
            n_noops = int(self._rng.integers(0, self.noop_max + 1))
            for _ in range(n_noops):
                state, _, done = self._env.step(self.noop_action)
                if done:
                    state = self._env.reset(seed=seed, options=options)
        lives = self._get_lives()
        if lives is not None:
            self._lives         = lives
            self._ale_available = True
        else:
            self._ale_available = False
        return state

    def step(self, action):
        if (self.action_repeat_probability > 0.0
                and self._rng.random() < self.action_repeat_probability):
            action = self._last_action
        else:
            self._last_action = action

        total_reward = 0.0
        state        = None
        done         = False
        for _ in range(self.frame_skip):
            state, reward, done = self._env.step(action)
            total_reward += reward
            if done:
                break

        if self.terminal_on_life_loss and self._ale_available:
            lives = self._get_lives()
            if lives is not None and 0 < lives < self._lives:
                done = True
            if lives is not None:
                self._lives = lives

        return state, total_reward, done

    def get_rgb_frame(self):
        return self._env.get_rgb_frame()

    def close(self):
        self._env.close()


ACTIONS = ['noop', 'fire', 'up', 'right', 'left', 'down']

SEP  = "=" * 60
SEP2 = "-" * 60


# ── 1. Print wrapper chain ────────────────────────────────────────────────────
def check_wrapper_chain(env_name):
    print(SEP)
    print("CHECK 1: Wrapper chain inside NSFRBaseEnv")
    print(SEP2)
    raw = NSFRBaseEnv.from_name(env_name, mode='logic')
    obj = raw
    depth = 0
    while True:
        print(f"  {'  ' * depth}{type(obj).__name__}")
        depth += 1
        inner = getattr(obj, 'env', None)
        if inner is None or inner is obj:
            break
        obj = inner
    print()

    # Check for MaxAndSkipEnv (frame skip)
    obj = raw
    found_skip = False
    while True:
        if 'MaxAndSkip' in type(obj).__name__ or 'FrameSkip' in type(obj).__name__:
            skip = getattr(obj, '_skip', getattr(obj, 'skip', '?'))
            print(f"  ✓ Frame skip wrapper found: {type(obj).__name__}  skip={skip}")
            found_skip = True
        if 'ClipReward' in type(obj).__name__:
            print(f"  ✓ ClipRewardEnv found — rewards ARE clipped to {{-1,0,1}}")
        if 'EpisodicLife' in type(obj).__name__:
            print(f"  ! EpisodicLifeEnv found inside NSFRBaseEnv — "
                  "terminal_on_life_loss is ALREADY applied internally. "
                  "GABRILEnvWrapper will double-apply it.")
        inner = getattr(obj, 'env', None)
        if inner is None or inner is obj:
            break
        obj = inner

    if not found_skip:
        print("  ! No frame skip wrapper found — NSFRBaseEnv may use frame_skip=1")
        print("    GABRIL uses eval_fs=4. This is a significant mismatch.")
    print()


# ── 2. ALE lives path ─────────────────────────────────────────────────────────
def check_ale_lives(env_name):
    print(SEP)
    print("CHECK 2: ALE lives path")
    print(SEP2)
    raw = NSFRBaseEnv.from_name(env_name, mode='logic')
    raw.reset(seed=42)

    # Try known path
    try:
        lives = raw.env.unwrapped.ale.lives()
        print(f"  ✓ raw.env.unwrapped.ale.lives() = {lives}  (path is correct)")
    except Exception as e:
        print(f"  ✗ raw.env.unwrapped.ale.lives() failed: {e}")

    # Try alternatives
    for path in ['raw.unwrapped.ale.lives()',
                 'raw.env.env.unwrapped.ale.lives()',
                 'raw.ale.lives()']:
        try:
            lives = eval(path)
            print(f"  ✓ {path} = {lives}  (alternative path works)")
        except Exception:
            pass
    print()


# ── 3. Noop steps advance the env ─────────────────────────────────────────────
def check_noops(env_name):
    print(SEP)
    print("CHECK 3: Noop steps advance environment state")
    print(SEP2)
    raw = NSFRBaseEnv.from_name(env_name, mode='logic')
    state0 = raw.reset(seed=42)
    logic0, _ = state0

    state1, _, _ = raw.step('noop')
    logic1, _ = state1

    l0 = logic0.cpu().numpy() if hasattr(logic0, 'cpu') else np.array(logic0)
    l1 = logic1.cpu().numpy() if hasattr(logic1, 'cpu') else np.array(logic1)

    if np.allclose(l0, l1):
        print("  ! Noop step did NOT change logic state — "
              "env may not advance on noop or frame skip is >1")
    else:
        diff = np.abs(l0 - l1).sum()
        print(f"  ✓ Noop step changed logic state (L1 diff = {diff:.4f})")
    print()


# ── 4. Sticky actions fire at ~25% ────────────────────────────────────────────
def check_sticky(env_name, n_steps=200):
    print(SEP)
    print(f"CHECK 4: Sticky action rate (expected ~25%, over {n_steps} steps)")
    print(SEP2)
    raw = NSFRBaseEnv.from_name(env_name, mode='logic')
    wrapped = GABRILEnvWrapper(raw, action_repeat_probability=0.25,
                               noop_max=0, terminal_on_life_loss=False,
                               seed=42)
    wrapped.reset(seed=42)

    intended    = []
    actually_exec = []

    orig_step = raw.step
    executed_actions = []

    def tracking_step(action, **kwargs):
        executed_actions.append(action)
        return orig_step(action, **kwargs)
    raw.step = tracking_step

    rng = np.random.default_rng(99)
    for _ in range(n_steps):
        a = ACTIONS[rng.integers(0, 6)]
        intended.append(a)
        state, r, done = wrapped.step(a)
        if done:
            wrapped.reset(seed=0)

    raw.step = orig_step  # restore

    if not executed_actions:
        print("  Could not track executed actions (step monkey-patch may not work)")
    else:
        repeats = sum(1 for i, e in zip(intended, executed_actions) if i != e)
        rate    = repeats / len(executed_actions)
        print(f"  Intended vs executed mismatch: {repeats}/{len(executed_actions)} = {rate:.1%}")
        if 0.15 < rate < 0.35:
            print(f"  ✓ Sticky rate is plausible (~25%)")
        else:
            print(f"  ! Sticky rate {rate:.1%} is outside expected 15-35% range")
    print()


# ── 5. Life-loss termination ──────────────────────────────────────────────────
def check_life_loss(env_name):
    print(SEP)
    print("CHECK 5: Life-loss termination")
    print(SEP2)
    raw = NSFRBaseEnv.from_name(env_name, mode='logic')
    wrapped = GABRILEnvWrapper(raw, action_repeat_probability=0.0,
                               noop_max=0, terminal_on_life_loss=True,
                               seed=42)
    wrapped.reset(seed=42)

    if not wrapped._ale_available:
        print("  ✗ ALE not accessible — life-loss termination is DISABLED")
        print("    Scores will be inflated vs GABRIL (no per-life resets)")
        print()
        return

    print(f"  Initial lives: {wrapped._lives}")
    print(f"  Running until first life loss (up to 10000 steps)...")

    done = False
    steps = 0
    life_losses = 0
    while not done and steps < 10000:
        _, _, done = wrapped.step('noop')
        steps += 1
        if done and wrapped._lives > 0:
            life_losses += 1
            print(f"  ✓ Life-loss termination triggered at step {steps} "
                  f"(lives remaining: {wrapped._lives})")
            break

    if life_losses == 0:
        print(f"  No life loss in {steps} steps — game ended naturally or "
              "lives didn't decrease")
    print()


# ── 6. Reward comparison: bare vs wrapped ─────────────────────────────────────
def compare_rewards(env_name, n_episodes=3, max_steps=2000):
    print(SEP)
    print(f"CHECK 6: Reward comparison — bare vs wrapped ({n_episodes} episodes, "
          f"random policy, max {max_steps} steps)")
    print(SEP2)
    rng = np.random.default_rng(0)

    for ep in range(n_episodes):
        seed = 42 + 1000 * ep

        # ── bare ──
        raw = NSFRBaseEnv.from_name(env_name, mode='logic')
        raw.reset(seed=seed)
        bare_r, bare_steps = 0.0, 0
        bare_done = False
        while not bare_done and bare_steps < max_steps:
            a = ACTIONS[rng.integers(0, 6)]
            _, r, bare_done = raw.step(a)
            bare_r += r
            bare_steps += 1
        raw.close()

        # ── wrapped ──
        raw2    = NSFRBaseEnv.from_name(env_name, mode='logic')
        wrapped = GABRILEnvWrapper(raw2, action_repeat_probability=0.25,
                                   noop_max=30, terminal_on_life_loss=True,
                                   seed=seed)
        wrapped.reset(seed=seed)
        rng2 = np.random.default_rng(0)  # same random policy
        wrap_r, wrap_steps = 0.0, 0
        wrap_done = False
        while not wrap_done and wrap_steps < max_steps:
            a = ACTIONS[rng2.integers(0, 6)]
            _, r, wrap_done = wrapped.step(a)
            wrap_r += r
            wrap_steps += 1
        wrapped.close()

        print(f"  Episode {ep+1}:  bare={bare_r:6.0f} ({bare_steps} steps)  "
              f"wrapped={wrap_r:6.0f} ({wrap_steps} steps)")

    print()
    print("  If wrapped scores are consistently >> bare: life-loss not firing")
    print("  If wrapped scores are consistently << bare: "
          "sticky actions or noops hurting too much, or reward clipping")
    print()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",      type=str, default="seaquest")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    print(f"\nDEBUGGING GABRILEnvWrapper on NSFRBaseEnv  (env={args.env})\n")

    check_wrapper_chain(args.env)
    check_ale_lives(args.env)
    check_noops(args.env)
    check_sticky(args.env)
    check_life_loss(args.env)
    compare_rewards(args.env, n_episodes=args.episodes)

    print(SEP)
    print("SUMMARY — things to look for:")
    print("  1. EpisodicLifeEnv inside NSFRBaseEnv → double life-loss")
    print("  2. ClipRewardEnv inside NSFRBaseEnv   → scores capped")
    print("  3. No frame skip / wrong frame skip   → biggest score driver")
    print("  4. ALE path fails                     → no life-loss termination")
    print(SEP)


if __name__ == "__main__":
    main()