import os
import argparse
import torch
import numpy as np
import multiprocessing as mp
from collections import deque
from pathlib import Path
import glob
import time
import pandas as pd

from nsfr.agents.imitation_agent import ImitationAgent
from nsfr.env import NSFRBaseEnv
from nsfr.utils import make_deterministic


# ── GABRILEnvWrapper — inlined so worker processes don't need a separate import
class GABRILEnvWrapper:
    """
    Wraps NSFRBaseEnv to match GABRIL's Atari evaluation settings:
      - frame_skip=4             repeat action 4 ALE frames, sum rewards
      - Sticky actions           (action_repeat_probability=0.25)
      - noop_max=30              random no-ops on reset
      - terminal_on_life_loss    treat life loss as episode end
      - Seed formula             handled by caller: seed + 1000 * episode_idx

    All NSFRBaseEnv attributes/methods are proxied through transparently.
    """

    def __init__(self, env, action_repeat_probability=0.25, noop_max=30,
                 terminal_on_life_loss=True, noop_action='noop', seed=42,
                 frame_skip=1):
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
            return self._env.env.unwrapped.ale.lives()
        except Exception:
            return None

    def reset(self, seed=None, options=None):
        state             = self._env.reset(seed=seed, options=options)
        self._last_action = self.noop_action
        self._rng         = np.random.default_rng(seed if seed is not None else 42)

        if self.noop_max > 0:
            n_noops = int(self._rng.integers(0, self.noop_max + 1))
            for _ in range(n_noops):
                # noops are single ALE frames — no frame_skip during reset
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
        # Sticky action
        if (self.action_repeat_probability > 0.0
                and self._rng.random() < self.action_repeat_probability):
            action = self._last_action
        else:
            self._last_action = action

        # Frame skip: repeat action frame_skip times, accumulate reward
        total_reward = 0.0
        state        = None
        done         = False
        for _ in range(self.frame_skip):
            state, reward, done = self._env.step(action)
            total_reward += reward
            if done:
                break

        # Life-loss termination check (once, after the skip block)
        if self.terminal_on_life_loss and self._ale_available:
            lives = self._get_lives()
            if lives is not None and 0 < lives < self._lives:
                done = True
            if lives is not None:
                self._lives = lives

        return state, total_reward, done

    @staticmethod
    def gabril_seed(base_seed, episode_idx):
        return base_seed + 1000 * episode_idx

    def get_rgb_frame(self):
        return self._env.get_rgb_frame()

    def extract_logic_state(self, raw_state):
        return self._env.extract_logic_state(raw_state)

    def extract_neural_state(self, raw_state):
        return self._env.extract_neural_state(raw_state)

    def close(self):
        self._env.close()

try:
    from scripts.gaze_predictor import Human_Gaze_Predictor
except ImportError:
    Human_Gaze_Predictor = None

try:
    from scripts.email_me import send_email
except ImportError:
    send_email = None

from core.utils.utils import preprocess_frame


# ── Worker message types ──────────────────────────────────────────────────────
_MSG_STATE = 0
_MSG_DONE  = 1
_MSG_STOP  = 2


def _env_worker(worker_id, episode_seeds, state_q, action_q, env_name,
                max_steps, send_frames=False, gabril_compat=False):
    """
    Runs in a separate process. Owns one NSFRBaseEnv instance.

    send_frames=True  → sends (worker_id, MSG_STATE, logic_np, frame_stack_np)
    send_frames=False → sends (worker_id, MSG_STATE, logic_np)

    gabril_compat=True → wraps env with GABRILEnvWrapper (sticky actions,
                         noop_max=30, terminal_on_life_loss=True).
                         Episode seeds must already be GABRIL-formula seeds
                         (base + 1000*ep), computed before spawning.

    Gaze CNN always runs in the main process on GPU.
    Workers are CPU-only and never load the gaze model.
    """
    import sys as _sys
    import os as _os
    _sys.stdout = open(_os.devnull, 'w')

    try:
        base_env = NSFRBaseEnv.from_name(env_name, mode='logic')
        if gabril_compat:
            env = GABRILEnvWrapper(
                base_env,
                action_repeat_probability=0.25,
                noop_max=30,
                terminal_on_life_loss=True,
                noop_action='noop',
                frame_skip=1,
            )
        else:
            env = base_env
    except Exception as e:
        import sys
        print(f"  Worker {worker_id}: env init failed: {e}", file=sys.__stderr__)
        state_q.put((worker_id, _MSG_STOP, None))
        return

    for seed in episode_seeds:
        state   = env.reset(seed=seed)
        done    = False
        total_r = 0.0
        steps   = 0

        frame_buffer = None
        if send_frames:
            from core.utils.utils import preprocess_frame as _pf
            frame_buffer = deque(maxlen=4)
            initial_gray = _pf(env.get_rgb_frame())
            for _ in range(4):
                frame_buffer.append(initial_gray)

        while not done and steps < max_steps:
            logic_state, _ = state
            logic_np = (logic_state.cpu().numpy()
                        if hasattr(logic_state, 'cpu')
                        else np.asarray(logic_state, dtype=np.float32))

            if send_frames and frame_buffer is not None:
                frame_np = np.stack(frame_buffer, axis=0).astype(np.float32)
                state_q.put((worker_id, _MSG_STATE, logic_np, frame_np))
            else:
                state_q.put((worker_id, _MSG_STATE, logic_np))

            msg = action_q.get()
            if msg[0] == _MSG_STOP:
                return

            state, reward, done = env.step(msg[1])
            total_r += reward
            steps   += 1

            if send_frames and frame_buffer is not None and not done:
                from core.utils.utils import preprocess_frame as _pf
                frame_buffer.append(_pf(env.get_rgb_frame()))

        state_q.put((worker_id, _MSG_DONE, total_r))

    state_q.put((worker_id, _MSG_STOP, None))


def _env_worker_safe(*args, **kwargs):
    try:
        _env_worker(*args, **kwargs)
    except Exception as e:
        import sys, traceback
        worker_id = args[0]
        state_q   = args[2]
        print(f"  Worker {worker_id} crashed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        state_q.put((worker_id, _MSG_STOP, None))


def _spawn_workers(num_workers, worker_seeds, state_q, action_qs,
                   env_name, max_steps, send_frames, train_run,
                   gabril_compat=False):
    """Spawn worker pool. Returns list of Process objects."""
    if train_run:
        ctx = mp.get_context('spawn')
    else:
        try:
            ctx = mp.get_context('fork')
        except Exception:
            ctx = mp.get_context('spawn')

    workers = []
    for wid in range(num_workers):
        p = ctx.Process(
            target=_env_worker_safe,
            args=(wid, worker_seeds[wid], state_q, action_qs[wid],
                  env_name, max_steps, send_frames, gabril_compat),
            daemon=True,
        )
        p.start()
        workers.append(p)
    return workers


def _load_gaze_model(env_name, gaze_model_path, device):
    """Load gaze predictor onto device. Returns None on failure."""
    try:
        from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
        gm = Human_Gaze_Predictor(env_name)
        gm.init_model(gaze_model_path)
        gm.model = gm.model.to(device)
        gm.model.eval()
        return gm
    except Exception as e:
        print(f"  Warning: could not load gaze predictor: {e}")
        return None


def _run_inference_loop(agent, state_q, action_qs, num_workers,
                        num_episodes, device, use_gaze, gaze_model,
                        verbose=True):
    """
    Core inference loop shared by evaluate_parallel and evaluate_checkpoints.

    Drain strategy:
      - Blocking get with 2ms timeout when nothing is pending (avoids spin)
      - Non-blocking get once work is in flight (minimises dispatch latency)
    Batches all pending workers into a single GPU forward pass.

    Returns list of episode rewards.
    """
    inv_map         = {v: k for k, v in agent.primitive_action_map.items()}
    episode_rewards = []
    workers_done    = 0
    pending         = {}   # worker_id -> (logic_np, frame_np | None)

    with torch.no_grad():
        while workers_done < num_workers or pending:

            # Drain all available messages
            drained = False
            while True:
                try:
                    timeout = 0.0 if (pending or drained) else 0.002
                    msg     = state_q.get(timeout=timeout)
                    drained = True
                except Exception:
                    break

                wid, msg_type = msg[0], msg[1]

                if msg_type == _MSG_STATE:
                    pending[wid] = (msg[2], msg[3] if use_gaze else None)

                elif msg_type == _MSG_DONE:
                    episode_rewards.append(msg[2])
                    if verbose:
                        print(f"  Episode {len(episode_rewards)}"
                              f"/{num_episodes}: Reward = {msg[2]:.1f}")

                elif msg_type == _MSG_STOP:
                    workers_done += 1

            # Batch GPU inference for all pending workers
            if pending:
                wids = list(pending.keys())
                batch_states = torch.tensor(
                    np.stack([pending[w][0] for w in wids]),
                    dtype=torch.float32, device=device
                )

                batch_gazes = None
                if use_gaze and gaze_model is not None:
                    frames_gpu = torch.tensor(
                        np.stack([pending[w][1] for w in wids]),
                        dtype=torch.float32, device=device
                    )
                    batch_gazes = gaze_model.predict_normalized(
                        frames_gpu
                    ).squeeze(1)  # (B, 84, 84)

                _, action_scores = agent.predict(batch_states, gazes=batch_gazes)

                for wid, scores in zip(wids, action_scores):
                    action_qs[wid].put((_MSG_STATE, inv_map[scores.argmax().item()]))

                pending.clear()

    return episode_rewards


def evaluate_parallel(agent, env_name, num_episodes=50, seed=42,
                      num_workers=None, max_steps=10000,
                      gaze_predictor=None, gaze_model_path=None,
                      use_gaze=False, train_run=False, verbose=True,
                      gabril_compat=False):
    """
    Parallel evaluation using a multiprocessing worker pool.

    Workers own envs (CPU only). Gaze CNN lives in main process (GPU).
    Workers send raw frame stacks; main batches them for GPU inference.

    gabril_compat=True applies GABRIL's env settings to every worker:
      - sticky actions (p=0.25)
      - noop_max=30 random no-ops on reset
      - terminal_on_life_loss=True
      - seed formula: base_seed + 1000 * episode_idx

    For scanning all checkpoints in a run, use evaluate_checkpoints() instead.
    """
    # Legacy fallback
    if gaze_predictor is not None and gaze_model_path is None:
        print("  Warning: pass gaze_model_path instead of gaze_predictor "
              "for parallel eval. Falling back to sequential.")
        env = NSFRBaseEnv.from_name(env_name, mode='logic')
        return evaluate(agent, env, num_episodes=num_episodes,
                        seed=seed, gaze_predictor=gaze_predictor,
                        max_steps=max_steps)

    if num_workers is None:
        num_workers = min(mp.cpu_count(), num_episodes)
    num_workers = min(num_workers, num_episodes)
    device      = agent.device

    # Compute episode seeds — GABRIL formula if gabril_compat
    if gabril_compat:
        episode_seeds = [GABRILEnvWrapper.gabril_seed(seed, ep)
                         for ep in range(num_episodes)]
    else:
        episode_seeds = [seed + ep for ep in range(num_episodes)]

    # Distribute round-robin across workers
    worker_seeds = [[] for _ in range(num_workers)]
    for ep, ep_seed in enumerate(episode_seeds):
        worker_seeds[ep % num_workers].append(ep_seed)

    state_q   = mp.Queue()
    action_qs = [mp.Queue() for _ in range(num_workers)]

    gaze_model = None
    if use_gaze and gaze_model_path:
        gaze_model = _load_gaze_model(env_name, gaze_model_path, device)
        if gaze_model:
            print(f"  Gaze predictor loaded on {device}")

    workers = _spawn_workers(num_workers, worker_seeds, state_q, action_qs,
                             env_name, max_steps, use_gaze, train_run,
                             gabril_compat=gabril_compat)

    agent.model.eval()
    rewards = _run_inference_loop(
        agent, state_q, action_qs, num_workers, num_episodes,
        device, use_gaze, gaze_model, verbose=verbose
    )

    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    if gaze_model is not None:
        del gaze_model
        torch.cuda.empty_cache()

    return rewards


def evaluate_checkpoints(run_dir, env_name, rules, device_str='cuda',
                          num_episodes=100, seed=42, num_workers=None,
                          max_steps=10000, use_gaze=False,
                          gaze_model_path=None, unnormalized=False,
                          visible_preds_only=False, alpha=0.1,
                          output_csv=None, verbose=False,
                          gabril_compat=False):
    """
    Evaluate all saved checkpoints in a run directory.

    Loads the agent and gaze model once, swaps weights per checkpoint.
    Workers are respawned per checkpoint (cheap vs GPU inference).

    Usage:
        python evaluate_model.py --run_dir trained_models/.../run_dir \\
            --env seaquest --rules claude_extensive --episodes 100

    Args:
        run_dir:     Directory containing epoch_N.pth files
        env_name:    Environment name
        rules:       Ruleset name
        device_str:  'cuda' or 'cpu'
        num_episodes: Episodes per checkpoint
        seed:        Base seed (episode i uses seed+i)
        num_workers: Parallel envs (default: auto = min(cpu_count, episodes))
        max_steps:   Max steps per episode
        use_gaze:    Whether to use gaze predictor
        gaze_model_path: Path to gaze predictor weights
        unnormalized: Use unnormalized gaze sums
        visible_preds_only: Only scale visible_ predicates
        alpha:       Laplace smoothing alpha (legacy, currently unused)
        output_csv:  Save results here (default: run_dir/eval_scan.csv)
        verbose:     Print per-episode rewards

    Returns:
        pd.DataFrame with columns [checkpoint, epoch, mean_reward, std_reward,
                                   min_reward, max_reward, n_episodes]
    """
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    ckpt_paths = sorted(
        glob.glob(os.path.join(run_dir, 'epoch_*.pth')),
        key=lambda p: int(Path(p).stem.split('_')[1])
    )
    if not ckpt_paths:
        raise FileNotFoundError(f"No epoch_*.pth files found in {run_dir}")

    print(f"Found {len(ckpt_paths)} checkpoints in {run_dir}")
    print(f"Evaluating {num_episodes} episodes per checkpoint, "
          f"{num_workers or 'auto'} workers\n")

    if num_workers is None:
        num_workers = min(mp.cpu_count(), num_episodes)
    num_workers = min(num_workers, num_episodes)

    # Compute episode seeds — GABRIL formula if gabril_compat
    if gabril_compat:
        episode_seeds = [GABRILEnvWrapper.gabril_seed(seed, ep)
                         for ep in range(num_episodes)]
    else:
        episode_seeds = [seed + ep for ep in range(num_episodes)]

    worker_seeds = [[] for _ in range(num_workers)]
    for ep, ep_seed in enumerate(episode_seeds):
        worker_seeds[ep % num_workers].append(ep_seed)

    # Load agent once — weights swapped per checkpoint
    gaze_threshold = 50.0 if use_gaze else None
    agent = ImitationAgent(
        env_name, rules, device,
        gaze_threshold=gaze_threshold,
        unnormalized=unnormalized,
        visible_preds_only=visible_preds_only,
        alpha=alpha,
    )

    # Load gaze model once
    gaze_model = None
    if use_gaze and gaze_model_path:
        gaze_model = _load_gaze_model(env_name, gaze_model_path, device)

    results = []

    for ckpt_path in ckpt_paths:
        epoch = int(Path(ckpt_path).stem.split('_')[1])
        print(f"Epoch {epoch:3d}: ", end='', flush=True)

        agent.load(ckpt_path)
        agent.model.eval()

        state_q   = mp.Queue()
        action_qs = [mp.Queue() for _ in range(num_workers)]
        workers   = _spawn_workers(
            num_workers, worker_seeds, state_q, action_qs,
            env_name, max_steps, use_gaze, train_run=False,
            gabril_compat=gabril_compat
        )

        t0      = time.perf_counter()
        rewards = _run_inference_loop(
            agent, state_q, action_qs, num_workers, num_episodes,
            device, use_gaze, gaze_model, verbose=verbose
        )
        elapsed = time.perf_counter() - t0

        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        if not rewards:
            print(f"  WARNING: No episodes completed for {ckpt_path} — all workers failed. Skipping.")
            continue

        mean_r = np.mean(rewards)
        std_r  = np.std(rewards)
        min_r  = np.min(rewards)
        max_r  = np.max(rewards)

        print(f"mean={mean_r:7.1f}  std={std_r:7.1f}  "
              f"min={min_r:5.0f}  max={max_r:5.0f}  "
              f"({num_episodes/elapsed:.1f} ep/s)")

        results.append({
            'checkpoint':  ckpt_path,
            'epoch':       epoch,
            'mean_reward': mean_r,
            'std_reward':  std_r,
            'min_reward':  min_r,
            'max_reward':  max_r,
            'n_episodes':  num_episodes,
        })

    if gaze_model is not None:
        del gaze_model
        torch.cuda.empty_cache()

    df = pd.DataFrame(results).sort_values('epoch').reset_index(drop=True)

    if output_csv is None:
        output_csv = os.path.join(run_dir, 'eval_scan.csv')
    df.to_csv(output_csv, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("CHECKPOINT SCAN SUMMARY")
    print("=" * 60)
    print(f"{'Epoch':>6}  {'Mean':>8}  {'Std':>8}  {'Min':>6}  {'Max':>6}")
    print("-" * 60)
    best_mean = df['mean_reward'].max()
    for _, row in df.iterrows():
        marker = " ◄ best" if row['mean_reward'] == best_mean else ""
        print(f"  {int(row['epoch']):4d}   {row['mean_reward']:8.1f}  "
              f"{row['std_reward']:8.1f}  {row['min_reward']:6.0f}  "
              f"{row['max_reward']:6.0f}{marker}")
    print("=" * 60)
    best = df.loc[df['mean_reward'].idxmax()]
    print(f"\nBest: epoch {int(best['epoch'])}  "
          f"mean={best['mean_reward']:.1f} ± {best['std_reward']:.1f}")
    print(f"Results saved to {output_csv}\n")

    return df


def evaluate(agent, env, env_name=None, num_episodes=5, seed=42,
             gaze_predictor=None, log_interval=100, valuation_interval=50,
             max_steps=10000):
    """Sequential evaluation. Kept for compatibility."""
    if env is None:
        env = NSFRBaseEnv.from_name(env_name, mode='logic')

    agent.model.eval()
    episode_rewards = []
    if seed is not None:
        make_deterministic(seed)

    for i in range(num_episodes):
        try:
            state = env.reset(seed=seed + i) if seed is not None else env.reset()
        except TypeError:
            state = env.reset()

        done         = False
        total_reward = 0.0
        step_count   = 0
        frame_buffer = None

        if gaze_predictor is not None:
            frame_buffer = deque(maxlen=4)
            initial_gray = preprocess_frame(env.get_rgb_frame())
            for _ in range(4):
                frame_buffer.append(initial_gray)

        while not done and step_count < max_steps:
            gaze_tensor = None
            if gaze_predictor is not None and frame_buffer is not None:
                input_tensor = torch.tensor(
                    np.stack(list(frame_buffer), axis=0),
                    dtype=torch.float32, device=agent.device
                ).unsqueeze(0)
                with torch.no_grad():
                    gaze_tensor = gaze_predictor.predict_normalized(
                        input_tensor
                    ).squeeze(0)

            logic_state, _ = state
            logic_state_tensor = torch.as_tensor(
                logic_state, dtype=torch.float32
            ).unsqueeze(0).to(agent.device)

            predicate           = agent.act(logic_state_tensor, gaze=gaze_tensor)
            state, reward, done = env.step(predicate)
            total_reward       += reward
            step_count         += 1

            if log_interval > 0 and step_count % log_interval == 0:
                print(f"  Episode {i+1} | Step {step_count} | "
                      f"Reward: {total_reward:.1f}")

            if gaze_predictor is not None and not done:
                frame_buffer.append(preprocess_frame(env.get_rgb_frame()))

        episode_rewards.append(total_reward)
        print(f"Episode {i+1}: Reward = {total_reward}")

    return episode_rewards


def main():
    parser = argparse.ArgumentParser()
    # ── Modes ─────────────────────────────────────────────────────────────────
    parser.add_argument("--model_path",         type=str, default=None,
                        help="Single checkpoint .pth to evaluate")
    parser.add_argument("--run_dir",            type=str, default=None,
                        help="Scan all epoch_*.pth checkpoints in this dir")
    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--env",                type=str, default="seaquest")
    parser.add_argument("--rules",              type=str, default="new")
    parser.add_argument("--device",             type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    # ── Eval ──────────────────────────────────────────────────────────────────
    parser.add_argument("--episodes",           type=int, default=100)
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--num_workers",        type=int, default=None)
    parser.add_argument("--max_steps",          type=int, default=10000)
    parser.add_argument("--sequential",         action="store_true")
    parser.add_argument("--verbose",            action="store_true")
    # ── Gaze ──────────────────────────────────────────────────────────────────
    parser.add_argument("--use_gaze",           action="store_true")
    parser.add_argument("--gaze_model_path",    type=str,
                        default="gaze_models/seaquest/seaquest_gaze_predictor_2.pth")
    parser.add_argument("--unnormalized",       action="store_true")
    parser.add_argument("--visible_preds_only", action="store_true")
    parser.add_argument("--alpha",              type=float, default=None)
    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--gabril_compat",      action="store_true",
                        help="Match GABRIL eval env: sticky actions p=0.25, "
                             "noop_max=30, terminal_on_life_loss, "
                             "seed=base+1000*ep")
    parser.add_argument("--output_csv",         type=str, default=None)
    args = parser.parse_args()

    if args.unnormalized:
        args.use_gaze = True
    if args.use_gaze and args.alpha is None and not args.unnormalized:
        args.alpha = 0.1

    make_deterministic(args.seed)

    # ── Checkpoint scan ───────────────────────────────────────────────────────
    if args.run_dir is not None:
        evaluate_checkpoints(
            run_dir=args.run_dir,
            env_name=args.env,
            rules=args.rules,
            device_str=args.device,
            num_episodes=args.episodes,
            seed=args.seed,
            num_workers=args.num_workers,
            max_steps=args.max_steps,
            use_gaze=args.use_gaze,
            gaze_model_path=args.gaze_model_path if args.use_gaze else None,
            unnormalized=args.unnormalized,
            visible_preds_only=args.visible_preds_only,
            alpha=args.alpha,
            output_csv=args.output_csv,
            verbose=args.verbose,
            gabril_compat=args.gabril_compat,
        )
        return

    # ── Single checkpoint ─────────────────────────────────────────────────────
    if args.model_path is None:
        print("Error: provide --model_path or --run_dir")
        return

    device = torch.device(args.device)
    agent  = ImitationAgent(
        args.env, args.rules, device,
        gaze_threshold=50.0 if args.use_gaze else None,
        unnormalized=args.unnormalized,
        visible_preds_only=args.visible_preds_only,
        alpha=args.alpha,
    )
    print(f"Loading model from {args.model_path}...")
    agent.load(args.model_path)

    if args.sequential:
        gaze_predictor = None
        if args.use_gaze:
            from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
            gaze_predictor = Human_Gaze_Predictor(args.env)
            gaze_predictor.init_model(args.gaze_model_path)
            gaze_predictor.model.eval()
        env     = NSFRBaseEnv.from_name(args.env, mode='logic')
        rewards = evaluate(agent, env, num_episodes=args.episodes,
                           seed=args.seed, gaze_predictor=gaze_predictor,
                           max_steps=args.max_steps)
    else:
        rewards = evaluate_parallel(
            agent,
            env_name=args.env,
            num_episodes=args.episodes,
            seed=args.seed,
            num_workers=args.num_workers,
            max_steps=args.max_steps,
            gaze_model_path=args.gaze_model_path if args.use_gaze else None,
            use_gaze=args.use_gaze,
            verbose=args.verbose,
            gabril_compat=args.gabril_compat,
        )

    mean_r, std_r = np.mean(rewards), np.std(rewards)
    print("\n" + "=" * 40)
    print(f"Model:       {args.model_path}")
    print(f"Episodes:    {args.episodes}")
    print(f"Mean Reward: {mean_r:.2f}")
    print(f"Std Dev:     {std_r:.2f}")
    print(f"Min / Max:   {min(rewards):.0f} / {max(rewards):.0f}")
    print("=" * 40)

    if args.output_csv:
        pd.DataFrame([{
            'checkpoint':  args.model_path,
            'mean_reward': mean_r,
            'std_reward':  std_r,
            'min_reward':  min(rewards),
            'max_reward':  max(rewards),
            'n_episodes':  len(rewards),
        }]).to_csv(args.output_csv, index=False)
        print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()