import os
import argparse
import torch
import numpy as np
import multiprocessing as mp
from collections import deque

from nsfr.agents.imitation_agent import ImitationAgent
from nsfr.env import NSFRBaseEnv
from nsfr.utils import make_deterministic

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
_MSG_STATE = 0   # worker -> main: logic state, wants action back
_MSG_DONE  = 1   # worker -> main: episode finished, sending reward
_MSG_STOP  = 2   # main -> worker / worker -> main: exit signal


def _env_worker(worker_id, episode_seeds, state_q, action_q, env_name,
                max_steps, gaze_model_path=None, gaze_env_name=None):
    """
    Runs in a separate process. Owns one NSFRBaseEnv instance.

    If gaze_model_path is provided, also owns a CPU gaze predictor.
    The gaze predictor runs entirely on CPU inside this worker process —
    no GPU required — so N workers can compute gaze in parallel.

    Each step sends (logic_state, gaze_tensor) to the main process.
    The gaze_tensor is zeros if no gaze predictor is loaded.

    For each assigned seed:
      1. Reset env, initialise 4-frame gaze buffer
      2. Compute gaze heatmap on CPU (if predictor loaded)
      3. Send (logic_state, gaze_tensor) to main process
      4. Wait for action, step env, update frame buffer
      5. Send total reward when episode ends
    """
    from collections import deque
    import numpy as np

    # Suppress stdout only — keeps ALE/gaze print noise silent
    # but preserves stderr so worker crashes are still visible.
    import sys as _sys
    import os as _os
    _sys.stdout = open(_os.devnull, 'w')

    try:
        env = NSFRBaseEnv.from_name(env_name, mode='logic')
    except Exception as e:
        import sys
        print(f"  Worker {worker_id}: env init failed: {e}", file=sys.__stderr__)
        state_q.put((worker_id, _MSG_STOP, None))
        return

    # Workers send raw grayscale frames to main process.
    # Main process batches frames and runs gaze CNN on GPU — much faster
    # than 8 separate CPU CNN calls, and avoids queue congestion from
    # sending large gaze tensors (28KB each) over the pipe.
    _use_gaze   = gaze_model_path is not None
    _FRAME_ZEROS = np.zeros((4, 84, 84), dtype=np.float32)

    for seed in episode_seeds:
        state   = env.reset(seed=seed)
        done    = False
        total_r = 0.0
        steps   = 0

        # 4-frame buffer of grayscale frames for gaze CNN
        frame_buffer = None
        if _use_gaze:
            from core.utils.utils import preprocess_frame
            frame_buffer = deque(maxlen=4)
            initial_gray = preprocess_frame(env.get_rgb_frame())
            for _ in range(4):
                frame_buffer.append(initial_gray)

        while not done and steps < max_steps:
            logic_state, _ = state

            if hasattr(logic_state, 'cpu'):
                logic_state_np = logic_state.cpu().numpy()
            else:
                logic_state_np = np.asarray(logic_state, dtype=np.float32)

            if _use_gaze and frame_buffer is not None:
                # Send (logic_state, frame_stack) — frame_stack is (4,84,84) uint8
                frame_np = np.stack(frame_buffer, axis=0).astype(np.float32)  # (B,4,84,84)
                state_q.put((worker_id, _MSG_STATE, logic_state_np, frame_np))
            else:
                # Omit frame array for no-gaze runs to avoid huge IPC bottleneck!
                state_q.put((worker_id, _MSG_STATE, logic_state_np))
                
            msg = action_q.get()
            if msg[0] == _MSG_STOP:
                return
            predicate = msg[1]
            state, reward, done = env.step(predicate)
            total_r += reward
            steps   += 1

            if _use_gaze and frame_buffer is not None and not done:
                from core.utils.utils import preprocess_frame
                frame_buffer.append(preprocess_frame(env.get_rgb_frame()))

        state_q.put((worker_id, _MSG_DONE, total_r))

    state_q.put((worker_id, _MSG_STOP, None))


def _env_worker_safe(*args, **kwargs):
    """Wrapper that catches unhandled exceptions and reports via queue."""
    # args[-3] is state_q, args[-4] is worker_id by position
    try:
        _env_worker(*args, **kwargs)
    except Exception as e:
        import sys, traceback
        worker_id = args[0]
        state_q   = args[2]
        print(f"  Worker {worker_id} crashed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        state_q.put((worker_id, _MSG_STOP, None))


def evaluate_parallel(agent, env_name, num_episodes=50, seed=42,
                      num_workers=None, max_steps=10000,
                      gaze_predictor=None, gaze_model_path=None,
                      use_gaze=False, _gaze_pred_obj=None,train_run=False):
    """
    Parallel evaluation using multiprocessing worker pool.

    Architecture:
      - N worker processes each own one NSFRBaseEnv (CPU only)
      - If gaze is enabled, each worker also owns a CPU gaze predictor
        and computes gaze heatmaps locally — no GPU needed in workers
      - Workers send (logic_state, gaze_tensor) to main process
      - Main process batches all pending states into ONE GPU forward pass
      - Actions dispatched back to the correct workers

    Args:
        agent:           ImitationAgent with loaded model weights
        env_name:        Environment name string e.g. 'seaquest'
        num_episodes:    Total episodes to evaluate
        seed:            Base seed — episode i uses seed+i
        num_workers:     Parallel envs. Default: min(cpu_count, num_episodes)
        max_steps:       Max steps per episode before forced termination
        gaze_predictor:  Legacy — if provided, falls back to sequential eval
        gaze_model_path: Path to gaze predictor .pth weights for parallel gaze
        use_gaze:        Whether to pass gaze tensors to agent.predict()

    Returns:
        List of total rewards, length == num_episodes
    """
    # Legacy fallback: if caller passes a gaze_predictor object (GPU-bound),
    # we cannot use it in worker processes. Fall back to sequential.
    if gaze_predictor is not None and gaze_model_path is None:
        print("  Warning: GPU gaze_predictor not supported in parallel eval.")
        print("  Pass gaze_model_path instead for parallel gaze evaluation.")
        print("  Falling back to sequential evaluation.")
        env = NSFRBaseEnv.from_name(env_name, mode='logic')
        return evaluate(agent, env, num_episodes=num_episodes,
                        seed=seed, gaze_predictor=gaze_predictor,
                        max_steps=max_steps)

    if num_workers is None:
        cpu_count   = mp.cpu_count()
        num_workers = min(cpu_count, num_episodes)
        if gaze_model_path:
            print(f"  Auto-selected {num_workers} workers "
                  f"({cpu_count} CPUs, {num_episodes} episodes, "
                  f"CPU gaze enabled)")
        else:
            print(f"  Auto-selected {num_workers} workers "
                  f"({cpu_count} CPUs, {num_episodes} episodes)")

    num_workers = min(num_workers, num_episodes)
    device      = agent.device

    # Distribute episodes round-robin across workers
    worker_seeds = [[] for _ in range(num_workers)]
    for ep in range(num_episodes):
        worker_seeds[ep % num_workers].append(seed + ep)

    # One shared inbound queue (workers -> main)
    # One outbound queue per worker (main -> worker)
    state_q   = mp.Queue()
    action_qs = [mp.Queue() for _ in range(num_workers)]

    # Load gaze predictor in main process on GPU — one shared model
    # Workers send raw frame stacks; main batches them for GPU inference
    _gaze_model = None
    if use_gaze and gaze_model_path is not None:
        try:
            from scripts.gaze.gaze_predictor import Human_Gaze_Predictor
            _gaze_model = Human_Gaze_Predictor(env_name)
            _gaze_model.init_model(gaze_model_path)
            _gaze_model.model = _gaze_model.model.to(device)
            _gaze_model.model.eval()
            print(f"  Gaze predictor loaded on {device} for batched inference")
        except Exception as e:
            print(f"  Warning: could not load gaze predictor: {e}")
            _gaze_model = None

    # Workers don't need gaze_model_path anymore — pass None
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
                  env_name, max_steps,
                  gaze_model_path if use_gaze else None,  # signals worker to send frames
                  env_name),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # ── Main inference loop ───────────────────────────────────────────────────
    agent.model.eval()
    episode_rewards = []
    workers_done    = 0
    # pending: worker_id -> (logic_state_np, gaze_np)
    pending         = {}
    inv_map         = {v: k for k, v in agent.primitive_action_map.items()}

    with torch.no_grad():
        while workers_done < num_workers or pending:
            # Drain queue — collect all available states before batching
            while True:
                try:
                    msg = state_q.get(timeout=0.005 if not pending else 0.0)
                except Exception:
                    break

                wid, msg_type = msg[0], msg[1]

                if msg_type == _MSG_STATE:
                    if use_gaze:
                        # msg = (wid, MSG_STATE, logic_state_np, gaze_np)
                        pending[wid] = (msg[2], msg[3])
                    else:
                        # msg = (wid, MSG_STATE, logic_state_np)
                        pending[wid] = (msg[2], None)

                elif msg_type == _MSG_DONE:
                    episode_rewards.append(msg[2])
                    n = len(episode_rewards)
                    print(f"  Episode {n}/{num_episodes}: Reward = {msg[2]:.1f}")

                elif msg_type == _MSG_STOP:
                    workers_done += 1

            # Run batched GPU inference for all workers currently waiting
            if pending:
                wids         = list(pending.keys())
                logic_states = np.stack([pending[w][0] for w in wids])
                frame_stacks = np.stack([pending[w][1] for w in wids])  # (B,4,84,84)

                batch_states = torch.tensor(
                    logic_states, dtype=torch.float32, device=device
                )

                # Run gaze CNN on GPU for the whole batch at once
                batch_gazes = None
                if use_gaze and _gaze_model is not None:
                    with torch.no_grad():
                        frames_gpu = torch.tensor(
                            frame_stacks, dtype=torch.float32, device=device
                        )  # (B,4,84,84)
                        batch_gazes = _gaze_model.predict_normalized(
                            frames_gpu
                        ).squeeze(1)  # (B,84,84)

                _, action_scores = agent.predict(
                    batch_states,
                    gazes=batch_gazes,
                )

                for wid, scores in zip(wids, action_scores):
                    predicate = inv_map[scores.argmax().item()]
                    action_qs[wid].put((_MSG_STATE, predicate))

                pending.clear()

    # Clean up worker processes
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    # Free gaze model from GPU memory before returning to training
    if _gaze_model is not None:
        del _gaze_model
        torch.cuda.empty_cache()

    return episode_rewards


def evaluate(agent, env, env_name=None, num_episodes=5, seed=42,
             gaze_predictor=None, log_interval=100, valuation_interval=50,
             max_steps=10000):
    """
    Original sequential evaluation. Kept for compatibility and gaze support.
    For fast no-gaze evaluation, use evaluate_parallel() instead.
    """
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
            print("Warning: env.reset() does not accept seed.")
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
            if gaze_predictor is not None:
                img_stack    = np.stack(frame_buffer, axis=-1)
                input_tensor = torch.tensor(
                    img_stack, dtype=torch.float32,
                    device=gaze_predictor.device
                ).permute(2, 0, 1).unsqueeze(0)
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
    parser.add_argument("--model_path",         type=str, required=True)
    parser.add_argument("--env",                type=str, default="seaquest")
    parser.add_argument("--rules",              type=str, default="new")
    parser.add_argument("--episodes",           type=int, default=50)
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--device",             type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_interval",       type=int, default=0)
    parser.add_argument("--valuation_interval", type=int, default=0)
    parser.add_argument("--use_gaze",           action="store_true")
    parser.add_argument("--gaze_threshold",     type=float, default=20.0)

    parser.add_argument("--gaze_model_path",    type=str,
                        default="seaquest_gaze_predictor_2.pth")
    parser.add_argument("--send_email",         action="store_true")
    parser.add_argument("--num_workers",        type=int, default=None,
                        help="Parallel envs (default: auto from CPU count)")
    parser.add_argument("--sequential",         action="store_true",
                        help="Force sequential eval (required with --use_gaze)")
    parser.add_argument("--max_steps",          type=int, default=10000)
    parser.add_argument("--unnormalized",       action="store_true", help="Use unnormalized gaze probabilities and multiply valuations")
    parser.add_argument("--visible_preds_only", action="store_true", help="Only scale visible_{object} predicates with gaze")
    parser.add_argument("--alpha",              type=float, default=None, help="Laplacian smoothing parameter for gaze normalization")
    args = parser.parse_args()
    
    # Auto-enable use_gaze if unnormalized is set
    if args.unnormalized:
        args.use_gaze = True

    if args.use_gaze and args.alpha is None and not args.unnormalized:
        args.alpha = 0.1
    device = torch.device(args.device)
    print(f"Using device: {device}")

    make_deterministic(args.seed)

    gaze_threshold = args.gaze_threshold if args.use_gaze else None
    agent = ImitationAgent(args.env, args.rules, device,
                           gaze_threshold=gaze_threshold,
                           unnormalized=args.unnormalized,
                           visible_preds_only=args.visible_preds_only,
                           alpha=args.alpha)

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    print(f"Loading model from {args.model_path}...")
    agent.load(args.model_path)
    print(f"Starting evaluation: {args.episodes} episodes...")

    # Parallel gaze evaluation: workers load gaze model on CPU themselves.
    # Sequential fallback only if --sequential is explicitly requested.
    if not args.sequential:
        gaze_path = args.gaze_model_path if args.use_gaze else None
        rewards = evaluate_parallel(
            agent,
            env_name=args.env,
            num_episodes=args.episodes,
            seed=args.seed,
            num_workers=args.num_workers,
            max_steps=args.max_steps,
            gaze_model_path=gaze_path,
            use_gaze=args.use_gaze,
            
        )
    else:
        # Sequential path — loads gaze predictor on GPU in main process
        gaze_predictor = None
        if args.use_gaze:
            if Human_Gaze_Predictor is None:
                print("Error: Could not import Human_Gaze_Predictor.")
                return
            print(f"Initializing Gaze Predictor from {args.gaze_model_path}...")
            gaze_predictor = Human_Gaze_Predictor(args.env)
            gaze_predictor.init_model(args.gaze_model_path)
            gaze_predictor.model.eval()
        env = NSFRBaseEnv.from_name(args.env, mode='logic')
        rewards = evaluate(
            agent, env,
            num_episodes=args.episodes,
            seed=args.seed,
            gaze_predictor=gaze_predictor,
            log_interval=args.log_interval,
            valuation_interval=args.valuation_interval,
            max_steps=args.max_steps,
        )

    mean_reward = np.mean(rewards)
    std_reward  = np.std(rewards)

    print("\n" + "=" * 40)
    print(f"Model:       {args.model_path}")
    print(f"Episodes:    {args.episodes}")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Dev:     {std_reward:.2f}")
    print(f"Min / Max:   {min(rewards):.0f} / {max(rewards):.0f}")
    print("=" * 40)

    if args.send_email and send_email is not None:
        try:
            send_email(
                f"Eval: {args.env} | N={args.episodes}",
                f"Mean: {mean_reward:.2f}  Std: {std_reward:.2f}\n"
                f"Min: {min(rewards):.0f}  Max: {max(rewards):.0f}\n"
                f"Seed: {args.seed}  Rules: {args.rules}",
            )
            print("Email sent.")
        except Exception as e:
            print(f"Email failed: {e}")


if __name__ == "__main__":
    main()