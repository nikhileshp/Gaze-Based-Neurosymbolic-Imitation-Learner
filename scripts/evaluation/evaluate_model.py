import os
import argparse
import torch
import numpy as np
import multiprocessing as mp
from collections import deque

from nsfr.agents.imitation_agent import ImitationAgent
from nsfr.env import NSFRBaseEnv
from nsfr.utils import make_deterministic
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['ALE_PY_QUIET'] = '1'
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


def _env_worker(worker_id, episode_seeds, state_q, action_q, env_name, max_steps):
    """
    Runs in a separate process. Owns one NSFRBaseEnv instance.

    For each assigned seed:
      1. Reset env
      2. Send logic state to main process via state_q
      3. Wait for action on action_q
      4. Step env, repeat until done or max_steps
      5. Send total reward via state_q with _MSG_DONE

    All env instances run on CPU — only the main process uses the GPU.
    """
    env = NSFRBaseEnv.from_name(env_name, mode='logic')

    for seed in episode_seeds:
        state   = env.reset(seed=seed)
        done    = False
        total_r = 0.0
        steps   = 0

        while not done and steps < max_steps:
            logic_state, _ = state
            
            # 🚨 Convert to numpy array safely to avoid torch mp queue fd sharing errors
            if hasattr(logic_state, 'cpu'):
                logic_state_np = logic_state.cpu().numpy()
            else:
                logic_state_np = logic_state
                
            state_q.put((worker_id, _MSG_STATE, logic_state_np))
            msg = action_q.get()
            if msg[0] == _MSG_STOP:
                return
            predicate = msg[1]
            state, reward, done = env.step(predicate)
            total_r += reward
            steps   += 1

        state_q.put((worker_id, _MSG_DONE, total_r))

    state_q.put((worker_id, _MSG_STOP, None))


def evaluate_parallel(agent, env_name, num_episodes=50, seed=42,
                      num_workers=None, max_steps=10000, gaze_predictor=None, train_run=False):
    """
    Parallel evaluation using multiprocessing worker pool.

    Architecture:
      - N worker processes each own one NSFRBaseEnv (CPU only)
      - Workers step independently, send logic states to main via queue
      - Main process batches all pending states into ONE GPU forward pass
      - Actions dispatched back to the correct workers

    This turns N sequential single-sample GPU calls into batched calls
    of size N, dramatically improving GPU utilisation and total throughput.

    Args:
        agent:         ImitationAgent with loaded model weights
        env_name:      Environment name string e.g. 'seaquest'
        num_episodes:  Total episodes to evaluate
        seed:          Base seed — episode i uses seed+i
        num_workers:   Parallel envs. Default: min(cpu_count, num_episodes)
        max_steps:     Max steps per episode before forced termination
        gaze_predictor: Not supported in parallel mode — pass None

    Returns:
        List of total rewards, length == num_episodes
    """
    if gaze_predictor is not None:
        print("  Warning: gaze_predictor not supported in parallel eval.")
        print("  Falling back to sequential evaluation.")
        env = NSFRBaseEnv.from_name(env_name, mode='logic')
        return evaluate(agent, env, num_episodes=num_episodes,
                        seed=seed, gaze_predictor=gaze_predictor,
                        max_steps=max_steps)

    if num_workers is None:
        cpu_count   = mp.cpu_count()
        num_workers = min(cpu_count, num_episodes)
        print(f"  Auto-selected {num_workers} workers "
              f"({cpu_count} CPUs, {num_episodes} episodes)")

    num_workers = min(num_workers, num_episodes)
    device      = agent.device

    # Distribute episodes round-robin across workers
    # Worker i handles episodes: seed+i, seed+i+num_workers, seed+i+2*num_workers ...
    worker_seeds = [[] for _ in range(num_workers)]
    for ep in range(num_episodes):
        worker_seeds[ep % num_workers].append(seed + ep)
    # One shared inbound queue (workers -> main)
    # One outbound queue per worker (main -> worker)
    state_q   = mp.Queue()
    action_qs = [mp.Queue() for _ in range(num_workers)]

    if train_run:
        ctx = mp.get_context('spawn')
    else:
        ctx = mp.get_context('fork')
    # Fork worker processes (spawn is safer via forkserver/spawn with CUDA active)
    workers = []
    for wid in range(num_workers):
        p = ctx.Process(
            target=_env_worker,
            args=(wid, worker_seeds[wid], state_q, action_qs[wid],
                  env_name, max_steps),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # ── Main inference loop ───────────────────────────────────────────────────
    agent.model.eval()
    episode_rewards = []
    workers_done    = 0
    pending         = {}   # worker_id -> logic_state awaiting action
    inv_map         = {v: k for k, v in agent.primitive_action_map.items()}

    with torch.no_grad():
        while workers_done < num_workers or pending:
            # Drain queue — collect all available states before batching
            # timeout=0.005 gives workers 5ms to produce new states
            drained = False
            while True:
                try:
                    wid, msg_type, payload = state_q.get(
                        timeout=0.005 if not pending else 0.0
                    )
                    drained = True
                except Exception:
                    break

                if msg_type == _MSG_STATE:
                    pending[wid] = payload

                elif msg_type == _MSG_DONE:
                    episode_rewards.append(payload)
                    n = len(episode_rewards)
                    # Print seed number along with episode number
                    print(f"  Episode {n}/{num_episodes} (Seed: {worker_seeds[wid]}): Reward = {payload:.1f}")

                elif msg_type == _MSG_STOP:
                    workers_done += 1

            # Run batched GPU inference for all workers currently waiting
            if pending:
                wids   = list(pending.keys())
                states = np.stack([pending[w] for w in wids])

                batch = torch.tensor(states, dtype=torch.float32, device=device)
                _, action_scores = agent.predict(batch)   # (B, num_actions)

                for wid, scores in zip(wids, action_scores):
                    predicate = inv_map[scores.argmax().item()]
                    action_qs[wid].put((_MSG_STATE, predicate))

                pending.clear()

    # Clean up worker processes
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

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
    parser.add_argument("--use_gazemap",        action="store_true")
    parser.add_argument("--gaze_model_path",    type=str,
                        default="seaquest_gaze_predictor_2.pth")
    parser.add_argument("--send_email",         action="store_true")
    parser.add_argument("--num_workers",        type=int, default=None,
                        help="Parallel envs (default: auto from CPU count)")
    parser.add_argument("--sequential",         action="store_true",
                        help="Force sequential eval (required with --use_gazemap)")
    parser.add_argument("--max_steps",          type=int, default=10000)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    gaze_predictor = None
    if args.use_gazemap:
        if Human_Gaze_Predictor is None:
            print("Error: Could not import Human_Gaze_Predictor.")
            return
        print(f"Initializing Gaze Predictor from {args.gaze_model_path}...")
        gaze_predictor = Human_Gaze_Predictor(args.env)
        gaze_predictor.init_model(args.gaze_model_path)
        gaze_predictor.model.eval()

    make_deterministic(args.seed)

    gaze_threshold = args.gaze_threshold if args.use_gaze else None
    agent = ImitationAgent(args.env, args.rules, device,
                           gaze_threshold=gaze_threshold)

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    print(f"Loading model from {args.model_path}...")
    agent.load(args.model_path)
    print(f"Starting evaluation: {args.episodes} episodes...")

    use_parallel = not args.sequential and gaze_predictor is None
    if use_parallel:
        rewards = evaluate_parallel(
            agent,
            env_name=args.env,
            num_episodes=args.episodes,
            seed=args.seed,
            num_workers=args.num_workers,
            max_steps=args.max_steps,
        )
    else:
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