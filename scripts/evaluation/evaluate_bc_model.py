import os
import cv2
import glob
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from core.utils.utils import preprocess_frame

from core.utils.linear_models import Encoder

PRIMITIVE_ACTIONS = {0: 'noop', 1: 'fire', 2: 'up', 3: 'right', 4: 'left', 5: 'down'}


# ── GABRILEnvWrapper — inlined to avoid cross-module import issues ─────────────
class GABRILEnvWrapper:
    """
    Wraps NSFRBaseEnv to match GABRIL's Atari evaluation settings:
      - frame_skip=4             repeat action 4 ALE frames, sum rewards
      - Sticky actions           (action_repeat_probability=0.25)
      - noop_max=30              random no-ops on reset
      - terminal_on_life_loss    treat life loss as episode end

    noop_action should be an integer (0) when BC steps with is_mapped=True.
    All NSFRBaseEnv attributes/methods are proxied through transparently.
    """

    def __init__(self, env, action_repeat_probability=0.25, noop_max=30,
                 terminal_on_life_loss=True, noop_action=0, seed=42,
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
                state, _, done = self._env.step(self.noop_action, is_mapped=True)
                if done:
                    state = self._env.reset(seed=seed, options=options)

        lives = self._get_lives()
        if lives is not None:
            self._lives         = lives
            self._ale_available = True
        else:
            self._ale_available = False

        return state

    def step(self, action, is_mapped=False):
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
            state, reward, done = self._env.step(action, is_mapped=is_mapped)
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

    def get_rgb_frame(self):
        return self._env.get_rgb_frame()

    def extract_logic_state(self, raw_state):
        return self._env.extract_logic_state(raw_state)

    def extract_neural_state(self, raw_state):
        return self._env.extract_neural_state(raw_state)

    def close(self):
        self._env.close()


def load_bc_model(run_dir, gaze_method="None", device="cuda", ckpt_prefix="best_", stack=4):
    """
    Loads encoder, pre_actor, and actor weights for testing.
    Also loads encoder_agil if AGIL is the method.
    """
    dev = torch.device(device)

    embedding_dim        = 64
    num_hiddens          = 128
    num_residual_layers  = 2
    num_residual_hiddens = 32
    z_dim                = 256
    encoder_out_dim      = 8 * 8 * embedding_dim
    action_dim           = 6

    actor_ckpt_path = f"{run_dir}/{ckpt_prefix}actor.pth"
    if not os.path.exists(actor_ckpt_path):
        if not ckpt_prefix:
            epoch_dirs = glob.glob(os.path.join(run_dir, "*_ep"))
            if epoch_dirs:
                alt_ckpt = f"{epoch_dirs[0]}/best_actor.pth"
                if os.path.exists(alt_ckpt):
                    actor_ckpt_path = alt_ckpt
            else:
                print("No epoch subdirectories found either.")

    if os.path.exists(actor_ckpt_path):
        try:
            dummy_ckpt = torch.load(actor_ckpt_path, map_location='cpu', weights_only=False)
            if '2.bias' in dummy_ckpt:
                action_dim = dummy_ckpt['2.bias'].shape[0]
                print(f"Inferred action_dim={action_dim} from weights.")
        except Exception as e:
            print(f"Error inferring action_dim from weights: {e}")

    encoder = Encoder(stack, embedding_dim, num_hiddens,
                      num_residual_layers, num_residual_hiddens).to(dev)

    pre_actor = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(encoder_out_dim, z_dim),
        nn.ReLU()
    ).to(dev)

    actor = nn.Sequential(
        nn.Linear(z_dim, z_dim), nn.ReLU(),
        nn.Linear(z_dim, action_dim)
    ).to(dev)

    encoder_agil = None
    if gaze_method == "AGIL":
        encoder_agil = Encoder(stack, embedding_dim, num_hiddens,
                               num_residual_layers, num_residual_hiddens).to(dev)

    if actor_ckpt_path and os.path.exists(actor_ckpt_path):
        try:
            encoder.load_state_dict(
                torch.load(f"{run_dir}/{ckpt_prefix}encoder.pth", map_location=dev))
            pre_actor.load_state_dict(
                torch.load(f"{run_dir}/{ckpt_prefix}pre_actor.pth", map_location=dev))
            actor.load_state_dict(
                torch.load(actor_ckpt_path, map_location=dev))
            if encoder_agil is not None:
                encoder_agil.load_state_dict(
                    torch.load(f"{run_dir}/{ckpt_prefix}encoder_agil.pth", map_location=dev))
        except FileNotFoundError as e:
            print(f"Error loading weights: {e}")
            print("Returning randomly initialised models.")
    else:
        print(f"Warning: {run_dir} not found or no actor checkpoint resolved. "
              "Returning randomly initialised models.")

    encoder.eval()
    pre_actor.eval()
    actor.eval()
    if encoder_agil is not None:
        encoder_agil.eval()

    return encoder, pre_actor, actor, encoder_agil


def evaluate_bc_model(env, run_dir, gaze_method="None", num_episodes=10, seed=42,
                      device="cuda", use_gaze=False,
                      gaze_model_path="seaquest_gaze_predictor_2.pth",
                      ckpt_prefix="best_", stack=4,
                      gabril_compat=False):
    """
    Loads a pretrained BC/AGIL baseline and runs it in the provided environment.

    Args:
        env:             NSFRBaseEnv instance (raw — wrapper applied here if gabril_compat)
        run_dir:         Path containing the pretrained .pth files
        gaze_method:     "None", "AGIL", or "Mask"
        num_episodes:    Number of episodes to evaluate
        seed:            Base seed. Episode i uses seed + 1000*i (GABRIL formula)
        device:          "cuda" or "cpu"
        use_gaze:        If True, load live Human_Gaze_Predictor
        gaze_model_path: Path to gaze predictor weights
        ckpt_prefix:     Checkpoint prefix (default "best_")
        stack:           Encoder frame stack depth
        gabril_compat:   If True, wrap env with GABRILEnvWrapper:
                           - sticky actions (p=0.25)
                           - noop_max=30
                           - terminal_on_life_loss=True
                           Seeds use GABRIL formula (seed + 1000 * episode_idx)
                           regardless of this flag.

    Returns:
        list of total rewards, length == num_episodes
    """
    dev = torch.device(device)

    # ── Apply GABRIL env wrapper if requested ─────────────────────────────────
    # noop_action=0 (integer) because BC calls env.step(action_idx, is_mapped=True)
    if gabril_compat:
        env = GABRILEnvWrapper(
            env,
            action_repeat_probability=0.25,
            noop_max=30,
            terminal_on_life_loss=True,
            noop_action=0,   # integer — BC steps with is_mapped=True
            seed=seed,
            frame_skip=1,
        )
        print("  GABRILEnvWrapper applied: sticky=0.25, noop_max=30, "
              "terminal_on_life_loss=True, frame_skip=4")

    encoder, pre_actor, actor, encoder_agil = load_bc_model(
        run_dir, gaze_method, device, ckpt_prefix=ckpt_prefix, stack=stack
    )

    gaze_predictor = None
    if (use_gaze or gaze_method in ['ViSaRL', 'Mask', 'AGIL']) and gaze_method != "None":
        try:
            from scripts.gaze_predictor import Human_Gaze_Predictor
            print(f"Initialising Test-Time Gaze Predictor from {gaze_model_path}...")
            gaze_predictor = Human_Gaze_Predictor("seaquest")
            gaze_predictor.init_model(gaze_model_path)
            gaze_predictor.model.eval()
        except ImportError:
            print("Warning: Could not import Human_Gaze_Predictor! Gaze = 0.")

    rewards = []

    for i in range(num_episodes):
        # ── GABRIL seed formula: seed + 1000 * episode_idx ───────────────────
        ep_seed = seed + 1000 * i

        try:
            state = env.reset(seed=ep_seed)
        except TypeError:
            state = env.reset()

        done    = False
        total_r = 0.0

        # 4-frame buffer for gaze predictor (always depth-4 regardless of stack)
        frame_buffer = deque(maxlen=4)
        raw_frame    = (env.get_rgb_frame() if hasattr(env, 'get_rgb_frame')
                        else env.render())
        gray = preprocess_frame(raw_frame)
        for _ in range(4):
            frame_buffer.append(gray)

        from tqdm import tqdm
        pbar        = tqdm(desc=f"Episode {i+1}")
        step_count  = 0
        valid_actions = (env.pred2action if hasattr(env, 'pred2action')
                         else {0: 'noop', 1: 'fire', 2: 'up',
                               3: 'right', 4: 'left', 5: 'down'})
        action_counts = {}
        action_idx_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        # ── GABRIL step cap: 5000 ─────────────────────────────────────────────
        while not done and step_count < 5000:
            step_count += 1

            raw_frame = (env.get_rgb_frame() if hasattr(env, 'get_rgb_frame')
                         else env.render())
            gray = preprocess_frame(raw_frame)

            img_stack_4 = np.stack(frame_buffer, axis=-1)  # (84, 84, 4)

            if stack == 1:
                xx = torch.tensor(gray, dtype=torch.float32, device=dev
                                  ).unsqueeze(0).unsqueeze(0)  # (1,1,84,84)
            else:
                img_stack_agent = np.stack(list(frame_buffer)[-stack:], axis=-1)
                xx = torch.tensor(img_stack_agent, dtype=torch.float32, device=dev
                                  ).permute(2, 0, 1).unsqueeze(0)  # (1,stack,84,84)

            gg = torch.zeros(1, 1, 84, 84, device=dev)
            if gaze_predictor is not None:
                input_tensor = (
                    torch.tensor(img_stack_4, dtype=torch.float32,
                                 device=gaze_predictor.device)
                    .permute(2, 0, 1).unsqueeze(0)  # (1,4,84,84)
                )
                with torch.no_grad():
                    gg = gaze_predictor.predict_normalized(input_tensor).to(dev)

            with torch.no_grad():
                xx_in = xx * gg if gaze_method == 'Mask' else xx
                z     = encoder(xx_in)
                if gaze_method == 'AGIL' and encoder_agil is not None:
                    z = (z + encoder_agil(xx * gg)) / 2
                logits     = actor(pre_actor(z))
                action_idx = logits.argmax(dim=1).item()

            
            if action_idx in action_idx_counts.keys():
                action_idx_counts[action_idx] = action_idx_counts[action_idx] + 1
            action_str = valid_actions.get(action_idx, "noop")
            action_counts[action_str] = action_counts.get(action_str, 0) + 1

            step_result = env.step(action_idx, is_mapped=True)
            if len(step_result) == 3:
                state, reward, done = step_result
            else:
                state, reward, done, truncated, info = step_result
                if truncated:
                    done = True

            total_r += reward

            if not done:
                next_raw  = (env.get_rgb_frame() if hasattr(env, 'get_rgb_frame')
                             else env.render())
                next_gray = cv2.cvtColor(next_raw, cv2.COLOR_RGB2GRAY)
                next_gray = cv2.resize(next_gray, (84, 84),
                                       interpolation=cv2.INTER_AREA) / 255.0
                frame_buffer.append(next_gray)

            pbar.update(1)

        pbar.close()
        rewards.append(total_r)
        print(f"Episode {i+1}/{num_episodes} — Reward: {total_r}  Steps: {step_count}")
        print(f"  Action Distribution: {action_counts}")
        print(f"  Action Index Distribution: {action_idx_counts}")
    return rewards

def main():
    import argparse
    from nsfr.env import NSFRBaseEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",        type=str, required=True)
    parser.add_argument("--gaze_method",    type=str, default="None",
                        choices=["None", "AGIL", "Mask"])
    parser.add_argument("--episodes",       type=int, default=100)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--device",         type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_gaze",       action="store_true")
    parser.add_argument("--gaze_model_path",type=str,
                        default="gaze_models/seaquest/seaquest_gaze_predictor_2.pth")
    parser.add_argument("--ckpt_prefix",    type=str, default="best_")
    parser.add_argument("--stack",          type=int, default=1)
    parser.add_argument("--gabril_compat",  action="store_true",
                        help="Apply GABRIL env: sticky p=0.25, noop_max=30, "
                             "terminal_on_life_loss, seed=base+1000*ep")
    args = parser.parse_args()

    print(f"Initialising NSFRBaseEnv for seaquest...")
    test_env = NSFRBaseEnv.from_name("seaquest", mode='logic')

    print(f"Loading {args.gaze_method} BC model from: {args.run_dir} "
          f"(prefix: {args.ckpt_prefix})")
    eval_rewards = evaluate_bc_model(
        test_env, args.run_dir,
        gaze_method=args.gaze_method,
        num_episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        use_gaze=args.use_gaze,
        gaze_model_path=args.gaze_model_path,
        ckpt_prefix=args.ckpt_prefix,
        stack=args.stack,
        gabril_compat=args.gabril_compat,
    )

    print(f"\nFinal Evaluation over {args.episodes} episodes:")
    print(f"Mean Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Min / Max:   {min(eval_rewards):.0f} / {max(eval_rewards):.0f}")
    
# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()