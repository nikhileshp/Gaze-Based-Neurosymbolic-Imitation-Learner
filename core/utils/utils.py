import sys, os as _os
_scripts_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_project_root = _os.path.dirname(_scripts_dir)
for _p in [_scripts_dir, _project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import os
import random
import gymnasium as gym
from tqdm import tqdm
from core.utils.gaze_utils import GazeToMask
import cv2

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from core.utils.gaze_utils import apply_gmd_dropout

MAX_EPISODES = {'Alien': 20, 'Asterix': 20, 'Assault': 20, 'Breakout': 20, 'ChopperCommand': 20,
                'DemonAttack': 20, 'Enduro': 20, 'Frostbite': 20, 'Freeway': 20, 'MsPacman': 20,
                'Phoenix': 20, 'Qbert': 20, 'RoadRunner': 20, 'Seaquest': 50, 'UpNDown': 20}

MAX_EPISODES_ATARI_HEAD = {'Seaquest': 20, 'MsPacman': 20, 'Enduro':20, 'Freeway': 10, 'Hero': 20, 'BankHeist': 20} # Atari_Head data

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Also set deterministic flag if available
    try:
        from nsfr.utils import make_deterministic
        make_deterministic(seed)
    except ImportError:
        pass

def preprocess_frame(frame):
    """Convert raw 210x160x3 RGB frame to 84x84 grayscale frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0

def format_results_table(results_log):
    if not results_log:
        return "No results yet."
    
    # Check what columns we have
    keys = results_log[0].keys()
    header = " | ".join([f"{k:^10}" for k in keys])
    divider = "-" * len(header)
    rows = []
    for entry in results_log:
        row = " | ".join([f"{str(entry.get(k, 'N/A')):^10}" for k in keys])
        rows.append(row)
    return "\n".join([header, divider] + rows)

def send_run_update(args, results_log, current_epoch, metrics, is_final=False, is_started=False, task_name="Training"):
    """
    Unified email update function.
    metrics: dict of values to display (e.g. {'last_loss': 0.5, 'best_reward': 100})
    """
    from core.utils.email_me import send_email
    import time
    
    status_prefix = "Started" if is_started else ("Final" if is_final else "Periodic")
    num_ep = getattr(args, 'num_episodes', 'All')
    
    subject = f"Server Run Update: {args.env} - {task_name} | Epoch {current_epoch} | {status_prefix}"
    
    metrics_str = "\n".join([f"{k.replace('_', ' ').title()}: {v if isinstance(v, str) else f'{v:.4f}'}" for k, v in metrics.items()])
    
    body = f"""
Status: {status_prefix} Update
Environment: {args.env}
Task: {task_name}
Episodes: {num_ep}
Current Epoch: {current_epoch}

Metrics:
{metrics_str}

Progress Table:
{format_results_table(results_log)}
"""
    send_email(subject, body.strip())

def load_pt_dataset(pt_path, num_episodes=None, use_gaze=False, stack=1):
    print(f"Loading custom dataset from {pt_path} ...")
    data = torch.load(pt_path, map_location='cpu', weights_only=False)

    obs      = data['observations']            # (N, H, W) uint8
    actions  = data['actions']                 # (N,)
    ep_nums  = data.get('episode_number', None)
    gaze     = data.get('gaze_image', None)    # (N, 84, 84) float32
    gaze_info = data.get('gaze_information', None)

    if not isinstance(obs, torch.Tensor):      obs = torch.from_numpy(obs)
    if not isinstance(actions, torch.Tensor):  actions = torch.from_numpy(actions)
    if ep_nums is not None and not isinstance(ep_nums, torch.Tensor):
        ep_nums = torch.from_numpy(ep_nums)
    if gaze is not None and not isinstance(gaze, torch.Tensor):
        gaze = torch.from_numpy(gaze)
    if gaze_info is not None and not isinstance(gaze_info, torch.Tensor):
        gaze_info = torch.from_numpy(gaze_info)

    actions = actions.long()
    obs     = obs.byte()

    # Filter to supported actions (0-5)
    mask = (actions <= 5)
    obs, actions = obs[mask], actions[mask]
    if ep_nums is not None: ep_nums = ep_nums[mask]
    if gaze is not None:    gaze    = gaze[mask]
    if gaze_info is not None: gaze_info = gaze_info[mask]

    # Handle Frame Stacking across episodes
    if ep_nums is not None:
        unique_eps = torch.unique(ep_nums)
        if num_episodes is not None:
            unique_eps = unique_eps[:num_episodes]
        
        stacked_obs = []
        stacked_gaze = []
        stacked_coords = []
        final_actions = []
        final_ep_nums = []

        for ep in unique_eps:
            mask_ep = (ep_nums == ep)
            o_ep = obs[mask_ep]
            a_ep = actions[mask_ep]
            g_ep = gaze[mask_ep] if gaze is not None else torch.zeros(len(o_ep), 84, 84)
            gc_ep = gaze_info[mask_ep][:, :2].float() if gaze_info is not None else torch.zeros(len(o_ep), 2)

            if len(o_ep) == 0: continue

            # Sliding Window For Stack Parameter
            pad_o = o_ep[0].unsqueeze(0).repeat(stack - 1, 1, 1)
            o_padded = torch.cat([pad_o, o_ep], dim=0)
            stacked_obs.append(torch.stack([o_padded[i:i+stack] for i in range(len(o_ep))]))

            pad_g = g_ep[0].unsqueeze(0).repeat(stack - 1, 1, 1)
            g_padded = torch.cat([pad_g, g_ep], dim=0)
            stacked_gaze.append(torch.stack([g_padded[i:i+stack] for i in range(len(g_ep))]))
            
            final_actions.append(a_ep)
            stacked_coords.append(gc_ep)
            final_ep_nums.append(ep_nums[mask_ep])

    if ep_nums is not None:
        obs     = torch.cat(stacked_obs, dim=0)
        gaze    = torch.cat(stacked_gaze, dim=0).float()
        actions = torch.cat(final_actions, dim=0)
        gaze_coords = torch.cat(stacked_coords, dim=0)
        final_ep_nums = torch.cat(final_ep_nums, dim=0)
    else:
        obs = obs.unsqueeze(1)
        gaze = gaze.unsqueeze(1).float() if gaze is not None else torch.zeros(len(obs), 1, 84, 84)
        gaze_coords = gaze_info[:, :2].float() if gaze_info is not None else torch.zeros(len(obs), 2)
        final_ep_nums = torch.zeros(len(obs), dtype=torch.int64)

    if not use_gaze:
        gaze = torch.zeros(len(obs), stack, 84, 84)

    if gaze.max() > 1.0: gaze = gaze / 255.0

    return obs, actions, gaze, gaze_coords, final_ep_nums


def load_dataset(env, seed, datapath, conf_type, conf_randomness, stack, num_episodes=None, use_gaze=False, data_source='Our', gaze_mask_sigma=15.0, gaze_mask_coef=0.7):
    if num_episodes:
        if data_source == 'Atari_HEAD':
            assert num_episodes <= MAX_EPISODES_ATARI_HEAD[env], f'The number of available episodes is {MAX_EPISODES_ATARI_HEAD[env]}, but {num_episodes} is requested'
        else:
            assert num_episodes <= MAX_EPISODES[env], f'The number of available episodes is {MAX_EPISODES[env]}, but {num_episodes} is requested'
    
    # load the data
    path = os.path.join(datapath, env)
    file_name = f"/num_episodes_{MAX_EPISODES[env]}_fs4_human.pt"
    atari_head_file_name = f'Atari_Head_Data/{env}/ordinary.pt'
    loaded_obj = torch.load(atari_head_file_name) if data_source == 'Atari_HEAD' else torch.load(path + file_name, weights_only=False)

    obs = loaded_obj['observations']
    rewards = loaded_obj['episode-rewards']
    actions = loaded_obj['actions']
    episode_lengths = loaded_obj['steps']
    truncateds = loaded_obj['truncateds']
    terminateds = loaded_obj['terminateds']
    
    gaze_info = loaded_obj['gaze_information'] if use_gaze else None

    assert num_episodes <= episode_lengths.shape[0], "num_episodes should be less than total saved episodes"

    obs = torch.from_numpy(obs)
    actions = torch.from_numpy(actions)
    episode_lengths = torch.from_numpy(episode_lengths)
    truncateds = torch.from_numpy(truncateds)
    terminateds = torch.from_numpy(terminateds)

    if use_gaze:
        gaze_info = torch.from_numpy(gaze_info)

        g = gaze_info[:, :2]
        g[g < 0] = 0
        g[g > 1] = 1

        gaze_info[:, :2] = g


    short_memory_length = 20 
    stride = 2

    saliency_sigmas = [gaze_mask_sigma/(0.99**(short_memory_length - i)) for i in range(short_memory_length+1)]
    coeficients = [gaze_mask_coef**(short_memory_length - i) for i in range(short_memory_length+1)]
    coeficients += coeficients[::-1][1:]
    saliency_sigmas += saliency_sigmas[::-1][1:]

    MASK = GazeToMask(84, saliency_sigmas, coeficients=coeficients)

    episode_obs = []
    episode_actions = []
    
    episode_gaze = []

    for episode, _ in enumerate(episode_lengths):
        start = sum(episode_lengths[:episode])
        end = start + episode_lengths[episode]
        episode_obs.append(
            obs[start + episode:end + episode + 1])
    
        episode_actions.append(actions[start:end])
        if use_gaze:
            episode_gaze.append(gaze_info[start:end])
        assert terminateds[end - 1] or truncateds[end - 1]

    if num_episodes:
        episode_obs = episode_obs[:num_episodes]
        episode_actions = episode_actions[:num_episodes]
        if use_gaze:
            episode_gaze = episode_gaze[:num_episodes]

    episode_obs = [ep[:-1] for ep in episode_obs]
    if use_gaze:
        episode_saliency_gaze = [torch.stack(
                [MASK.find_bunch_of_maps(means=
                    ep[max(0, j - stride * short_memory_length):min(short_memory_length * stride + j + 1, len(ep)):stride],
                        offset_start=max(short_memory_length - j , 0))
                            for j in range(len(ep))], 0)
                                for ep in tqdm (episode_gaze)]
        episode_gaze_coordinates = [ep[:, :2] for ep in episode_gaze]
        
    else:
        episode_saliency_gaze =    [torch.zeros_like(episode_obs[i], dtype=torch.float32) for i in range(len(episode_obs))]
        episode_gaze_coordinates = [torch.zeros((len(episode_obs[i]), 2), dtype=torch.float32) for i in range(len(episode_obs))]
    
    # repeat the first frame for stack - 1 times
    episode_obs = [torch.cat([ep[0].unsqueeze(0)] * (stack - 1) + [ep]) for ep in episode_obs]
    episode_saliency_gaze = [torch.cat([ep[0].unsqueeze(0)] * (stack - 1) + [ep]) for ep in episode_saliency_gaze]

    for i, (ep_obs, ep_gaze) in enumerate(zip(episode_obs, episode_saliency_gaze)):
        new_episode_obs = []
        new_episode_saliency_gaze = []
        for s in range(stack):
            end = None if s == stack - 1 else s - stack + 1
            new_episode_obs.append(ep_obs[s:end])
            new_episode_saliency_gaze.append(ep_gaze[s:end])
        
        episode_obs[i] = torch.stack(new_episode_obs, dim=1)
        episode_saliency_gaze[i] = torch.stack(new_episode_saliency_gaze, dim=1)

    episode_actions = [ep[1:] for ep in episode_actions]
    episode_saliency_gaze = [ep[1:] for ep in episode_saliency_gaze]
    episode_gaze_coordinates = [ep[1:] for ep in episode_gaze_coordinates]

    observations = torch.cat(episode_obs)
    actions = torch.cat(episode_actions)
    gaze_saliency_maps = torch.cat(episode_saliency_gaze)
    gaze_coordinates = torch.cat(episode_gaze_coordinates)

    
    return observations, actions, gaze_saliency_maps, gaze_coordinates

def evaluate(env, pre_actor, actor, model, device, args, verbose=False):
    model.eval()
    pre_actor.eval()
    actor.eval()
    if args.gaze_method == 'AGIL':
        args.encoder_agil.eval()
    
    # Optimization: tell RenderWrapper if we are recording
    try:
        env.get_wrapper_attr('set_recording')(args.eval_record)
    except AttributeError:
        pass

    # Infer the action dimension dynamically from the actor's last linear layer
    # The actor is typically an nn.Sequential where the last layer is an nn.Linear
    last_layer = list(actor.children())[-1]
    if isinstance(last_layer, nn.Linear):
        num_actions = last_layer.out_features
    else:
        num_actions = env.action_space.n # Fallback

    scores = []
    for episode in tqdm(range(args.num_eval_episodes)):

        done = False
        episode_reward = 0
        step = 0

        stacked_obs, info = env.reset(seed=args.seed + episode)
        rnd_generator = np.random.default_rng(args.seed + episode)
        prev_action = 0

        while not done:

            stacked_obs = np.asarray(stacked_obs, dtype=np.uint8)

            if args.eval_type == 'confounded':
                action_index = prev_action
            else:
                action_index = None

            if args.eval_type in ['confounded']:
                if rnd_generator.random() < args.randomness:
                    action_index = rnd_generator.integers(num_actions)

            if args.eval_record and episode < 2:
                _ = env.get_wrapper_attr('render_')(gaze=None, record_frame=True)

            with torch.no_grad():
                stacked_obs = torch.as_tensor(stacked_obs, device=device, dtype=torch.float32).unsqueeze(0) / 255.0
                if args.gaze_method in ['ViSaRL', 'Mask', 'AGIL'] or args.dp_method in ['IGMD', 'GMD']: # models that need gaze prediction
                    g = args.gaze_predictor(stacked_obs)
                    g[g < 0] = 0
                    g[g > 1] = 1
                
                if args.gaze_method == 'ViSaRL':
                    stacked_obs = torch.cat([stacked_obs, g], dim=1)
                elif args.gaze_method == 'Mask':
                    stacked_obs = stacked_obs * g
                
                dropout_mask = None
                if args.dp_method == 'IGMD':
                    dropout_mask = g[:,-1:]

                z = model(stacked_obs, dropout_mask=dropout_mask)
                
                if args.gaze_method == 'AGIL':
                    z = (z + args.encoder_agil(stacked_obs * g)) / 2
                
                if args.dp_method == 'GMD':
                    z = apply_gmd_dropout(z, g[:,-1:], test_mode=True)

                z = pre_actor(z)
                action = actor(z).argmax(1)[0].cpu().item()

            stacked_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            prev_action = action

            episode_reward += reward
            step += 1
            
            if step > 5000:
                done = True

        scores.append(episode_reward)
        if verbose:
            print(
                f"Episode {episode} reward: {episode_reward}(mean: {np.mean(scores)}, std: {np.std(scores)}), steps: {step}")

    if args.eval_record:
        print(env.get_wrapper_attr('save_video')(f'vids/{args.add_path}.mp4'))
    return np.mean(scores), np.std(scores)

def plot_gaze_and_obs(gaze, obs, save_path=None):
    # Create data for the plots
    y1 = gaze
    
    y2 = obs
    if obs.dtype == torch.uint8:
        y2 = obs.to(torch.float32)/255
    
    y3 = (y1 * y2).to(torch.float32)

    if len(y3.shape) == 3:
        if y3.shape[0] == 3:
            y3 = y3.permute(1, 2, 0)
        elif y3.shape[0] == 1:
            y3 = y3[0]

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Plot 1
    ax1.imshow(y1, cmap='gray', vmax=1.0, vmin=0.0)
    ax1.set_title('pure gaze')

    # Plot 2
    ax2.imshow(y2, cmap='gray', vmax=1.0, vmin=0.0)
    ax2.set_title('pure obs')

    # Plot 3
    ax3.imshow(y3, cmap='gray', vmax=1.0, vmin=0.0)
    ax3.set_title('merged gaze and obs')
    
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])

    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    
    # Show the plots
    plt.show()

if __name__ == '__main__':
    observations, actions, gaze_saliency_maps, gaze_coordinates =   load_dataset(env='Seaquest', 
                                                                    seed=1, 
                                                                    datapath='dataset', 
                                                                    conf_type='normal', 
                                                                    conf_randomness=0.0, 
                                                                    stack=1, 
                                                                    num_episodes=50, 
                                                                    use_gaze=True)