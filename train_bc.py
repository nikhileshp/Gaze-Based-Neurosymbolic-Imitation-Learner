
import torch
import torch.nn as nn
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import numpy as np

from baselines.models.linear_models import Encoder, weight_init, VectorQuantizer, Decoder, AutoEncoder
from atari_env_manager import create_env as create_atari_environment
from baselines.gaze.gaze_utils import get_gaze_mask, apply_gmd_dropout
from torch.utils.tensorboard import SummaryWriter
import datetime
import pandas as pd
from collections import deque
import time
from scripts.email_me import send_email

from nudge.env import NudgeBaseEnv
from nudge.utils import make_deterministic
from atari_env_manager import create_env as create_atari_environment
from scripts.utils import evaluate

def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    make_deterministic(seed)

def format_results_table(results_log):
    if not results_log:
        return "No results yet."
    
    header = f"{'Epoch':<6} | {'Loss':<10} | {'Mean Reward':<12} | {'Std Reward':<12}"
    divider = "-" * len(header)
    rows = []
    for res in results_log:
        epoch = res.get('epoch', '-')
        loss = res.get('train_loss', 0.0)
        mean_r = res.get('mean_reward', 0.0)
        std_r = res.get('std_reward', 0.0)
        # Handle NaN for mean_reward if evaluation didn't happen this epoch
        mean_r_str = f"{mean_r:<12.2f}" if not np.isnan(mean_r) else f"{'N/A':<12}"
        std_r_str = f"{std_r:<12.2f}" if not np.isnan(std_r) else f"{'N/A':<12}"
        rows.append(f"{epoch:<6} | {loss:<10.4f} | {mean_r_str} | {std_r_str}")
    
    return "\n".join([header, divider] + rows)

def send_run_update(args, results_log, current_ep, current_epoch, last_loss, best_loss, last_reward, best_reward, is_final=False, is_started=False):
    method = args.gaze_method
    num_ep = args.num_episodes if args.num_episodes is not None else "All"
    
    if args.incremental:
        training_type = "Incremental"
    elif args.independent:
        training_type = "Independent"
    else:
        training_type = "Standard"
        
    if is_started:
        status_prefix = "Started"
    elif is_final:
        status_prefix = "Final"
    else:
        status_prefix = "Periodic"
    
    subject = f"Server Run Update: Run {args.env} - BC Training - {method} | Training Type: {training_type} | Using {num_ep} episodes | Using {args.n_epochs} epochs"
    
    # Format NaN rewards
    last_reward_str = f"{last_reward:.2f}" if not np.isnan(last_reward) else "N/A"
    best_reward_str = f"{best_reward:.2f}" if not np.isnan(best_reward) else "N/A"

    body = f"""
Status: {status_prefix} Update
Environment: {args.env}
Method: {method}
Episodes: {num_ep}
Current Episode: {current_ep}
Current Epoch: {current_epoch}
Last Train Loss: {last_loss:.4f}
Best Train Loss: {best_loss:.4f}
Last Mean Reward: {last_reward_str}
Best Mean Reward: {best_reward_str}

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

def get_args(run_in_notebook=False):
    import argparse
    parser = argparse.ArgumentParser()

    # Seed & Env
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--env", default="Seaquest", type=str)
    parser.add_argument("--datapath", default="data/seaquest/full_data_28_episodes_10p0_sigma_win_10_obj_49.pt", type=str)
    parser.add_argument("--stack", default=1, type=int)
    parser.add_argument("--train_type", default='normal', choices=['confounded', 'normal'], type=str)
    parser.add_argument("--eval_type", default='normal', choices=['confounded', 'normal'], type=str)
    parser.add_argument("--eval_fs", default=4, type=int, help='Frame skip for evaluation')
    parser.add_argument("--randomness", default=0.0, type=float)
    parser.add_argument("--sticky_probability", default=0.25, type=float)

    # Save & Evaluation
    parser.add_argument("--save_interval", default=500, type=int)
    parser.add_argument("--eval_interval", default=100, type=int)
    parser.add_argument("--num_episodes", default=28, type=int)
    parser.add_argument("--num_eval_episodes", default=50, type=int)
    parser.add_argument("--n_epochs", default=500, type=int)
    parser.add_argument("--add_path", default="", type=str)
    parser.add_argument("--result_save_dir", default="", type=str)
    parser.add_argument("--val_split_ratio", default=0.95, type=float)
    parser.add_argument("--dataset_source", default='Our', choices=['Atari-Head', 'Our'], type=str)

    # Encoder & Hyperparams
    parser.add_argument("--embedding_dim", default=64, type=int)
    parser.add_argument("--num_embeddings", default=512, type=int)
    parser.add_argument("--num_hiddens", default=128, type=int)
    parser.add_argument("--num_residual_layers", default=2, type=int)
    parser.add_argument("--num_residual_hiddens", default=32, type=int)
    parser.add_argument("--bs", default=512, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--step_size", default=100, type=int)
    parser.add_argument("--wd", default=0, type=float)

    # For MLP
    parser.add_argument("--z_dim", default=256, type=int)



    parser.add_argument("--eval_render_mode", default='rgb_array', choices=['rgb_array', 'human'], type=str)
    parser.add_argument("--eval_record", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)

    parser.add_argument("--gaze_method", type=str, default='BC', choices=['BC', 'None', 'Teacher', 'Reg', 'Mask', 'Contrastive', 'ViSaRL', 'AGIL', 'GRIL'])
    parser.add_argument("--dp_method", type=str, default='None', choices=['None', 'Oreo', 'IGMD', 'GMD'], help='Dropout method')
    parser.add_argument("--gaze_mask_sigma", type=float, default=15.0, help='Sigma of the Gaussian for the gaze mask')
    parser.add_argument("--gaze_mask_coeff", type=float, default=0.7, help='Base coefficient of the Gaussian for the gaze mask')
    parser.add_argument("--gaze_ratio", type=float, default=1.0, help='Ratio of episodes to use for gaze prediction')
    parser.add_argument("--gaze_beta", type=float, default=50.0, help='Softmax temperature for GABRIL')
    parser.add_argument("--gaze_lambda", type=float, default=10, help='Loss coefficient hyperparameter')
    parser.add_argument("--gaze_contrastive_threshold", type=float, default=0, help='Contrastive loss margin hyperparameter for the Contrastive method')
    parser.add_argument("--prob_dist_type", type=str, default="MSE", choices=["MSE", "TV", "KL", "JS"])
    parser.add_argument("--oreo_prob", default=0.5, type=float)
    parser.add_argument("--oreo_num_mask", default=5, type=int)
    parser.add_argument("--incremental", action="store_true", default=False, help="Train incrementally episode by episode")
    parser.add_argument("--independent", action="store_true", default=False, help="Train on independent episodes (one at a time) without resetting model weights")
    parser.add_argument("--result_csv", type=str, default="", help="Path to save results in CSV format")
    parser.add_argument("--send_email", action="store_true", help="Enable periodic email updates")
    parser.add_argument("--email_interval", type=int, default=30, help="Interval in minutes between email updates")

    if run_in_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    args.device = 'cuda'

    return args

def train(args, verbose=False):
     
    device = torch.device(args.device)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    observations, actions, gaze_masks, gaze_coordinates, ep_nums = load_pt_dataset(
        args.datapath, args.num_episodes, use_gaze=(args.gaze_method not in ["None", "BC"]), stack=args.stack
    )
    
    args.gaze_predictor = None
    # Methods that need gaze predictor
    if args.gaze_method in ['ViSaRL', 'Mask', 'Teacher', 'AGIL'] or args.dp_method in ['GMD', 'IGMD']:
        print("Loading Human_Gaze_Predictor model...")
        try:
            from scripts.gaze_predictor import Human_Gaze_Predictor
            gaze_predictor = Human_Gaze_Predictor(args.env)
            gaze_predictor.init_model("models/gaze_predictor/seaquest_gaze_predictor_2.pth")
            gaze_predictor.model.eval()
            for param in gaze_predictor.model.parameters():
                param.requires_grad = False
            args.gaze_predictor = gaze_predictor.model
        except ImportError:
            print("Failed to load Human_Gaze_Predictor!")

    elif args.gaze_method == 'Contrastive':
        print('Creating Blurred Images:')
        from scipy.ndimage import gaussian_filter
        positive_images = []
        negative_images = []
        for img, gaze in tqdm(zip(observations, gaze_masks), total = len(observations)):
            img = img.numpy()
            gaze = gaze.numpy()
            positive_image = gaussian_filter(img, sigma=3)
            negative_image = gaussian_filter(img, sigma=3)
            
            positive_image = positive_image * (1 - gaze) + img * gaze
            negative_image = negative_image * gaze + img * (1 - gaze)
            
            positive_images.append(torch.from_numpy(positive_image))
            negative_images.append(torch.from_numpy(negative_image))
        
        positive_images = torch.stack(positive_images).unsqueeze(1)
        negative_images = torch.stack(negative_images).unsqueeze(1)
        gaze_masks = torch.cat([positive_images, negative_images], dim=1)
        if gaze_masks.max() > 1.0: gaze_masks = gaze_masks / 255.0
        
    set_seed_everywhere(args.seed)

    #shuffle and validation split
    indices = list(range(len(observations)))
    random.shuffle(indices)
    train_indices = indices[:int(len(observations) * args.val_split_ratio)]
    val_indices = indices[int(len(observations) * args.val_split_ratio):]

    observations, observations_val = observations[train_indices], observations[val_indices]
    actions, actions_val = actions[train_indices], actions[val_indices]
    gaze_masks, gaze_masks_val = gaze_masks[train_indices], gaze_masks[val_indices]
    gaze_coordinates, gaze_coordinates_val = gaze_coordinates[train_indices], gaze_coordinates[val_indices]
    if ep_nums is not None:
        ep_nums, ep_nums_val = ep_nums[train_indices], ep_nums[val_indices]
    
    is_valid_gaze = torch.ones(len(observations), dtype=torch.float32)
    
    if args.gaze_method in ['Reg', 'Contrastive', 'GRIL']:
        is_valid_gaze[int(len(observations) * args.gaze_ratio):] = 0

    print(f"Train size: {len(observations)} | Val size: {len(observations_val)}")

    ## Stage 1
    print("Building models..")
    print("Start stage 1...")

    start_fire = args.env in ['Breakout']
    env = create_atari_environment(env_name=args.env, num_stack= args.stack, render_mode=args.eval_render_mode,
                                    frame_skip=args.eval_fs, obs_size=84, noop_max=0,
                                    screen_render_width=720, screen_render_height=450, action_repeat_probability=args.sticky_probability,
                                      start_fire=start_fire)
    action_dim = int(actions.max().item()) + 1

    if args.incremental:
      
        save_tag = "incremental_seed_{}_lr_{}_nep{}_zdim{}_stack_{}_epochs_{}".format(
            args.seed,
            args.lr,
            args.num_episodes,
            args.z_dim,
            args.stack,
            args.n_epochs
        )
    elif args.independent:
        save_tag = "independent_seed_{}_lr_{}_nep{}_zdim{}_stack_{}_epochs_{}".format(
            args.seed,
            args.lr,
            args.num_episodes,
            args.z_dim,
            args.stack,
            args.n_epochs
        )
    else:
        save_tag = "seed_{}_lr_{}_nep{}_zdim{}_stack_{}_epochs_{}".format(
            args.seed,
            args.lr,
            args.num_episodes,
            args.z_dim,
            args.stack,
            args.n_epochs
        )

    if args.gaze_method in ['Reg', 'Teacher']:
        save_tag += f"_gaze_{args.gaze_method}_beta_{args.gaze_beta}_lambda_{args.gaze_lambda}_prob_dist_type_{args.prob_dist_type}"
    elif args.gaze_method in ['ViSaRL', 'Mask', 'AGIL']:
        save_tag += f"_gaze_{args.gaze_method}"
    elif args.gaze_method in ['GRIL']:
        save_tag += f"_gaze_{args.gaze_method}_lambda_{args.gaze_lambda}"
    elif args.gaze_method == 'Contrastive':
        save_tag += f"_gaze_{args.gaze_method}_threshold_{args.gaze_contrastive_threshold}_lambda_{args.gaze_lambda}"
        
    if args.dp_method in ['Oreo', 'IGMD', 'GMD']:
        save_tag += f"_dp_{args.dp_method}"
    
    if args.add_path:
        save_tag += "_" + args.add_path

    now_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = os.path.join(f"models/{args.env}/{args.gaze_method}", save_tag, now_time)
    last_save_dir = ""
    
    if not args.eval_only:
        log_dir = os.path.join(f"logs/imitation_learning_models/{args.env}/{args.gaze_method}", now_time, save_tag)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        os.makedirs(save_dir, exist_ok=True)
        # Always save args to the log folder
        args_dict = vars(args).copy()
        if 'gaze_predictor' in args_dict: del args_dict['gaze_predictor']
        if 'encoder_agil' in args_dict: del args_dict['encoder_agil']
        pd.DataFrame([args_dict]).to_csv(os.path.join(log_dir, "args.csv"), index=False)

    coeff = 2 if args.gaze_method == 'ViSaRL' else 1
    encoder = Encoder(coeff * args.stack, args.embedding_dim, args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens).to(device)

    args.encoder_agil = None
    if args.gaze_method  == 'AGIL':
        encoder_agil = Encoder(args.stack, args.embedding_dim, args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens).to(device)
        args.encoder_agil = encoder_agil
    
    encoder_output_dim = 8 * 8 * args.embedding_dim
    pre_actor = nn.Sequential( nn.Flatten(start_dim=1), nn.Linear(encoder_output_dim, args.z_dim))
    actor = nn.Sequential( nn.Linear(args.z_dim, args.z_dim), nn.ReLU(), nn.Linear(args.z_dim, action_dim),)
    pre_actor.apply(weight_init)
    pre_actor.to(device)
    actor.apply(weight_init)
    actor.to(device)
    
    if args.gaze_method == 'GRIL':
        gril_gaze_coord_predictor = nn.Sequential( nn.Linear(args.z_dim, args.z_dim), nn.ReLU(), nn.Linear(args.z_dim, 2),)
        gril_gaze_coord_predictor.apply(weight_init)
        gril_gaze_coord_predictor.to(device)
    
    if args.eval_only:
        print("Evaluation starts..")
        args.load_epoch = args.n_epochs
        encoder.load_state_dict(torch.load(args.eval_path + f"/ep{args.load_epoch}_encoder.pth"))
        pre_actor.load_state_dict(torch.load(args.eval_path + f"/ep{args.load_epoch}_pre_actor.pth"))
        actor.load_state_dict(torch.load(args.eval_path + f"/ep{args.load_epoch}_actor.pth"))
        encoder.eval()
        pre_actor.eval()
        actor.eval()
        
        if args.gaze_method == 'AGIL':
            encoder_agil.load_state_dict(torch.load(args.eval_path + f"/ep{args.load_epoch}_encoder_agil.pth"))
            encoder_agil.eval()
        
        score, std = evaluate(env, pre_actor, actor, encoder, device, args, verbose=True)
        print(f"Score: {score} | Std: {std}")
        # Save score and std here
        saveable_object = {'score': [], 'std': [], 'epoch': []}
        saveable_object['score'].append(score)
        saveable_object['std'].append(std)
        saveable_object['epoch'].append(0)
        if args.result_save_dir != "":
            save_path = os.path.join(args.result_save_dir, args.env, args.gaze_method, str(args.seed))
            os.makedirs(save_path, exist_ok=True)
            torch.save(saveable_object, os.path.join(save_path, "results.pth"))
        return

    
    params_list = list(encoder.parameters()) + list(pre_actor.parameters()) + list(actor.parameters())

    if args.gaze_method == 'AGIL':
        params_list += list(encoder_agil.parameters())
    elif args.gaze_method == 'GRIL':
        params_list += list(gril_gaze_coord_predictor.parameters())
    
    actor_optimizer = torch.optim.Adam(
        params_list,
        lr=args.lr,
        weight_decay = args.wd
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(actor_optimizer, step_size=args.step_size, gamma=0.5)

    criterion = nn.CrossEntropyLoss()
    
    if args.dp_method == 'Oreo':
        raise NotImplementedError("Oreo requires VQ-VAE. Skipping for this dataset.")
        # quantizer = VectorQuantizer(args.embedding_dim, args.num_embeddings, 0.25).to(device)
        
        # vqvae_path = f"trained_models/vqvae_models/{args.env}/seed_1_stack_{args.stack}_ep_{args.num_episodes}_train_type_{args.train_type}/model.torch"
        
        # for p in quantizer.parameters():
        #     p.requires_grad = False
        # vqvae_dict = torch.load(vqvae_path, map_location="cpu", weights_only=True)
        # encoder.load_state_dict(
        #     {k[9:]: v for k, v in vqvae_dict.items() if "_encoder" in k}
        # )
        # quantizer.load_state_dict(
        #     {k[11:]: v for k, v in vqvae_dict.items() if "_quantizer" in k}
        # )
    
        # encoder.eval()
        # quantizer.eval()
        # total_encoding_indices = []
        # with torch.no_grad():
        #     dataloader_temp = torch.utils.data.DataLoader(observations, batch_size=32, shuffle=False)
        #     for xx in tqdm (dataloader_temp, total = len(dataloader_temp)):
        #         xx = torch.as_tensor(xx, device=device, dtype=torch.float32) / 255.0
        #         z = encoder(xx)
        #         z, *_, encoding_indices, _ = quantizer(z)
        #         total_encoding_indices.append(encoding_indices.cpu())
        # total_encoding_indices = torch.cat(total_encoding_indices, dim=0)

        # del quantizer, dataloader_temp
    else:
        total_encoding_indices = torch.zeros([len(observations), encoder_output_dim // args.embedding_dim], dtype=torch.int64, device='cpu')
    

    train_data_loader = torch.utils.data.DataLoader(
        list(zip(observations, actions, gaze_masks, gaze_coordinates, is_valid_gaze, total_encoding_indices)), batch_size=args.bs, shuffle=True
    )

    val_data_loader = torch.utils.data.DataLoader(
        list(zip(observations_val, actions_val, gaze_masks_val)), batch_size=args.bs, shuffle=False
    )

    saveable_object = {'score': [], 'std': [], 'epoch': []}

    if args.incremental or args.independent:
        unique_eps = torch.sort(torch.unique(ep_nums))[0]
        ep_to_indices = {ep.item(): torch.where(ep_nums == ep)[0] for ep in unique_eps}
        num_stages = len(unique_eps)
    else:
        num_stages = 1

    global_epoch = 0
    best_loss = float('inf')
    best_score = -float('inf')
    results_log = []
    last_email_time = time.time()
    score = float('nan')
    std = float('nan')
    current_ep = 0
    current_epoch = 0
    last_loss = float('nan')
    last_reward = float('nan')

    best_reward = float('nan')
    send_run_update(args, results_log, current_ep, current_epoch, last_loss, best_loss, last_reward, best_reward, is_final=False, is_started=True)
    
    pbar_stage = tqdm(unique_eps if (args.incremental or args.independent) else [0], desc="Overall Progress")
    
    encoder, pre_actor, actor = None, None, None
    encoder_agil = None
    gril_gaze_coord_predictor = None
    actor_optimizer, scheduler = None, None

    for stage, current_ep in enumerate(pbar_stage):
        # Reset weights for Incremental mode at every stage, or initialize for the first time
        if args.incremental or stage == 0:
            if args.incremental:
                tqdm.write(f"\n--- [RESET] Initializing fresh weights for Stage {stage+1} ---")
            
            coeff = 2 if args.gaze_method == 'ViSaRL' else 1
            encoder = Encoder(coeff * args.stack, args.embedding_dim, args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens).to(device)

            args.encoder_agil = None
            if args.gaze_method  == 'AGIL':
                encoder_agil = Encoder(args.stack, args.embedding_dim, args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens).to(device)
                args.encoder_agil = encoder_agil
            
            encoder_output_dim = 8 * 8 * args.embedding_dim
            pre_actor = nn.Sequential( nn.Flatten(start_dim=1), nn.Linear(encoder_output_dim, args.z_dim))
            actor = nn.Sequential( nn.Linear(args.z_dim, args.z_dim), nn.ReLU(), nn.Linear(args.z_dim, action_dim),)
            pre_actor.apply(weight_init)
            pre_actor.to(device)
            actor.apply(weight_init)
            actor.to(device)
            
            if args.gaze_method == 'GRIL':
                gril_gaze_coord_predictor = nn.Sequential( nn.Linear(args.z_dim, args.z_dim), nn.ReLU(), nn.Linear(args.z_dim, 2),)
                gril_gaze_coord_predictor.apply(weight_init)
                gril_gaze_coord_predictor.to(device)

            params_list = list(encoder.parameters()) + list(pre_actor.parameters()) + list(actor.parameters())
            if args.gaze_method == 'AGIL':
                params_list += list(encoder_agil.parameters())
            elif args.gaze_method == 'GRIL':
                params_list += list(gril_gaze_coord_predictor.parameters())
            
            actor_optimizer = torch.optim.Adam(params_list, lr=args.lr, weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.StepLR(actor_optimizer, step_size=args.step_size, gamma=0.5)

        if args.incremental:
            # Accumulate data up to current stage
            active_episodes = unique_eps[:stage+1]
            indices = torch.cat([ep_to_indices[ep.item()] for ep in active_episodes])
            current_train_loader = torch.utils.data.DataLoader(
                list(zip(observations[indices], actions[indices], gaze_masks[indices], gaze_coordinates[indices], is_valid_gaze[indices], total_encoding_indices[indices])),
                batch_size=args.bs, shuffle=True
            )
            tqdm.write(f"\n--- Increment Step {stage+1}/{num_stages} | Data: {len(indices)} samples ---")
        elif args.independent:
            # Only current episode data
            indices = ep_to_indices[current_ep.item()]
            current_train_loader = torch.utils.data.DataLoader(
                list(zip(observations[indices], actions[indices], gaze_masks[indices], gaze_coordinates[indices], is_valid_gaze[indices], total_encoding_indices[indices])),
                batch_size=args.bs, shuffle=True
            )
            tqdm.write(f"\n--- Independent Step {stage+1}/{num_stages} | Episode: {current_ep.item()} | Data: {len(indices)} samples ---")
        else:
            current_train_loader = train_data_loader

        pbar_epoch = tqdm(range(args.n_epochs), desc=f"Stage {stage+1}", leave=False)
        for epoch in pbar_epoch:
            global_epoch += 1
            ############################################# Training
            encoder.train()
            pre_actor.train()
            actor.train()
            if args.gaze_method == 'AGIL':
                encoder_agil.train()
            elif args.gaze_method == 'GRIL':
                gril_gaze_coord_predictor.train()

            train_loss = 0
            train_acc = 0
            train_count = 0

            for i, (xx, aa, gg, gc, ivg, tei) in enumerate(current_train_loader):
                xx = torch.as_tensor(xx, device=device, dtype=torch.float32) / 255.0
                aa = torch.as_tensor(aa, device=device).long()
                gg = torch.as_tensor(gg, device=device, dtype=torch.float32)
                gc = torch.as_tensor(gc, device=device, dtype=torch.float32)
                ivg = torch.as_tensor(ivg, device=device, dtype=torch.float32)
                tei = torch.as_tensor(tei, device=device, dtype=torch.int64)
                
                actor_optimizer.zero_grad()
                
                gaze_droput_mask = None
                if args.gaze_method == 'ViSaRL':
                    xx = torch.cat([xx, gg], dim=1)
                elif args.gaze_method == 'Mask':
                    xx = xx * gg
                
                if args.dp_method == 'IGMD':
                    gaze_droput_mask = gg[:,-1:]

                z = encoder(xx, dropout_mask = gaze_droput_mask)
                reg_loss = 0

                if args.gaze_method in ['Teacher', 'Reg']:

                    g1 = gg[:,-1:,:,:][ivg > 0]
                    # abs -> mean -> softmax 2D -> upscale
                    g2 = get_gaze_mask(z, args.gaze_beta, (xx.shape[-2], xx.shape[-1]))[ivg > 0]

                    if args.prob_dist_type in ['TV', 'JS', 'KL']:
                        g1 = g1 / (g1.sum(dim = (-1, -2, -3), keepdim=True)).detach()
                        g2 = g2 / (g2.sum(dim = (-1, -2, -3), keepdim=True)).detach()
                    def KL(g1, g2):
                        return (g1 * torch.log ((g1+ 1e-7) / (g2 + 1e-7))).sum(dim = (1,2,3)).mean(0)
                    if args.prob_dist_type == 'KL':
                        reg_loss = KL(g1, g2)
                    elif args.prob_dist_type == 'TV':
                        reg_loss = (g1  - g2).abs().sum(dim = (1,2,3)).mean(0)
                    elif args.prob_dist_type == 'JS':
                        reg_loss = 1/2 * (KL(g1, (g1+g2)/2) + KL(g2, (g1+g2)/2))
                    elif args.prob_dist_type == 'MSE':
                        reg_loss = nn.functional.mse_loss(g1, g2)
                    else:
                        assert False, 'Invalid dist type'

                    reg_loss *= args.gaze_lambda

                elif args.gaze_method == 'Contrastive':
                    positive_images = gg[ivg > 0][:,:args.stack]
                    negative_images = gg[ivg > 0][:,args.stack:]
                    z_plus  = encoder(positive_images)
                    z_minus = encoder(negative_images)
                    t1 = torch.linalg.vector_norm(z[ivg > 0] - z_plus, dim=(1, 2, 3)) ** 2
                    t2 = torch.linalg.vector_norm(z[ivg > 0] - z_minus, dim=(1, 2, 3)) ** 2
                    
                    reg_loss = args.gaze_lambda * torch.max(torch.zeros_like(t1), t1 - t2 + args.gaze_contrastive_threshold).mean()
                    # print(positive_images.device, negative_images.device, z.device)

                elif args.gaze_method == 'AGIL':
                    z = (z + encoder_agil(xx * gg)) / 2

                if args.dp_method == 'GMD':
                    z = apply_gmd_dropout(z, gg[:,-1:], test_mode=False)
                elif args.dp_method == 'Oreo':
                    with torch.no_grad():
                        encoding_indices = tei  # B x 64
                        prob = torch.ones(xx.shape[0] * args.oreo_num_mask, args.num_embeddings) * (
                            1 - args.oreo_prob
                        )
                        code_mask = torch.bernoulli(prob).to(device)  # B x 512

                        ## one-hot encoding
                        encoding_indices_flatten = encoding_indices.view(-1)  # (Bx64)
                        encoding_indices_onehot = torch.zeros(
                            (len(encoding_indices_flatten), args.num_embeddings),
                            device=encoding_indices_flatten.device,
                        )
                        encoding_indices_onehot.scatter_(
                            1, encoding_indices_flatten.unsqueeze(1), 1
                        )
                        encoding_indices_onehot = encoding_indices_onehot.view(
                            xx.shape[0], -1, args.num_embeddings
                        )  # B x 64 x 512

                        mask = (
                            code_mask.unsqueeze(1)
                            * torch.cat(
                                [encoding_indices_onehot for m in range(args.oreo_num_mask)], dim=0
                            )
                        ).sum(2)
                        mask = mask.reshape(-1, 8, 8)

                    z = torch.cat([z for m in range(args.oreo_num_mask)], dim=0) * mask.unsqueeze(1)
                    z = z / (1.0 - args.oreo_prob)
                
                z = pre_actor(z)
                logits = actor(z)

                if args.dp_method  == 'Oreo':
                    aa = torch.cat([aa for m in range(args.oreo_num_mask)], dim=0)
                                
                actor_loss = criterion(logits, aa)
                
                if args.gaze_method == 'GRIL':
                    gaze_coord_pred = gril_gaze_coord_predictor(z[ivg > 0])
                    gaze_coord_loss = nn.functional.mse_loss(gaze_coord_pred, gc[ivg > 0])
                    reg_loss = args.gaze_lambda * gaze_coord_loss

                (reg_loss + actor_loss).backward()

                actor_optimizer.step()

                train_loss += actor_loss.item() * aa.size(0)
                if args.dp_method == 'Oreo':
                    aa = aa[:xx.shape[0]]
                    logits = logits[:xx.shape[0]]
                
                train_acc += (aa == logits.argmax(1)).float().sum().item()
                train_count += aa.size(0)


            learning_rate = actor_optimizer.param_groups[0]["lr"]
            scheduler.step()        
            
            writer.add_scalar("Loss/train", train_loss / train_count, global_epoch)
            writer.add_scalar("Accuracy/train", train_acc / train_count, global_epoch)
            writer.add_scalar("LR", learning_rate, global_epoch)


            ############################################# Validation
            encoder.eval()
            pre_actor.eval()
            actor.eval()

            if args.gaze_method == 'AGIL':
                encoder_agil.eval()


            val_loss = 0
            val_acc = 0
            val_count = 0

            for i, (xx, aa, gg) in enumerate(val_data_loader):
                xx = torch.as_tensor(xx, device=device, dtype=torch.float32) / 255.0
                aa = torch.as_tensor(aa, device=device).long()
                gg = torch.as_tensor(gg, device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    
                    if args.gaze_method == 'ViSaRL':
                        xx = torch.cat([xx, gg], dim=1)
                    elif args.gaze_method == 'Mask':
                        xx = xx * gg
                    
                    gaze_droput_mask = None
                    if args.dp_method == 'IGMD':
                        gaze_droput_mask = gg[:,-1:]

                    z = encoder(xx, dropout_mask = gaze_droput_mask)

                    if args.gaze_method == 'AGIL':
                        z = (z + encoder_agil(xx * gg)) / 2
                    
                    if args.dp_method == 'GMD':
                        z = apply_gmd_dropout(z, gg[:,-1:], test_mode=True)

                    z = pre_actor(z)
                    logits = actor(z)
                actor_loss = criterion(logits, aa)

                val_loss += actor_loss.item() * aa.size(0)
                val_acc += (aa == logits.argmax(1)).float().sum().item()
                val_count += aa.size(0)

            writer.add_scalar("Loss/val", val_loss / val_count, global_epoch)
            writer.add_scalar("Accuracy/val", val_acc / val_count, global_epoch)

            pbar_epoch.set_postfix({
                "T-Loss": f"{train_loss / train_count:.3f}",
                "V-Loss": f"{val_loss / val_count:.3f}",
                "T-Acc": f"{train_acc / train_count:.3f}",
                "V-Acc": f"{val_acc / val_count:.3f}"
            })

            if global_epoch % args.eval_interval == 0 and global_epoch > 0:
                encoder.eval()
                pre_actor.eval()
                actor.eval()
                if args.gaze_method == 'AGIL':
                    encoder_agil.eval()

                score, std = evaluate(env, pre_actor, actor, encoder, device, args,)
                saveable_object['score'].append(score)
                saveable_object['std'].append(std)
                saveable_object['epoch'].append(global_epoch)

                tqdm.write("(Eval) Epoch {} | Score: {:.2f} | Std: {:.2f}".format(global_epoch, score, std))
                
                if args.result_csv != "":
                    local_df = pd.DataFrame([{
                        "epoch": global_epoch,
                        "score": score,
                        "std": std,
                        "num_eval": args.num_eval_episodes,
                        "last_train_acc": train_acc / train_count,
                        "last_val_acc": val_acc / val_count,
                        "last_train_loss": train_loss / train_count,
                        "last_val_loss": val_loss / val_count,
                        "date": now_time
                    }])
                    res_df = pd.DataFrame([{
                        "run_time": now_time,
                        "epoch": global_epoch,
                        "score": score,
                        "std": std,
                        "num_eval": args.num_eval_episodes,
                        "last_train_acc": train_acc / train_count,
                        "last_val_acc": val_acc / val_count,
                        "last_train_loss": train_loss / train_count,
                        "last_val_loss": val_loss / val_count,
                        "num_episodes": stage + 1 if (args.incremental or args.independent) else args.num_episodes,
                        "incremental": args.incremental,
                        "independent": args.independent,
                        "gaze_method": args.gaze_method,
                        "learning_rate": learning_rate,
                        "method": args.gaze_method + " + " + args.dp_method + " + " + args.eval_type,
                        "env": args.env,
                        "last_save_dir": last_save_dir
                    }])
                    
                    local_df.to_csv(os.path.join(save_dir, "eval_results.csv"), mode='a', header=not os.path.exists(os.path.join(save_dir, "eval_results.csv")), index=False)
                    res_df.to_csv(args.result_csv, mode='a', header=not os.path.exists(args.result_csv), index=False)
            
            # Update best metrics
            if not np.isnan(score) and score > best_score:
                best_score = score
            if (train_loss / train_count) < best_loss:
                best_loss = train_loss / train_count
            
            # Append to results log
            results_log.append({
                'epoch': global_epoch,
                'train_loss': train_loss / train_count,
                'mean_reward': score,
                'std_reward': std
            })

            # Periodic Email Update
            if args.send_email:
                current_time = time.time()
                if (current_time - last_email_time) / 60 >= args.email_interval:
                    send_run_update(args, results_log, stage, global_epoch, train_loss/train_count, best_loss, score, best_score)
                    last_email_time = current_time
            
            writer.flush()

        if global_epoch % args.save_interval == 0 or args.incremental or args.independent:
            last_save_dir = os.path.join(save_dir, f"ep{global_epoch}")
            os.makedirs(last_save_dir, exist_ok=True)
            torch.save(
                encoder.state_dict(),
                os.path.join(save_dir, f"ep{global_epoch}", "ep{}_encoder.pth".format(global_epoch)),
            )
            torch.save(
                actor.state_dict(),
                os.path.join(save_dir, f"ep{global_epoch}", "ep{}_actor.pth".format(global_epoch),),
            )
            torch.save(
                pre_actor.state_dict(),
                os.path.join(save_dir, f"ep{global_epoch}", "ep{}_pre_actor.pth".format(global_epoch),),
            )

            if args.gaze_method == 'GRIL':
                torch.save(
                    gril_gaze_coord_predictor.state_dict(),
                    os.path.join(save_dir, f"ep{global_epoch}", "ep{}_gril_gaze_coord_predictor.pth".format(global_epoch),),
                )

            if args.gaze_method == 'AGIL':
                torch.save(
                    encoder_agil.state_dict(),
                    os.path.join(save_dir, f"ep{global_epoch}", "ep{}_encoder_agil.pth".format(global_epoch)),
                )


    if args.result_save_dir != "":
        torch.save(saveable_object, args.result_save_dir)
    
    if args.gaze_predictor:
        args.gaze_predictor = None
    
    if args.encoder_agil:
        args.encoder_agil = None
    
    torch.cuda.empty_cache()
    
    # Final Email Update
    if args.send_email:
        last_loss = results_log[-1]['train_loss'] if results_log else 0.0
        send_run_update(args, results_log, stage, global_epoch, last_loss, best_loss, score, best_score, is_final=True)
    

if __name__ == "__main__":
    args = get_args()
    if args.send_email:
        try:
            
            train(args, True)
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            subject = f"CRASH: {args.env} - BC Training - {args.gaze_method}"
            body = f"The training script crashed with the following error:\n\n{error_msg}"
            send_email(subject, body)
            raise e
    else:
        train(args, True)
