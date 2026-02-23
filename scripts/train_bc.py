import torch
import torch.nn as nn
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
from models.linear_models import Encoder, weight_init, VectorQuantizer, Decoder, AutoEncoder
from utils import load_dataset, evaluate, set_seed_everywhere
from atari_env_manager import create_env as create_atari_environment
from gaze.gaze_utils import get_gaze_mask, apply_gmd_dropout
from torch.utils.tensorboard import SummaryWriter
import datetime   


def get_args(run_in_notebook=False):
    import argparse
    parser = argparse.ArgumentParser()

    # Seed & Env
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--env", default="Phoenix", type=str)
    parser.add_argument("--datapath", default="dataset", type=str)
    parser.add_argument("--stack", default=1, type=int)
    parser.add_argument("--train_type", default='normal', choices=['confounded', 'normal'], type=str)
    parser.add_argument("--eval_type", default='normal', choices=['confounded', 'normal'], type=str)
    parser.add_argument("--eval_fs", default=4, type=int, help='Frame skip for evaluation')
    parser.add_argument("--randomness", default=0.0, type=float)
    parser.add_argument("--sticky_probability", default=0.25, type=float)

    # Save & Evaluation
    parser.add_argument("--save_interval", default=500, type=int)
    parser.add_argument("--eval_interval", default=100, type=int)
    parser.add_argument("--num_episodes", default=20, type=int)
    parser.add_argument("--num_eval_episodes", default=100, type=int)
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
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--step_size", default=100, type=int)
    parser.add_argument("--wd", default=0, type=float)

    # For MLP
    parser.add_argument("--z_dim", default=256, type=int)



    parser.add_argument("--eval_render_mode", default='rgb_array', choices=['rgb_array', 'human'], type=str)
    parser.add_argument("--eval_record", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)

    # Update the path if you are just evaluating a model already trained
    parser.add_argument("--eval_path", type=str, default="trained_models/imitation_learning_models/")

    parser.add_argument("--gaze_method", type=str, default='Teacher', choices=['None', 'Teacher', 'Reg', 'Mask', 'Contrastive', 'ViSaRL', 'AGIL', 'GRIL'])
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

    observations, actions, gaze_masks, gaze_coordinates = load_dataset(
        args.env,
        args.seed,
        args.datapath,
        args.train_type,
        args.randomness,
        args.stack,
        args.num_episodes,
        (args.gaze_method in ['Reg', 'Contrastive', 'GRIL']) and (not args.eval_only),
        args.dataset_source,
        gaze_mask_sigma=args.gaze_mask_sigma,
        gaze_mask_coef=args.gaze_mask_coeff                
    )
    
    args.gaze_predictor = None
    # Methods that need gaze predictor
    if args.gaze_method in ['ViSaRL', 'Mask', 'Teacher', 'AGIL'] or args.dp_method in ['GMD', 'IGMD']:
        print("Loading gaze predictor model")
                
        encoder_gp = Encoder(args.stack, args.embedding_dim, args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens,)
        decoder_gp = Decoder(args.stack, args.embedding_dim, args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens,)
        gaze_predictor = AutoEncoder(encoder_gp, decoder_gp).to(device)
        
        load_episodes = int(args.num_episodes * args.gaze_ratio)
        gaze_predictor_path = f'trained_models/gaze_predictor_models/{args.env}/seed_1_stack_{args.stack}_ep_{load_episodes}_train_type_{args.train_type}/model.torch'
        gaze_predictor.load_state_dict(torch.load(gaze_predictor_path, weights_only=True))
        gaze_predictor.eval()
        
        for param in gaze_predictor.parameters():
            param.requires_grad = False
        
        args.gaze_predictor = gaze_predictor
        
        if not (args.gaze_method in ['Reg', 'Contrastive', 'GRIL']):
            print("Predicting gaze masks ...")
            gaze_masks = []
            dataloader_temp = torch.utils.data.DataLoader(observations, batch_size=32, shuffle=False)
            for xx in tqdm (dataloader_temp, total = len(dataloader_temp)):
                xx = torch.as_tensor(xx, device=device, dtype=torch.float32) / 255.0
                gaze_pred = gaze_predictor(xx)
                gaze_masks.append(gaze_pred.cpu())
            
            gaze_masks = torch.cat(gaze_masks, dim=0)
            gaze_masks[gaze_masks < 0] = 0
            gaze_masks[gaze_masks > 1] = 1

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
        
        positive_images = torch.stack(positive_images)
        negative_images = torch.stack(negative_images)
        gaze_masks = torch.cat([positive_images, negative_images], dim=1)
        gaze_masks = gaze_masks / 255.0
        
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
    action_dim = env.action_space.n

    
    save_tag = "seed{}_{}_randomness{}_nep{}".format(
        args.seed,
        args.train_type,
        args.randomness,
        args.num_episodes,
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

    now_time = datetime.datetime.now()
    save_dir = "{}_{}".format(now_time.strftime("%Y_%m_%d_%H_%M_%S"), save_tag)
    
    if not args.eval_only:
        writer = SummaryWriter(f"logs/imitation_learning_models/{args.env}/" + save_dir)
        save_dir = os.path.join(f"trained_models/imitation_learning_models/{args.env}", save_dir)
        os.makedirs(save_dir, exist_ok=True)

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
        
        score, std = evaluate(env, pre_actor, actor, encoder, device, args, verbose=True,)
        print(f"Score: {score} | Std: {std}")
        # Save score and std here
        saveable_object = {'score': [], 'std': [], 'epoch': []}
        saveable_object['score'].append(score)
        saveable_object['std'].append(std)
        saveable_object['epoch'].append(0)
        if args.result_save_dir != "":
            torch.save(saveable_object, args.result_save_dir)
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
        quantizer = VectorQuantizer(args.embedding_dim, args.num_embeddings, 0.25).to(device)
        
        vqvae_path = f"trained_models/vqvae_models/{args.env}/seed_1_stack_{args.stack}_ep_{args.num_episodes}_train_type_{args.train_type}/model.torch"
        
        for p in quantizer.parameters():
            p.requires_grad = False
        vqvae_dict = torch.load(vqvae_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(
            {k[9:]: v for k, v in vqvae_dict.items() if "_encoder" in k}
        )
        quantizer.load_state_dict(
            {k[11:]: v for k, v in vqvae_dict.items() if "_quantizer" in k}
        )
    
        encoder.eval()
        quantizer.eval()
        total_encoding_indices = []
        with torch.no_grad():
            dataloader_temp = torch.utils.data.DataLoader(observations, batch_size=32, shuffle=False)
            for xx in tqdm (dataloader_temp, total = len(dataloader_temp)):
                xx = torch.as_tensor(xx, device=device, dtype=torch.float32) / 255.0
                z = encoder(xx)
                z, *_, encoding_indices, _ = quantizer(z)
                total_encoding_indices.append(encoding_indices.cpu())
        total_encoding_indices = torch.cat(total_encoding_indices, dim=0)

        del quantizer, dataloader_temp
    else:
        total_encoding_indices = torch.zeros([len(observations), encoder_output_dim // args.embedding_dim], dtype=torch.int64, device='cpu')
    

    train_data_loader = torch.utils.data.DataLoader(
        list(zip(observations, actions, gaze_masks, gaze_coordinates, is_valid_gaze, total_encoding_indices)), batch_size=args.bs, shuffle=True
    )

    val_data_loader = torch.utils.data.DataLoader(
        list(zip(observations_val, actions_val, gaze_masks_val)), batch_size=args.bs, shuffle=False
    )

    saveable_object = {'score': [], 'std': [], 'epoch': []}

    for epoch in tqdm(range(args.n_epochs)):

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

        for i, (xx, aa, gg, gc, ivg, tei) in enumerate(train_data_loader):
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
        
        writer.add_scalar("Loss/train", train_loss / train_count, epoch)
        writer.add_scalar("Accuracy/train", train_acc / train_count, epoch)
        writer.add_scalar("LR", learning_rate, epoch)


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

        writer.add_scalar("Loss/val", val_loss / val_count, epoch)
        writer.add_scalar("Accuracy/val", val_acc / val_count, epoch)

        if verbose:
            print("(Train) Epoch {} | LR {:.6f} | Train Loss: {:.4f} | Val Loss: {:.4f} | Train Acc: {:.4f} | Val Acc: {:.4f}".format(epoch + 1, learning_rate, train_loss / train_count, val_loss / val_count, train_acc / train_count, val_acc / val_count, ))

        if (epoch + 1) % args.eval_interval == 0:
            encoder.eval()
            pre_actor.eval()
            actor.eval()
            if args.gaze_method == 'AGIL':
                encoder_agil.eval()

            score, std = evaluate(env, pre_actor, actor, encoder, device, args,)

            saveable_object['score'].append(score)
            saveable_object['std'].append(std)
            saveable_object['epoch'].append(epoch)

            writer.add_scalar("Score/eval", score, epoch)


            print("(Eval) Epoch {} | Score: {:.2f} | Std: {:.2f}".format(epoch + 1, score, std))

        writer.flush()

        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                encoder.state_dict(),
                os.path.join(save_dir, "ep{}_encoder.pth".format(epoch + 1)),
            )
            torch.save(
                actor.state_dict(),
                os.path.join(save_dir, "ep{}_actor.pth".format(epoch + 1),),
            )
            torch.save(
                pre_actor.state_dict(),
                os.path.join(save_dir, "ep{}_pre_actor.pth".format(epoch + 1),),
            )

            if args.gaze_method == 'GRIL':
                torch.save(
                    gril_gaze_coord_predictor.state_dict(),
                    os.path.join(save_dir, "ep{}_gril_gaze_coord_predictor.pth".format(epoch + 1),),
                )

            if args.gaze_method == 'AGIL':
                torch.save(
                    encoder_agil.state_dict(),
                    os.path.join(save_dir, "ep{}_encoder_agil.pth".format(epoch + 1)),
                )


    if args.result_save_dir != "":
        torch.save(saveable_object, args.result_save_dir)
    
    if args.gaze_predictor:
        args.gaze_predictor = None
    
    if args.encoder_agil:
        args.encoder_agil = None
    
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    args = get_args()
    train(args, True)