import argparse
import numpy as np
import cv2
import torch
import sys
import os

# Add parent directory to sys.path to allow imports from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.gaze_predictor import Human_Gaze_Predictor

def main():
    parser = argparse.ArgumentParser(description="Visualize the converted .pt dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the .pt dataset file')
    parser.add_argument('--predicted', type=str, default=None, help='Path to predicted gaze model weights use models/gaze_predictor/seaquest_gaze_predictor.pth')
    parser.add_argument('--start', type=int, default=0, help='Start frame index')
    parser.add_argument('--fps', type=int, default=30, help='Playback speed (FPS)')
    parser.add_argument('--video', type=str, default=None, help='Path to save output video (e.g. output.mp4)')
    parser.add_argument('--split_episodes', action='store_true', help='If --video is set, saves one video per episode (e.g. output_ep_1.mp4)')
    parser.add_argument('--episode', type=str, default=None, help='Visualize only a specific episode or range (e.g., "5" or "2-4")')
    parser.add_argument('--multiplier', type=float, default=2.0, help='Multiplier for gaze heatmap intensity overlay')
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    dataset = torch.load(args.dataset, weights_only=False)
    obs = dataset['observations']
    gaze_img = dataset['gaze_image']
    logic = dataset['logic_state']
    ep_nums = dataset['episode_number']
    rewards_arr = dataset.get('episode-rewards', None)

    N = len(obs)
    print(f"Loaded {N} frames. Controls: [p]=Pause, [n]=Next frame, [b]=Prev frame, [q]=Quit")

    # Load Predicted Gaze Model
    gp = None
    if args.predicted and os.path.exists(args.predicted):
        print(f"Loading Predicted Gaze Model from {args.predicted}...")
        gp = Human_Gaze_Predictor("seaquest")
        gp.init_model(args.predicted)
        gp.model.eval()
    else:
        print(f"Warning: Predicted gaze model not found at {args.predicted}")

    # Map object types to colors (BGR format)
    colors = [
        (0, 200, 0),       # 0: Shark / Submarine (Green)
        (0, 80, 255),      # 1: Diver (Red-Orange)
        (200, 200, 0),     # 2: OxygenBar (Yellow)
        (255, 80, 0),      # 3: Player (Blue-Orange)
        (0, 255, 255),     # 4: (unused)
        (255, 0, 200),     # 5: EnemyMissile (Magenta/Pink)
        (128, 128, 128),   # 6: CollectedDiver (Gray)
        (128, 0, 128),     # 7: Surface (Purple)
        (0, 200, 255),     # 8: PlayerMissile (Cyan-Yellow)
    ]

    paused = False
    delay = int(1000 / args.fps)
    
    i = args.start
    
    # Filter by episode range
    valid_eps = None
    if args.episode is not None:
        if '-' in args.episode:
            start_ep, end_ep = map(int, args.episode.split('-'))
            valid_eps = set(range(start_ep, end_ep + 1))
        else:
            valid_eps = {int(args.episode)}
            
        # Fast-forward i to first valid frame
        while i < N and int(ep_nums[i]) not in valid_eps:
            i += 1

    # Setup Initial Video Writer
    SCALE_IMG = 4
    display_size = (84 * SCALE_IMG, 84 * SCALE_IMG)
    # Side panel | GT | Predicted (optional)
    out_w = display_size[0] * (3 if gp else 2)
    out_h = display_size[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def open_video_writer(ep=None):
        if not args.video: return None
        path = args.video
        # Ensure extension exists
        if '.' not in os.path.basename(path):
            path += '.mp4'
            
        if args.split_episodes and ep is not None:
            parts = path.rsplit('.', 1)
            path = f"{parts[0]}_ep_{ep}.{parts[1]}"
        print(f"Exporting video to {path} at {args.fps} FPS...")
        return cv2.VideoWriter(path, fourcc, args.fps, (out_w, out_h))

    video_writer = open_video_writer() if args.video and not args.split_episodes else None
    current_video_ep = None
        
    from tqdm import tqdm
    frames_to_process = N - i
    if valid_eps is not None:
        frames_to_process = sum(1 for e in ep_nums[i:] if int(e) in valid_eps)
    pbar = tqdm(total=frames_to_process, desc="Rendering Frames") if args.video else None

    # Frame stacking buffer (4 frames)
    k = 4
    obs_f32 = obs.astype(np.float32) / 255.0

    while i < N:
        frame_ep = int(ep_nums[i])
        
        if valid_eps is not None and frame_ep not in valid_eps:
            i += 1
            if frame_ep > max(valid_eps): break
            continue
            
        if args.video and args.split_episodes and frame_ep != current_video_ep:
            if video_writer is not None: video_writer.release()
            video_writer = open_video_writer(frame_ep)
            current_video_ep = frame_ep

        frame_obs = obs[i]
        frame_gaze = gaze_img[i]
        frame_logic = logic[i]
        frame_rew = float(rewards_arr[i]) if rewards_arr is not None else 0.0
        
        if not hasattr(main, '_ep_rewards'): main._ep_rewards = {}
        ep_key = frame_ep
        main._ep_rewards[ep_key] = main._ep_rewards.get(ep_key, 0.0) + frame_rew

        # Prepare images
        obs_bgr = cv2.cvtColor(frame_obs, cv2.COLOR_GRAY2BGR)
        obs_bgr_scaled = cv2.resize(obs_bgr, display_size, interpolation=cv2.INTER_NEAREST)
        
        # Ground Truth Heatmap
        gt_color = cv2.applyColorMap(np.clip(frame_gaze * 255 * args.multiplier, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        gt_scaled = cv2.resize(gt_color, display_size, interpolation=cv2.INTER_NEAREST)
        display_gt = cv2.addWeighted(obs_bgr_scaled, 0.4, gt_scaled, 0.6, 0)

        # Predicted Heatmap
        display_pred = obs_bgr_scaled.copy()
        if gp is not None and i >= k - 1:
            stack = np.zeros((1, 84, 84, k), dtype=np.float32)
            for j in range(k):
                stack[0, :, :, j] = obs_f32[i - (k - 1 - j)]
            
            stack_tensor = torch.from_numpy(stack).permute(0, 3, 1, 2).to(gp.device)
            with torch.no_grad():
                pred_out = gp.model(stack_tensor).cpu().numpy()[0, 0]
            
            mx = pred_out.max()
            if mx > 0: pred_out = pred_out / mx
            
            pred_color = cv2.applyColorMap(np.clip(pred_out * 255 * args.multiplier, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            pred_scaled = cv2.resize(pred_color, display_size, interpolation=cv2.INTER_NEAREST)
            display_pred = cv2.addWeighted(obs_bgr_scaled, 0.4, pred_scaled, 0.6, 0)

        # Draw sidebar
        display_side = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 6: 0, 8: 0}
        
        for obj in frame_logic:
            present, cx, cy, w, h, orient, type_id = obj
            if present:
                draw_type = int(type_id)
                if draw_type == 5:
                    draw_type = 8 if orient != 0 else 5
                if draw_type in counts: counts[draw_type] += 1
                
                # Draw small circles/boxes on gaze views
                sx = int((cx / 160.0) * display_size[0])
                sy = int((cy / 210.0) * display_size[1])
                sw = int((w / 160.0) * display_size[0])
                sh = int((h / 210.0) * display_size[1])
                color = colors[draw_type] if 0 <= draw_type < len(colors) else (255, 255, 255)
                
                cv2.circle(display_gt, (sx, sy), 3, color, -1)
                if gp:
                    cv2.circle(display_pred, (sx, sy), 3, color, -1)
                
                if draw_type == 2: b1, b2 = (sx, sy), (sx+sw, sy+sh)
                else: b1, b2 = (int(sx-sw/2), int(sy-sh/2)), (int(sx+sw/2), int(sy+sh/2))
                cv2.rectangle(display_gt, b1, b2, color, 1)
                if gp:
                    cv2.rectangle(display_pred, b1, b2, color, 1)

        # Text info
        FONT = cv2.FONT_HERSHEY_DUPLEX
        names = [(0, 'Enemies'), (1, 'Divers'), (2, 'Oxygen'), (3, 'Player'), (5, 'Enemy Missiles'), (8, 'Player Missiles'), (6, 'Coll. Divers')]
        cv2.putText(display_side, f"Frame: {i}/{N}", (15, 30), FONT, 0.5, (255, 255, 255), 1)
        cv2.putText(display_side, f"Episode: {ep_key}", (15, 55), FONT, 0.5, (255, 255, 255), 1)
        cv2.putText(display_side, f"Reward: {main._ep_rewards[ep_key]:.1f}", (15, 80), FONT, 0.5, (255, 255, 255), 1)
        
        y_off = 130
        cv2.putText(display_side, "Object Counts:", (15, y_off), FONT, 0.5, (200, 200, 200), 1)
        for t_id, name in names:
            y_off += 25
            cv2.putText(display_side, f"{name}: {counts.get(t_id, 0)}", (20, y_off), FONT, 0.45, colors[t_id], 1)

        # Labels for views
        cv2.putText(display_gt, "GROUND TRUTH", (10, 25), FONT, 0.6, (255, 255, 255), 2)
        if gp:
            cv2.putText(display_pred, "PREDICTED", (10, 25), FONT, 0.6, (255, 255, 255), 2)
            combined = np.hstack((display_side, display_gt, display_pred))
        else:
            combined = np.hstack((display_side, display_gt))
        
        if video_writer:
            video_writer.write(combined)
            pbar.update(1)
            i += 1
            continue

        cv2.imshow("Gaze Comparison: Side | GT | Pred", combined)
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused
        elif key == ord('n') and paused: i = min(N - 1, i + 1)
        elif key == ord('b') and paused: i = max(0, i - 1)
        elif not paused: i += 1

    if video_writer:
        video_writer.release()
        pbar.close()
        print(f"Video saved to {args.video}")
    else:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
