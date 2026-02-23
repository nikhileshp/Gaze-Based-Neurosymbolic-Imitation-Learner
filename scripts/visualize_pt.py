import argparse
import numpy as np
import cv2
import torch

def main():
    parser = argparse.ArgumentParser(description="Visualize the converted .pt dataset.")
    parser.add_argument('--dataset', type=str, help='Path to the .pt dataset file')
    parser.add_argument('--start', type=int, default=0, help='Start frame index')
    parser.add_argument('--fps', type=int, default=30, help='Playback speed (FPS)')
    parser.add_argument('--video', type=str, default=None, help='Path to save output video (e.g. output.mp4)')
    parser.add_argument('--split_episodes', action='store_true', help='If --video is set, saves one video per episode (e.g. output_ep_1.mp4)')
    parser.add_argument('--episode', type=str, default=None, help='Visualize only a specific episode or range (e.g., "5" or "2-4")')
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

    # Map object types to colors (BGR format)
    # type 5 = EnemyMissile/PlayerMissile (both mapped to 5 in convert script)
    # We split them by orientation: enemies tend to y-increasing (orientation > 0)
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
        while i < N and ep_nums[i] not in valid_eps:
            i += 1

    # Setup Initial Video Writer
    video_writer = None
    current_video_ep = None
    SCALE_IMG = 4
    display_size = (84 * SCALE_IMG, 84 * SCALE_IMG)
    out_w = display_size[0] * 2
    out_h = display_size[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def open_video_writer(ep=None):
        if not args.video: return None
        path = args.video
        if args.split_episodes and ep is not None:
            # Insert _ep_{num} before the extension
            parts = path.rsplit('.', 1)
            path = f"{parts[0]}_ep_{ep}.{parts[1]}" if len(parts) == 2 else f"{path}_ep_{ep}"
        print(f"Exporting video to {path} at {args.fps} FPS...")
        return cv2.VideoWriter(path, fourcc, args.fps, (out_w, out_h))

    if args.video and not args.split_episodes:
        video_writer = open_video_writer()
        
    # Use tqdm for progress if we are compiling a video
    from tqdm import tqdm
    frames_to_process = N - i
    if valid_eps is not None:
        frames_to_process = sum(1 for e in ep_nums[i:] if e in valid_eps)
    pbar = tqdm(total=frames_to_process, desc="Rendering Frames") if args.video else None

    while i < N:
        frame_ep = ep_nums[i]
        
        # Skip if out of requested episode range
        if valid_eps is not None and frame_ep not in valid_eps:
            i += 1
            if frame_ep > max(valid_eps):
                break # Reached the end of the requested range
            continue
            
        # Handle split_episodes writer rotation
        if args.video and args.split_episodes and frame_ep != current_video_ep:
            if video_writer is not None:
                video_writer.release()
            video_writer = open_video_writer(frame_ep)
            current_video_ep = frame_ep

        frame_obs = obs[i]
        frame_gaze = gaze_img[i]
        frame_logic = logic[i]
        frame_rew = float(rewards_arr[i]) if rewards_arr is not None else 0.0
        
        # Track cumulative reward per episode
        if not hasattr(main, '_ep_rewards'):
            main._ep_rewards = {}
        ep_key = int(ep_nums[i])
        main._ep_rewards[ep_key] = main._ep_rewards.get(ep_key, 0.0) + frame_rew

        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 8: 0}

        # Render images (scale up)
        obs_bgr = cv2.cvtColor(frame_obs, cv2.COLOR_GRAY2BGR)
        obs_bgr = cv2.resize(obs_bgr, display_size, interpolation=cv2.INTER_NEAREST)
        gaze_color = cv2.applyColorMap((frame_gaze * 255).astype(np.uint8), cv2.COLORMAP_JET)
        gaze_scaled = cv2.resize(gaze_color, display_size, interpolation=cv2.INTER_NEAREST)
        display_gaze = cv2.addWeighted(obs_bgr, 0.4, gaze_scaled, 0.6, 0)
        display_left = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)

        # Draw logic state objects on right pane
        player_slots = set()  # Indices that belong to the Player category
        # First pass: identify Player object row offsets
        for idx, obj in enumerate(frame_logic):
            present, cx, cy, w, h, orient, type_id = obj
            if present and type_id == 3:
                player_slots.add(idx)

        for idx, obj in enumerate(frame_logic):
            # 7 features: present, x, y, width, height, orientation, type_id
            present, cx, cy, w, h, orient, type_id = obj
            
            if present:
                # Differentiate PlayerMissile (type 5 near player) from EnemyMissile:
                # In convert script both are stored as type 5. Orientation==0 -> player-fired
                draw_type = type_id
                if type_id == 5 and orient != 0:
                    draw_type = 8  # Reclassify as PlayerMissile (cyan)
                if type_id == 5 and orient == 0:
                    draw_type = 5  # Reclassify as EnemyMissile (magenta)
                if draw_type in counts:
                    counts[draw_type] += 1
                elif type_id in counts:
                    counts[type_id] += 1
                
                # Scale coordinates to display size
                scaled_x = int((cx / 160.0) * 84 * SCALE_IMG)
                scaled_y = int((cy / 210.0) * 84 * SCALE_IMG)
                scaled_w = int((w / 160.0) * 84 * SCALE_IMG)
                scaled_h = int((h / 210.0) * 84 * SCALE_IMG)
                
                color = colors[draw_type] if 0 <= draw_type < len(colors) else (255, 255, 255)
                
                # Draw on the heatmap overlaid side
                cv2.circle(display_gaze, (scaled_x, scaled_y), 3, color, -1)
                
                # Draw actual bounding box based on width and height
                if type_id == 2: # OxygenBar is mapped top-left
                    box_x1 = scaled_x
                    box_y1 = scaled_y
                    box_x2 = scaled_x + scaled_w
                    box_y2 = scaled_y + scaled_h
                else: # All other objects are mapped by their center
                    box_x1 = int(scaled_x - (scaled_w / 2))
                    box_y1 = int(scaled_y - (scaled_h / 2))
                    box_x2 = int(scaled_x + (scaled_w / 2))
                    box_y2 = int(scaled_y + (scaled_h / 2))
                
                cv2.rectangle(display_gaze, (box_x1, box_y1), (box_x2, box_y2), color, 2)

        # Draw Object Counts on left panel
        FONT = cv2.FONT_HERSHEY_DUPLEX
        FS = 0.45  # font scale
        FT = 1     # font thickness
        names = [
            (0, 'Enemies (Shark/Sub)'),
            (1, 'Divers'),
            (2, 'Oxygen'),
            (3, 'Player'),
            (5, 'Enemy Missiles'),
            (8, 'Player Missiles'),
            (6, 'Coll. Divers'),
        ]
        cv2.putText(display_left, "Object Counts:", (10, 20), FONT, FS, (255, 255, 255), FT)
        y_offset = 42
        for t_id, name in names:
            color = colors[t_id] if 0 <= t_id < len(colors) else (200, 200, 200)
            cv2.putText(display_left, f"{name}: {counts.get(t_id, 0)}", (10, y_offset), FONT, FS, color, FT)
            y_offset += 26

        # Concatenate side by side: [Left Data Pane | Gaze Overlay]
        combined = np.hstack((display_left, display_gaze))
        
        # Add HUD text at bottom of combined frame
        cum_rew = main._ep_rewards.get(ep_key, 0.0)
        h_img = combined.shape[0]
        cv2.putText(combined, f"Frame {i}/{N}", (10, h_img - 40), FONT, 0.45, (220, 220, 220), 1)
        cv2.putText(combined, f"Ep: {ep_key}  CumRew: {cum_rew:.1f}", (10, h_img - 18), FONT, 0.45, (220, 220, 220), 1)
        if paused:
            cv2.putText(combined, "PAUSED — n/b to step", (display_size[0] + 10, 20), FONT, 0.4, (0, 120, 255), 1)

        if video_writer:
            video_writer.write(combined)
            pbar.update(1)
            i += 1
            continue

        cv2.imshow("Dataset Visualization: Grayscale + Logic State vs Gaze Heatmap", combined)

        wait_time = 0 if paused else delay
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q') or key == 27: # q or esc
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('n') and paused:
            i += 1
            if i >= N: i = N - 1
            continue
        elif key == ord('b') and paused:
            i = max(0, i - 1)
            continue
            
        if not paused:
            i += 1

    if video_writer:
        video_writer.release()
        pbar.close()
        print(f"Video saved to {args.video}")
    else:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
