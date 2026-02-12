import os
import glob
import re
import argparse
import sys
import numpy as np
import torch
import pygame
import cv2
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Also add nsfr package path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nsfr'))

from nsfr.utils.common import load_module
from nudge.agents.neural_agent import ActorCritic
from ocatari.vision.seaquest import _detect_objects
from ocatari.vision.game_objects import GameObject, NoObject
from ocatari.ram.seaquest import MAX_NB_OBJECTS, _init_objects_ram 

# Ensure Seaquest-specific classes are available if needed
# Actually, _detect_objects instantiates them directly from ocatari.vision.seaquest
# We need to make sure we have the list initialized with enough slots.
# MAX_NB_OBJECTS is a dict {type: count}.


# Define constants
SCREEN_WIDTH = 160 * 4
SCREEN_HEIGHT = 210 * 4
IMAGE_SIZE = (160, 210)
PREDICATE_PROBS_COL_WIDTH = 300
CELL_BACKGROUND_DEFAULT = np.array([40, 40, 40])

class TrajectoryVisualizer:
    def __init__(self, data_path: str, agent_path: str, start_frame: int = 0):
        self.data_path = data_path
        self.agent_path = agent_path
        self.current_frame_idx = start_frame
        
        # Load agent
        self.agent = self._load_agent()
        
        # Load images and data
        self.images = self._load_images()
        self.trajectory_data = self._load_trajectory_data()
        
        # Initialize Ocatari objects
        self.objects = self._init_ocatari_objects()
        
        # Pygame setup
        pygame.init()
        pygame.display.set_caption("Trajectory Visualization")
        self.window = pygame.display.set_mode((SCREEN_WIDTH + PREDICATE_PROBS_COL_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Calibri', 24)
        
        # Playback control
        self.paused = False # Auto-play for debugging
        self.running = True

    def _load_agent(self):
        print(f"Loading agent from {self.agent_path}...")
        
        # We assume defaults: env=seaquest, rules=default
        device = "cpu"
        
        try:
            from nudge.agents.imitation_agent import ImitationAgent
            
            # Init agent
            agent = ImitationAgent("seaquest", "default", device)
            agent.load(self.agent_path)
            
            # We also need the env wrapper for state extraction logic
            from nudge.env import NudgeBaseEnv
            self.env_wrapper = NudgeBaseEnv.from_name("seaquest", mode="logic")
            
            return agent
            
        except Exception as e:
            print(f"Failed to load agent via ImitationAgent: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _load_images(self) -> List[str]:
        # Find all .png files in the data path
        # The data path is the directory containing images
        images = sorted(glob.glob(os.path.join(self.data_path, "*.png")), 
                        key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        print(f"Found {len(images)} frames.")
        return images

    def _load_trajectory_data(self) -> Dict[str, Any]:
        # Try to find the .txt file matching the folder name in the parent directory
        folder_name = os.path.basename(os.path.normpath(self.data_path))
        parent_dir = os.path.dirname(os.path.normpath(self.data_path))
        
        potential_txt = os.path.join(parent_dir, folder_name + ".txt")
        target_txt = None
        
        if os.path.exists(potential_txt):
            target_txt = potential_txt
        else:
            # Fallback: look inside the folder
            txt_files = glob.glob(os.path.join(self.data_path, "*.txt"))
            # Filter out relationships_output.txt if possible unless it's the only one and matches format
            candidates = [f for f in txt_files if "relationships" not in f]
            if candidates:
                target_txt = candidates[0]
            elif txt_files:
                 # If only relationships file exists, maybe check if it works? 
                 # But previous run failed on it.
                 print(f"Only found {txt_files[0]} but it might be wrong format. Skipping.")
                 pass
        
        if not target_txt:
            print(f"No trajectory .txt file found for {folder_name}.")
            return {}
                
        print(f"Loading data from {target_txt}")
        
        data_map = {}
        with open(target_txt, 'r') as f:
            header = f.readline().strip().split(',')
            # Expecting: frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions...
            
            for line in f:
                parts = line.strip().split(',')
                frame_id = parts[0]
                try:
                    action = int(parts[5])
                except ValueError:
                    action = 0 # Default to NOOP if null or invalid
                # Handle 'null' or empty strings
                gaze_data = []
                for x in parts[6:]:
                    x = x.strip()
                    if x and x.lower() != 'null':
                        try:
                            gaze_data.append(float(x))
                        except ValueError:
                            pass # Skip invalid floats
                            
                gaze_points = []
                for i in range(0, len(gaze_data), 2):
                    if i+1 < len(gaze_data):
                        gaze_points.append((gaze_data[i], gaze_data[i+1]))
                
                data_map[frame_id + ".png"] = { # Matching image filename convention
                    "action": action,
                    "gaze": gaze_points
                }
        return data_map

    def _init_ocatari_objects(self):
        # Initialize objects list based on MAX_NB_OBJECTS
        # Seaquest MAX_NB_OBJECTS in ocatari currently:
        # { Player: 1, Shark: 12, Submarine: 12, Diver: 4, EnemyMissile: 4, PlayerMissile: 1, OxygenBar: 1, CollectedDiver: 6, PlayerScore: 1, Lives: 1, OxygenBarDepleted: 1 }
        # And some logos sometimes.
        # NudgeEnv.env.py has logic:
        # 34:         if 'EnemyMissile' in MAX_ESSENTIAL_OBJECTS:
        # 35:              MAX_ESSENTIAL_OBJECTS['EnemyMissile'] = 8
        # We should replicate the size calculation to be safe.
        
        # We must initialize the objects list with the correct types expected by Ocatari
        # _detect_objects (vision) expects precise indices for Player etc. and updates them in-place.
        # If we just use NoSet, they stay NoSet and are filtered out.
        # So we use the RAM init function to get the structure.
        return _init_objects_ram(hud=True)

    def extract_logic_state(self, objects):
        # We need to convert Ocatari objects to the logic tensor format expected by the agent.
        # Referencing `in/envs/seaquest/env.py`: extract_logic_state
        
        # We need to map the Ocatari objects (which are now updated in `self.objects`) 
        # to the `raw_state` list expected by `NudgeEnv.extract_logic_state`.
        # `NudgeEnv.extract_logic_state` iterates over `raw_state` (list of objects) and checks `obj.category`.
        
        # Ocatari Vision objects have `category` attribute derived from class name usually.
        # e.g. wih `class Player(GameObject):` -> category is "Player" (or "player"? need to check)
        # GameObject lowercases class name for category by default? Or we check `category` prop.
        
        return self.env_wrapper.extract_logic_state(objects)

    def process_frame(self, frame_path):
        # 1. Load Image
        image = cv2.imread(frame_path)
        if image is None:
            return None, None
        
        # RGB for Ocatari/Pygame
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Extract Objects
        # _detect_objects(objects, obs, hud=False)
        _detect_objects(self.objects, image_rgb, hud=True)
        
        # 3. Get Logic State & Neural State 
        # We filter out NoObject for the env wrapper
        active_objects = [obj for obj in self.objects if not isinstance(obj, NoObject)]
        
        # Patching objects to ensure they have expected attributes if missing
        # env.py expects: category, x, y, (or center), orientation (optional), w (for OxygenBar)
        # Ocatari Vision objects have x, y, w, h, xywh.
        # They assume category is set.
        
        logic_state = self.extract_logic_state(active_objects)
        
        # Flatten for neural part if needed, or just pass logic state if model handles it.
        # The agent needs (logic_state, neural_state) usually.
        neural_state = self.env_wrapper.extract_neural_state(active_objects)
        
        # 4. Get Valuation
        # Agent.act or get_valuation
        # We need the valuation tensor.
        # Assuming agent.model has V_0 or similar as seen in play_il_gui.py
        
        # Helper to unsqueeze batch dim
        logic_state_b = logic_state.unsqueeze(0)
        neural_state_b = neural_state.unsqueeze(0)
        
        valuation = None
        with torch.no_grad():
            # We assume the agent's act method or forward method populates the valuation in the model
            # or returns it.
            # ImitationAgent.act expects a tensor (logic state), not a tuple.
            # So we pass logic_state_b only.
            _ = self.agent.act(logic_state_b)
            
            if hasattr(self.agent.model, 'V_0'):
                valuation = self.agent.model.V_0.squeeze(0).cpu().numpy()
            elif hasattr(self.agent.model, 'program_generator'):
                 # Try digging into RDN specific structure if V_0 isn't direct
                 pass

        return image_rgb, valuation

    def render(self, image, valuation, frame_name):
        self.window.fill(CELL_BACKGROUND_DEFAULT)
        
        # Render Game Image
        # Scale up
        img_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        img_surface = pygame.transform.scale(img_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.window.blit(img_surface, (0, 0))
        
        # Render Gaze
        if frame_name in self.trajectory_data:
            gaze_points = self.trajectory_data[frame_name].get("gaze", [])
            for gp in gaze_points:
                # Gaze is likely normalized or in 160x210 coords?
                # The text file shows values like ~80, ~100. Max bounds 160, 210.
                # So they are likely pixel coords in original resolution.
                gx = int(gp[0] * 4)
                gy = int(gp[1] * 4)
                pygame.draw.circle(self.window, (255, 0, 0), (gx, gy), 5) # Red gaze point
        
        # Render Valuation (Predicates)
        if valuation is not None:
            # We need the list of atoms/predicates to label them.
            # self.agent.actor.atoms usually holds them?
            # It seems 'ImitationAgent' might wrap 'nsfr_reasoner' (the model).
            # The atoms are usually in the model instance.
            
            atoms = None
            if hasattr(self.agent.model, 'atoms'):
                atoms = self.agent.model.atoms
            
            if atoms:
                self._render_predicate_probs(valuation, atoms)
        
        # Render Info
        info_text = self.font.render(f"Frame: {self.current_frame_idx} | {frame_name}", True, (255, 255, 255))
        self.window.blit(info_text, (10, 10))
        
        pygame.display.flip()

    def _render_predicate_probs(self, valuation, atoms):
        # Adapted from ILRenderer
        start_x = SCREEN_WIDTH + 10
        start_y = 50
        
        # Sort top high value predicates
        # Filter > 0.1
        pairs = []
        for i, val in enumerate(valuation):
            if i < len(atoms):
                pairs.append((atoms[i], val))
        
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = pairs[:15]
        
        title_text = self.font.render("Top Predicates:", True, "yellow")
        self.window.blit(title_text, (start_x, start_y - 30))
        
        for i, (atom, val) in enumerate(top_pairs):
            val_norm = min(max(val, 0), 1)
            color = val_norm * np.array([40, 200, 100]) + (1 - val_norm) * CELL_BACKGROUND_DEFAULT
            color = np.clip(color, 0, 255).astype(int)
            
            pygame.draw.rect(self.window, color.tolist(), [
                start_x, 
                start_y + i * 30, 
                PREDICATE_PROBS_COL_WIDTH - 20, 
                25
            ])
            
            text = self.font.render(f"{val:.2f} {str(atom)}", True, "white")
            self.window.blit(text, (start_x + 5, start_y + i * 30 + 2))

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_RIGHT:
                        self.current_frame_idx += 1
                    elif event.key == pygame.K_LEFT:
                        self.current_frame_idx -= 1
                    elif event.key == pygame.K_q:
                        self.running = False

            if self.current_frame_idx < 0: self.current_frame_idx = 0
            if self.current_frame_idx >= len(self.images): self.current_frame_idx = len(self.images) - 1
            
            current_image_path = self.images[self.current_frame_idx]
            frame_name = os.path.basename(current_image_path)
            
            image_rgb, valuation = self.process_frame(current_image_path)
            
            if image_rgb is not None:
                self.render(image_rgb, valuation, frame_name)
            
            if not self.paused:
                self.current_frame_idx += 1
                if self.current_frame_idx >= len(self.images):
                    self.paused = True
            
            self.clock.tick(30) # 30 FPS

        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/seaquest/gaze_data_tmp/54_RZ_2461867_Aug-11-09-35-18", help="Path to trajectory folder (images)")
    parser.add_argument("--agent_path", type=str, default="out/imitation/seaquest_defaultil.pth", help="Path to trained agent .pth")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    
    args = parser.parse_args()
    
    # Adjust paths relative to project root if needed
    # Note: data path in arg default assumes being run from NeSY-Imitation-Learning or NUDGE root?
    # The default above looks like relative path.
    # We should make it robust.
    
    # Resolve absolute paths
    project_root = "/home/nikhilesh/Projects/NUDGE" # running from here?
    nesy_root = "/home/nikhilesh/Projects/NeSY-Imitation-Learning"
    
    # Default override if standard paths
    if "NeSY-Imitation-Learning" in args.data_path or "data/seaquest" in args.data_path:
        # Check if absolute or relative
        if not os.path.isabs(args.data_path):
             # Try combining with nesy root
             combined = os.path.join(nesy_root, args.data_path)
             if os.path.exists(combined):
                 args.data_path = combined
    
    visualizer = TrajectoryVisualizer(args.data_path, args.agent_path, start_frame=args.start_frame)
    visualizer.run()
