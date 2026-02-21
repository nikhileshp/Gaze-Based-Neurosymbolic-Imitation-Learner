from nudge.renderer import Renderer, yellow, print_program, PREDICATE_PROBS_COL_WIDTH, CELL_BACKGROUND_DEFAULT, CELL_BACKGROUND_HIGHLIGHT, CELL_BACKGROUND_SELECTED
from argparse import ArgumentParser
from pathlib import Path
import torch
import pygame
import numpy as np
from nudge.agents.imitation_agent import ImitationAgent
from nudge.env import NudgeBaseEnv
from hackatari.core import HackAtari



parser = ArgumentParser()
parser.add_argument("-g", "--game", type=str, default="seaquest")
parser.add_argument("-r", "--rules", type=str, default="default")
parser.add_argument("-a", "--agent_path", type=str, default="out/imitation/seaquest_il.pth")
parser.add_argument("-np", "--no_predicates", action="store_true")
parser.add_argument("-d", "--device", type=str, default="cpu")
parser.add_argument("-db", "--debug",type=bool, default=False)
parser.add_argument("--use_gazemap", action="store_true", help="Visualize gaze predictions dynamically")
parser.add_argument("--gaze_model_path", type=str, default="seaquest_gaze_predictor_2.pth")

try:
    from scripts.gaze_predictor import Human_Gaze_Predictor
except ImportError:
    Human_Gaze_Predictor = None

from collections import deque
import cv2

def preprocess_frame(frame):
    """Convert raw 210x160x3 RGB frame to 84x84 grayscale frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

class AgentWrapper:
    def __init__(self, agent, env, debug=False, gaze_predictor=None):
        self.agent = agent
        self.env = env
        self.actor = agent.model # For Renderer to access prednames and print_program
        self.step_count = 0
        self.current_neural_predicates = []  # Store for GUI display
        self.current_objects = []  # Store detected objects
        self.debug = debug
        self.gaze_predictor = gaze_predictor
        self.latest_gaze_map = None
        self.frame_buffer = None
        
        if self.gaze_predictor is not None:
            self.frame_buffer = deque(maxlen=4)
            initial_rgb = env.get_rgb_frame()
            initial_gray = preprocess_frame(initial_rgb)
            for _ in range(4):
                self.frame_buffer.append(initial_gray)

    def act(self, state):
        # Renderer passes state as tensor with batch dim (1, num_atoms, num_features)
        # ImitationAgent.act handles batch dim check
        
        self.step_count += 1
        
        # Store detected objects for display
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'objects'):
            objects = self.env.env.objects
            self.current_objects = [(obj.__class__.__name__, 
                                    f"({obj.x}, {obj.y})" if hasattr(obj, 'x') else "N/A") 
                                   for obj in objects[:10]]
        
        # Advance Gaze Frame Buffer and Predict
        gaze_tensor = None
        if self.gaze_predictor is not None:
            if self.step_count > 1: # On exact first step, we use the prefilled buffer from init
                next_rgb = self.env.get_rgb_frame()
                next_gray = preprocess_frame(next_rgb)
                self.frame_buffer.append(next_gray)
            
            img_stack = np.stack(self.frame_buffer, axis=-1)
            input_tensor = torch.tensor(img_stack, dtype=torch.float32, device=self.gaze_predictor.device)
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) # (1, 4, H, W)
            
            with torch.no_grad():
                gaze_pred = self.gaze_predictor.model(input_tensor)
            gaze_tensor = gaze_pred.squeeze(0).squeeze(0) # (84, 84)
            self.latest_gaze_map = gaze_tensor.cpu().numpy()
            
        # Get action probabilities and extract valuation tensor
        with torch.no_grad():
            if state.dim() == 1:
                state_input = state.unsqueeze(0)
            else:
                state_input = state
            
            # Pass gaze to NSFR model if available
            probs = self.agent.model(state_input, gaze=gaze_tensor).squeeze(0)
            
            # Get the valuation tensor from the model (this has the actual atom probabilities)
            if hasattr(self.agent.model, 'V_0'):
                valuation = self.agent.model.V_0.squeeze(0)  # Remove batch dimension
            else:
                valuation = None
        
        # Extract high-value neural predicates from VALUATION tensor (not input state!)
        if valuation is not None and hasattr(self.actor, 'atoms'):
            atoms = self.actor.atoms
            val_np = valuation.detach().cpu().numpy()
            
            # The valuation is a 1D tensor of probabilities (0-1) for each atom
            # Find predicates with values > 0.5 and store top 15
            high_value_indices = [(i, float(val_np[i])) for i in range(min(len(val_np), len(atoms))) if float(val_np[i]) > 0.1]
            high_value_indices.sort(key=lambda x: x[1], reverse=True)
            
            self.current_neural_predicates = [(str(atoms[idx]), val) for idx, val in high_value_indices[:25]]
        else:
            self.current_neural_predicates = []
        
        action_idx = self.agent.act(state, gaze=gaze_tensor)
        
        # Enhanced debug output every 10 steps
        if self.debug:
            if self.step_count % 10 == 1:
                prednames = self.actor.prednames
                # print(f"\n{'='*60}")
                # print(f"Step {self.step_count}: {prednames[action_idx]} (prob: {probs[action_idx]:.3f})")
            
            # Show detected objects
            if hasattr(self.env, 'env') and hasattr(self.env.env, 'objects'):
                objects = self.env.env.objects
                # print(f"\nDetected Objects ({len(objects)} total):")
                
                # Show ALL non-NoObject instances with their indices
                non_no_objects = []
                for i, obj in enumerate(objects):
                    obj_type = obj.__class__.__name__
                    if obj_type != 'NoObject':
                        pos = f"({obj.x}, {obj.y})" if hasattr(obj, 'x') else "N/A"
                        non_no_objects.append((i, obj_type, pos))
                        
                for idx, obj_type, pos in non_no_objects:
                    print(f"  obj{idx}: {obj_type} at {pos}")
                
                # Also check specific object indices mentioned in predicates
                # print(f"\nSpecific Object Indices (from predicates):")
                for check_idx in [0, 1, 2, 3, 4, 5, 37]:
                    if check_idx < len(objects):
                        obj = objects[check_idx]
                        obj_type = obj.__class__.__name__
                        pos = f"({obj.x}, {obj.y})" if hasattr(obj, 'x') and hasattr(obj, 'y') else "N/A"
                        # print(f"  obj{check_idx}: {obj_type} at {pos}")
            
            # Show top neural predicates with their object references
            # print(f"\nTop Neural Predicates:")
            for pred_str, val in self.current_neural_predicates[:20]:
                print(f"  {val:.3f} - {pred_str}")
            
            # Check for type predicates specifically
            # print(f"\nType Predicates Check:")
            if hasattr(self.actor, 'atoms'):
                atoms = self.actor.atoms
                
                # Prepare state for debug inspection (kept for reference but not used for preds)
                if state.dim() == 1:
                    state_squeezed = state
                else:
                    state_squeezed = state.squeeze(0)
                state_np = state_squeezed.detach().cpu().numpy()
                
                # Find all type predicates
                for idx, atom in enumerate(atoms):
                    atom_str = str(atom)
                    if 'type(' in atom_str and idx < len(val_np):
                         val = float(val_np[idx])
                         if val > 0.3:  # Show any type with >0.3 value
                             print(f"  {val:.3f} - {atom_str}")

                # DEBUG: Specific check for close_by predicates
                # print(f"\nClose-by Predicates Check:")
                for idx, atom in enumerate(atoms):
                    atom_str = str(atom)
                    if 'higher_than' in atom_str and idx < len(val_np):
                         val = float(val_np[idx])
                         print(f"  {val:.3f} - {atom_str}")

                # DEBUG: Specific check for close_by predicates
                # print(f"\nClose-by Predicates Check:")
                for idx, atom in enumerate(atoms):
                    atom_str = str(atom)
                    if 'close_by' in atom_str and idx < len(val_np):
                         val = float(val_np[idx])
                         print(f"  {val:.3f} - {atom_str}")

                # DEBUG: Specific check for same_depth predicates
                # print(f"\nSame-depth Predicates Check:")
                # print(f"DEBUG: len(atoms)={len(atoms)}, len(val_np)={len(val_np)}")
                
                for idx, atom in enumerate(atoms):
                    atom_str = str(atom)
                    if 'same_depth' in atom_str:
                         if idx < len(val_np):
                             val = float(val_np[idx])
                            #  print(f"  {val:.3f} - {atom_str}")
                         else:
                            pass
                            #  print(f"  [OUT OF BOUNDS] - {atom_str} (idx={idx})")
            
            # DEBUG: Show raw object tensor values
            # print(f"\nDEBUG - Object Tensor Values:")
            if hasattr(self.env, 'env') and hasattr(self.env.env, 'objects'):
                objects = self.env.env.objects
                # Check the convert_state method to see object tensors
                if hasattr(self.env, 'convert_state'):
                    logic_state_debug, _ = self.env.convert_state(objects)
                    # print(f"  Logic state shape: {logic_state_debug.shape}")
                    # print(f"  Logic state type: {type(logic_state_debug)}")
                    
                    # Try to understand the structure
                    if hasattr(self.env, 'obj_encoder') and hasattr(self.env.obj_encoder, 'encode_objects'):
                        try:
                            encoded = self.env.obj_encoder.encode_objects(objects)
                            # print(f"  Encoded objects shape: {encoded.shape}")
                            # Show first few object tensors
                            for i in range(min(5, encoded.shape[0])):
                                obj_tensor = encoded[i]
                                # print(f"  obj{i} tensor[0:4]: {obj_tensor[:4].tolist()}")
                        except Exception as e:
                            pass
                            # print(f"  Could not inspect encoded objects: {e}")
            
            # print(f"{'='*60}\n")
            
        return torch.tensor(action_idx), None

class ILRenderer(Renderer):
    def __init__(self, model, env, device="cpu", fps=15, deterministic=True, env_kwargs=None, render_predicate_probs=True):
        self.fps = fps
        self.deterministic = deterministic
        self.render_predicate_probs = render_predicate_probs

        # Set model and env directly
        self.model = model
        self.env = env
        self.env.reset()

        print(f"Playing '{self.model.env.name}' with {'' if deterministic else 'non-'}deterministic policy.")

        try:
            self.action_meanings = self.env.env.get_action_meanings()
            ocenv = self.env.env
            if isinstance(ocenv, HackAtari):
                ocenv = ocenv.env
            self.keys2actions = ocenv.get_keys_to_action()
        except Exception:
            print(yellow("Info: No key-to-action mapping found for this env. No manual user control possible."))
            self.action_meanings = None
            self.keys2actions = {}
        self.current_keys_down = set()

        self.nsfr_reasoner = self.model.actor
        print("====== LEARNED PROGRAM ======")
        print_program(self.model)
        self.predicates = self.nsfr_reasoner.prednames

        self._init_pygame()

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.reset = False
        self.takeover = False

    def _render_env(self):
        super()._render_env()
        # Overlay the live Gaze Predictor heatmap if available
        if hasattr(self.model, 'latest_gaze_map') and self.model.latest_gaze_map is not None:
            gaze_map = self.model.latest_gaze_map
            import cv2
            import numpy as np
            import pygame
            
            # gaze_map is (84, 84). We need to resize it to self.env_render_shape (which is (width, height) usually (160, 210))
            # Wait, self.env_render_shape is (210, 160) from frame.shape[:2]. cv2.resize takes (width, height)
            # We want it to stretch out to match the window frame exactly.
            target_size = (self.env_render_shape[1], self.env_render_shape[0]) 
            heatmap = cv2.resize(gaze_map, target_size)
            
            # Normalize map to 0-255 (intensify by 2.0x for visibility)
            heatmap_norm = np.clip(heatmap * 255.0 * 2.5, 0, 255).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
            
            # Pygame surfaces expect (width, height, channels), but CV2 array is (height, width, channels)
            heatmap_surface = pygame.surfarray.make_surface(colored_heatmap.swapaxes(0, 1))
            heatmap_surface.set_alpha(110) # Semi-transparent
            self.window.blit(heatmap_surface, (0, 0))

    def _render_predicate_probs(self):
        """Override to show action probabilities and neural predicates"""
        # First show action probabilities (from parent)
        anchor = (self.env_render_shape[0] + 10, 25)
        
        nsfr = self.nsfr_reasoner
        pred_vals = {pred: nsfr.get_predicate_valuation(pred, initial_valuation=False) for pred in nsfr.prednames}
        i_max = np.argmax(list(pred_vals.values()))
        
        # Render action probabilities
        # IF debug is true
        for i, (pred, val) in enumerate(pred_vals.items()):
            # Render cell background
            if i == i_max:
                color = CELL_BACKGROUND_SELECTED
            else:
                color = val * CELL_BACKGROUND_HIGHLIGHT + (1 - val) * CELL_BACKGROUND_DEFAULT
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2,
                anchor[1] - 2 + i * 35,
                PREDICATE_PROBS_COL_WIDTH - 12,
                28
            ])

            text = self.font.render(str(f"{100*val:.2f} - {pred}"), True, "white", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, 25 + i * 35)
            self.window.blit(text, text_rect)
        
        # Calculate starting position for detected objects
        objects_start_y = 5 + len(pred_vals) * 35 + 30
        
        # Render detected objects section
        # small_font = pygame.font.SysFont('Calibri', 16)
        # title_text = small_font.render("Detected Objects:", True, "cyan", None)
        # title_rect = title_text.get_rect()
        # title_rect.topleft = (self.env_render_shape[0] + 10, objects_start_y - 20)
        # self.window.blit(title_text, title_rect)
        
        # Get detected objects from environment
        num_objects_shown = 0
        # if hasattr(self.env, 'env') and hasattr(self.env.env, 'objects'):
        #     objects = self.env.env.objects
        #     tiny_font = pygame.font.SysFont('Calibri', 14)
            
        #     # Show up to 8 objects
        #     for i, obj in enumerate(objects[:8]):
        #         obj_type = obj.__class__.__name__
        #         if obj_type != 'NoObject':  # Skip NoObjects
        #             text = tiny_font.render(f"obj{i}: {obj_type}", True, "white", None)
        #             text_rect = text.get_rect()
        #             text_rect.topleft = (self.env_render_shape[0] + 10, objects_start_y + num_objects_shown * 18)
        #             self.window.blit(text, text_rect)
        #             num_objects_shown += 1
                    
        #             if num_objects_shown >= 8:
        #                 break
        
        # Calculate starting position for neural predicates (after objects)
        neural_pred_start_y = objects_start_y + max(num_objects_shown * 18, 20) 
        
        # Render title for neural predicates
        title_text = self.font.render("Neural Predicates (>0.5):", True, "yellow", None)
        title_rect = title_text.get_rect()
        title_rect.topleft = (self.env_render_shape[0] + 10, neural_pred_start_y - 5)
        self.window.blit(title_text, title_rect)
        
        # Render neural predicates
        if hasattr(self.model, 'current_neural_predicates'):
            neural_preds = self.model.current_neural_predicates[:20]  # Top 12 (reduced for space)
            count=0
            for i, (pred_str, val) in enumerate(neural_preds):
                # Normalize value to 0-1 range (in case values are > 1)
                
                val_normalized = min(val, 1.0)
                if(val_normalized):
                    print(i, pred_str, val_normalized)
                
                    # Render background
                    color = val_normalized * np.array([40, 200, 100]) + (1 - val_normalized) * CELL_BACKGROUND_DEFAULT
                    color = np.clip(color, 0, 255).astype(int)  # Ensure valid RGB values
                    pygame.draw.rect(self.window, color.tolist(), [
                        anchor[0] - 2,
                        neural_pred_start_y - 2 + i * 25,
                        PREDICATE_PROBS_COL_WIDTH - 12,
                        22
                    ])
                    count+=1
                    # Render text (smaller font for neural predicates)
                    small_font = pygame.font.SysFont('Calibri', 14)
                    text = small_font.render(f"{val:.2f} - {pred_str}", True, "white", None)
                    text_rect = text.get_rect()
                    text_rect.topleft = (self.env_render_shape[0] + 10, neural_pred_start_y + i * 25)
                    self.window.blit(text, text_rect)
        else:
            # Show message if no predicates
            text = self.font.render("No predicates > 0.5", True, "gray", None)
            text_rect = text.get_rect()
            text_rect.topleft = (self.env_render_shape[0] + 10, neural_pred_start_y)
            self.window.blit(text, text_rect)

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Initialize Environment
    # We need to pass render_mode="human" or similar? 
    # Renderer uses env.env.render() which usually returns an array.
    # NudgeBaseEnv defaults: render_mode="rgb_array"
    env = NudgeBaseEnv.from_name(args.game, mode="logic", render_oc_overlay=True)
    
    # Initialize Gaze Predictor if requested
    gaze_predictor = None
    if args.use_gazemap:
        if Human_Gaze_Predictor is None:
            print("Error: Could not import Human_Gaze_Predictor. Ensure gaze_predictor.py exists in scripts/.")
            import sys; sys.exit(1)
            
        print(f"Initializing Gaze Predictor from {args.gaze_model_path}...")
        gaze_predictor = Human_Gaze_Predictor(args.game)
        gaze_predictor.init_model(args.gaze_model_path)
        gaze_predictor.model.eval()

    # Initialize Agent
    agent = ImitationAgent(args.game, args.rules, args.device)
    
    # Load Model
    print(f"Loading model from {args.agent_path}...")
    # agent.load(args.agent_path)
    print("WARNING: SKIPPING MODEL LOAD FOR DEBUGGING (USING FRESH MODEL)")
    agent.model.eval() # Set to eval mode
    
    # Wrap Agent
    model = AgentWrapper(agent, env, debug=args.debug, gaze_predictor=gaze_predictor)
    
    # Run Renderer
    renderer = ILRenderer(model=model,
                          env=env,
                          fps=15,
                          deterministic=True,
                          render_predicate_probs=not(args.no_predicates))
    
    renderer.run()
