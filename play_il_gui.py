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

class AgentWrapper:
    def __init__(self, agent, env, debug=False):
        self.agent = agent
        self.env = env
        self.actor = agent.model # For Renderer to access prednames and print_program
        self.step_count = 0
        self.current_neural_predicates = []  # Store for GUI display
        self.current_objects = []  # Store detected objects
        self.debug = debug

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
        
        # Get action probabilities and extract valuation tensor
        with torch.no_grad():
            if state.dim() == 1:
                state_input = state.unsqueeze(0)
            else:
                state_input = state
            probs = self.agent.model(state_input).squeeze(0)
            
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
        
        action_idx = self.agent.act(state)
        
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
                if(val_normalized < 0.99):
                    print(i, pred_str, val_normalized)
                
                    # Render background
                    color = val_normalized * np.array([40, 200, 100]) + (1 - val_normalized) * CELL_BACKGROUND_DEFAULT
                    color = np.clip(color, 0, 255).astype(int)  # Ensure valid RGB values
                    pygame.draw.rect(self.window, color.tolist(), [
                        anchor[0] - 2,
                        neural_pred_start_y - 2 + count * 25,
                        PREDICATE_PROBS_COL_WIDTH - 12,
                        22
                    ])
                    count+=1
                    # Render text (smaller font for neural predicates)
                    small_font = pygame.font.SysFont('Calibri', 14)
                    text = small_font.render(f"{val:.2f} - {pred_str}", True, "white", None)
                    text_rect = text.get_rect()
                    text_rect.topleft = (self.env_render_shape[0] + 10, neural_pred_start_y + count * 25)
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
    
    # Initialize Agent
    agent = ImitationAgent(args.game, args.rules, args.device)
    
    # Load Model
    print(f"Loading model from {args.agent_path}...")
    # agent.load(args.agent_path)
    print("WARNING: SKIPPING MODEL LOAD FOR DEBUGGING (USING FRESH MODEL)")
    agent.model.eval() # Set to eval mode
    
    # Wrap Agent
    model = AgentWrapper(agent, env, debug=args.debug)
    
    # Run Renderer
    renderer = ILRenderer(model=model,
                          env=env,
                          fps=15,
                          deterministic=True,
                          render_predicate_probs=not(args.no_predicates))
    
    renderer.run()
