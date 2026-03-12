from typing import Sequence
 
# from nudge.env import NudgeBaseEnv
from nsfr.env import NSFRBaseEnv
from ocatari.core import OCAtari
import numpy as np
 
 
class NSFREnv(NSFRBaseEnv):
    name = "asterix"
    pred2action = {
        'noop': 0,
        'up': 1,
        'right': 2,
        'left': 3,
        'down': 4,
    }
    pred_names: Sequence
 
    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False, oc_mode="vision"):
        super().__init__(mode)
        self.env = OCAtari(env_name="ALE/Asterix-v5", mode=oc_mode,
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay,
                           repeat_action_probability=0.25,
                           max_episode_steps=10000)
 
    def reset(self, seed: int = None, options: dict = None):
        self.env.reset(seed=seed, options=options)
        # Random no-op starts (standard ALE evaluation protocol, DQN paper).
        # Apply 1-30 noop steps at episode start to reach diverse starting states
        # even for a deterministic policy. Use the episode seed for reproducibility.
        rng = np.random.RandomState(seed if seed is not None else 0)
        noop_count = rng.randint(1, 31)
        for _ in range(noop_count):
            _, _, terminated, truncated, _ = self.env.step(0)  # 0 = noop
            if terminated or truncated:
                self.env.reset(seed=seed, options=options)
                break
        state = self.env.objects
        return self.convert_state(state)
 
 
    def step(self, action, is_mapped: bool = False):
        if not is_mapped:
            action = self.map_action(action)
        _, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        state = self.env.objects
        return self.convert_state(state), reward, done
 
    def extract_logic_state(self, raw_state):
        n_features = 7
        n_objects = 25 # 1 player + 8 enemies + 8 rewards + 8 consumables
        logic_state = np.zeros((n_objects, n_features))
 
        # Consumable item class names from OCAtari vision module
        consumable_names = {"Cauldron", "Helmet", "Shield", "Lamp", "Apple", "Fish", "Meat", "Mug"}
 
        obj_idx = 0 # Start from 0 to match obj0, obj1, ... objN
        for entity in raw_state:
            if obj_idx >= n_objects:
                break
            
            category = entity.category
            if category == "NoObject":
                continue
            
            # Index 0: Visibility
            logic_state[obj_idx][0] = 1

            # Indices 1-4: One-hot Type
            if category == "Player":
                logic_state[obj_idx][1] = 1
            elif category == "Enemy":
                logic_state[obj_idx][2] = 1
            elif category in consumable_names or "Bonus" in category or category == "Consumable":
                logic_state[obj_idx][3] = 1
            elif "Reward" in category:
                logic_state[obj_idx][4] = 1
                
            # Coordinates at the end
            logic_state[obj_idx][5:7] = entity.xy
            obj_idx += 1
 
        return logic_state
 
    def extract_neural_state(self, raw_state):
        # Neural state is just a flattened version of logic state for this agent
        logic_state = self.extract_logic_state(raw_state)
        return logic_state.flatten().tolist()
 
    def get_rgb_frame(self):
        """Returns the raw RGB frame from the underlying OCAtari environment."""
        return self.env.render()
 
    def close(self):
        self.env.close()