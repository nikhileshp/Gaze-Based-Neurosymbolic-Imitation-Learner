from typing import Sequence

from nudge.env import NudgeBaseEnv
# from ocatari.core import OCAtari
import numpy as np
import torch as th
from ocatari.ram.seaquest import MAX_NB_OBJECTS as MAX_ESSENTIAL_OBJECTS
from hackatari import HackAtari


class NudgeEnv(NudgeBaseEnv):
    name = "seaquest"
    pred2action = {
        'noop': 0,
        'fire': 1,
        'up': 2,
        'right': 3,
        'left': 4,
        'down': 5,
    }
    pred_names: Sequence

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False):
        super().__init__(mode)
        self.env = HackAtari(env_name="ALE/Seaquest-v5", mode="vision",
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay)
        self.n_objects = 49 # Increased from 47 to include Surface
        self.n_features = 7  # visible, x-pos, y-pos, width, height, right-facing, type_id

        # Compute index offsets. Needed to deal with multiple same-category objects
        self.obj_offsets = {}
        offset = 0
        
        # Override EnemyMissile limit
        if 'EnemyMissile' in MAX_ESSENTIAL_OBJECTS:
             MAX_ESSENTIAL_OBJECTS['EnemyMissile'] = 8
             
        for (obj, max_count) in MAX_ESSENTIAL_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_ESSENTIAL_OBJECTS.keys())

    def reset(self, seed: int = None, options: dict = None):
        self.env.reset(seed=seed, options=options)
        state = self.env.objects
        return self.convert_state(state)

    def step(self, action, is_mapped: bool = False):
        if not is_mapped:
            action = action.split("_")[0]
            action = self.map_action(action)
        _, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        state = self.env.objects
        return self.convert_state(state), reward, done

    def get_rgb_frame(self):
        # HackAtari wraps the environment. We need to access the unwrapped ALE env.
        return self.env.unwrapped.ale.getScreenRGB()

    def extract_logic_state(self, raw_state):
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)

        obj_count = {k: 0 for k in MAX_ESSENTIAL_OBJECTS.keys()}

        type_map = {
            'Shark': 0, 'Submarine': 0, 'SurfaceSubmarine': 0,
            'Diver': 1, 'CollectedDiver': 6,
            'OxygenBar': 2,
            'Player': 3,
            'EnemyMissile': 5, 'PlayerMissile': 5,
            'Surface': 7
        }

        for obj in raw_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            type_id = type_map.get(obj.category, 0) # Default to 0 (enemy) if unknown
            # print(obj.category)
            if obj.category == "OxygenBar":
                #print all object keys and values
                # In Seaquest, oxygen level is represented by the bar's WIDTH
                oxygen_level = getattr(obj, "w", 0)  # Use width instead of value
                
                # DEBUG: Print OxygenBar attributes once
                if not hasattr(self, '_oxygen_debug_printed'):
                    # print(f"DEBUG OxygenBar attributes: {dir(obj)}\")")
                    # print(f"  value={getattr(obj, 'value', 'N/A')}, x={obj.x}, y={obj.y}, w={obj.w}, h={obj.h}")
                    # print(f"  Using width (w={oxygen_level}) as oxygen level")
                    self._oxygen_debug_printed = True
                
                # [vis, x, y, w, h, orientation, type_id]
                state[idx] = th.tensor([1, int(obj.x), int(obj.y), int(obj.w), int(obj.h), 0, type_id], dtype=th.int32)
            else:
                orientation = getattr(obj, "orientation", None)
                orientation = orientation.value if orientation is not None else 0
                w = getattr(obj, "w", 0)
                h = getattr(obj, "h", 0)
                state[idx] = th.tensor([1, *obj.center, w, h, orientation, type_id])
            obj_count[obj.category] += 1

        return state

    def extract_neural_state(self, raw_state):
        return th.flatten(self.extract_logic_state(raw_state))

    def close(self):
        self.env.close()
