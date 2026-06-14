from typing import Sequence

import torch as th
from nsfr.env import NSFRBaseEnv
from ocatari.core import OCAtari
from ocatari.ram.spaceinvaders import MAX_NB_OBJECTS as MAX_ESSENTIAL_OBJECTS


# type_id values — MUST match the order of the `type:` line in
# logic/space_invaders_root/consts.txt (player,alien,shield,bullet,satellite).
TYPE_MAP = {
    "Player": 0,
    "Alien": 1,
    "Shield": 2,
    "Bullet": 3,
    "Satellite": 4,
}


class NSFREnv(NSFRBaseEnv):
    name = "space_invaders"
    # OCAtari minimal action set order: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE.
    # RIGHTFIRE/LEFTFIRE are intentionally omitted (rare combos, ~3.7% of human
    # frames) -> 4-action set.
    pred2action = {
        "noop": 0,
        "fire": 1,
        "move_right": 2,
        "move_left": 3,
    }
    pred_names: Sequence

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False, oc_mode="ram"):
        super().__init__(mode)
        self.env = OCAtari(env_name="ALE/SpaceInvaders-v5", mode=oc_mode,
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay)
        self.n_features = 6  # present, x, y, w, h, type_id

        # Slot offsets per category so multiple same-category objects get distinct
        # rows (Player 0, Shield 1-3, Bullet 4-6, Satellite 7, Alien 8-43).
        self.obj_offsets = {}
        offset = 0
        for (obj, max_count) in MAX_ESSENTIAL_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.n_objects = offset  # 1 + 3 + 3 + 1 + 36 = 44
        self.relevant_objects = set(MAX_ESSENTIAL_OBJECTS.keys())

    def reset(self, seed: int = None, options: dict = None):
        self.env.reset(seed=seed, options=options)
        return self.convert_state(self.env.objects)

    def step(self, action, is_mapped: bool = False):
        if not is_mapped:
            action = self.map_action(action)
        _, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return self.convert_state(self.env.objects), reward, done

    def get_rgb_frame(self):
        return self.env.unwrapped.ale.getScreenRGB()

    def extract_logic_state(self, raw_state) -> th.Tensor:
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)
        obj_count = {k: 0 for k in MAX_ESSENTIAL_OBJECTS}

        for obj in raw_state:
            # OCAtari pads empty slots with a 'NoObject' placeholder — skip those
            # and anything outside the tracked categories; their rows stay present=0.
            if obj.category not in self.relevant_objects:
                continue
            if obj_count[obj.category] >= MAX_ESSENTIAL_OBJECTS[obj.category]:
                continue
            # OCAtari reports inactive bullets/objects parked at the origin; treat
            # them as absent so they don't trigger spurious relational predicates.
            if int(obj.x) <= 0 and int(obj.y) <= 0:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            obj._logic_slot = idx  # tag for the renderer overlay (objN matches the panel)
            type_id = TYPE_MAP[obj.category]
            w = int(getattr(obj, "w", 0))
            h = int(getattr(obj, "h", 0))
            state[idx] = th.tensor(
                [1, int(obj.x), int(obj.y), w, h, type_id], dtype=th.int32
            )
            obj_count[obj.category] += 1

        return state

    def extract_neural_state(self, raw_state) -> th.Tensor:
        return th.flatten(self.extract_logic_state(raw_state))

    def close(self):
        self.env.close()
