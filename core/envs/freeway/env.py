from typing import Sequence

from nsfr.env import NSFRBaseEnv
from ocatari.core import OCAtari
import numpy as np


class NSFREnv(NSFRBaseEnv):
    name = "freeway"
    pred2action = {
        'noop': 0,
        'up': 1,
        'down': 2,
    }
    pred_names: Sequence

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False, oc_mode="ram"):
        super().__init__(mode)
        self.env = OCAtari(env_name="ALE/Freeway-v5", mode=oc_mode,
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay)

    def reset(self, seed: int = None, options: dict = None):
        self.env.reset(seed=seed, options=options)
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
        num_of_feature = 5  # [visible, is_chicken, is_car, x, y]
        num_of_object = 12
        logic_state = np.zeros((num_of_object, num_of_feature))

        # OCAtari Freeway always reports 2 chickens (the player + a STATIC decoy at
        # the bottom, x~108) and up to 10 cars. The decoy is irrelevant to the task,
        # so keep ONLY the player chicken at obj0 and pack the cars densely into
        # obj1.. . The decoy, NoObjects and empty slots are dropped (left all-zero =>
        # not visible). This keeps the symbolic state consistent with the rule
        # domains (oagent:obj0, ocar:obj1..) and with the (remapped) training data.
        chicken_found = False
        car_idx = 1
        for entity in raw_state:
            cat = entity.category
            if cat == "Chicken" and not chicken_found:
                logic_state[0][0] = 1       # visible
                logic_state[0][1] = 1       # is_chicken (player)
                logic_state[0][3:5] = entity.xy
                entity._logic_slot = 0      # tag for the renderer overlay
                chicken_found = True
            elif cat == "Car" and car_idx < num_of_object:
                logic_state[car_idx][0] = 1  # visible
                logic_state[car_idx][2] = 1  # is_car
                logic_state[car_idx][3:5] = entity.xy
                entity._logic_slot = car_idx
                car_idx += 1

        return logic_state

    def extract_neural_state(self, raw_state):
        return self.extract_logic_state(raw_state).flatten()

    def get_rgb_frame(self):
        return self.env.render()

    def close(self):
        self.env.close()
