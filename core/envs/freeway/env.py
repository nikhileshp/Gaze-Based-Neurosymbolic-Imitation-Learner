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

    def reset(self):
        self.env.reset()
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
        num_of_feature = 5 # visibility, chicken_type, car_type, x, y
        num_of_object = 11
        logic_state = np.zeros((num_of_object, num_of_feature))

        for i, entity in enumerate(raw_state):
            if i >= num_of_object:
                break
            logic_state[i][0] = 1 # Visibility
            if entity.category == "Chicken":
                logic_state[i][1] = 1 # Type Chicken
                logic_state[i][3:5] = entity.xy
            elif entity.category == 'Car':
                logic_state[i][2] = 1 # Type Car
                logic_state[i][3:5] = entity.xy

        return logic_state

    def extract_neural_state(self, raw_state):
        return self.extract_logic_state(raw_state).flatten()

    def close(self):
        self.env.close()
