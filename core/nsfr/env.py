from typing import Dict
from abc import ABC
from nsfr.utils.common import load_module
import torch
import os
from contextlib import redirect_stdout, redirect_stderr

class NSFRBaseEnv(ABC):
    name: str
    pred2action: Dict[str, int]  # predicate name to action index
    env: object  # the wrapped environment

    def __init__(self, mode: str):
        self.mode = mode  # either 'ppo' or 'logic'

    def reset(self, seed: int = None, options: dict = None) -> (torch.tensor, torch.tensor):
        """Returns logic_state, neural_state"""
        raise NotImplementedError

    def step(self, action, is_mapped: bool = False) -> ((torch.tensor, torch.tensor), float, bool):
        """If is_mapped is False, the action will be mapped from model action space to env action space.
        I.e., if is_mapped is True, this method feeds 'action' into the wrapped env directly.
        Returns (logic_state, neural_state), reward, done"""
        raise NotImplementedError

    def get_rgb_frame(self):
        """Returns the raw RGB frame from the underlying environment (e.g., 210x160x3 for Atari)."""
        raise NotImplementedError

    def extract_logic_state(self, raw_state) -> torch.tensor:
        """Turns the raw state representation into logic representation."""
        raise NotImplementedError

    def extract_neural_state(self, raw_state) -> torch.tensor:
        """Turns the raw state representation into neural representation."""
        raise NotImplementedError

    def convert_state(self, state) -> (torch.tensor, torch.tensor):
        return self.extract_logic_state(state), self.extract_neural_state(state)

    def map_action(self, model_action) -> int:
        """Converts a model action to the corresponding env action."""
        if self.mode == 'ppo':
            return model_action + 1
        else:  # logic
            # Sort predicate names by length descending to match longest valid prefixes
            # first (e.g., 'up_right' before 'up').
            pred_names = sorted(list(self.pred2action.keys()), key=len, reverse=True)
            for pred_name in pred_names:
                if model_action.startswith(pred_name):
                    return self.pred2action[pred_name]
            raise ValueError(f"Invalid predicate '{model_action}' provided. "
                             f"Must start with any of {pred_names}.")

    def n_actions(self) -> int:
        return len(list(set(self.pred2action.items())))

    @staticmethod
    def from_name(name: str, **kwargs):
        # Try 'in/envs' first, then fall back to 'core/envs'
        in_path = f"in/envs/{name}/env.py"
        core_path = f"core/envs/{name}/env.py"
        
        if os.path.exists(in_path):
            env_path = in_path
        else:
            env_path = core_path
            
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                env_module = load_module(env_path)
        return env_module.NSFREnv(**kwargs)

    def close(self):
        pass
