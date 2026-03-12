from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch as th
import pygame

from .agents.logic_agent import NsfrActorCritic
from .agents.neural_agent import ActorCritic
from .utils import load_model, yellow, print_program

from ocatari.core import OCAtari
from hackatari.core import HackAtari


SCREENSHOTS_BASE_PATH = "out/screenshots/"
PREDICATE_PROBS_COL_WIDTH = 300
CELL_BACKGROUND_DEFAULT = np.array([40, 40, 40])
CELL_BACKGROUND_HIGHLIGHT = np.array([40, 150, 255])
CELL_BACKGROUND_SELECTED = np.array([80, 80, 80])

class Renderer:
    model: Union[NsfrActorCritic, ActorCritic]
    window: pygame.Surface
    clock: pygame.time.Clock

    def __init__(self,
                 agent_path: str = None,
                 device: str = "cpu",
                 fps: int = None,
                 deterministic=True,
                 env_kwargs: dict = None,
                 render_predicate_probs=True):

        self.fps = fps
        self.deterministic = deterministic
        self.render_predicate_probs = render_predicate_probs

        # Load model and environment
        self.model = load_model(agent_path, env_kwargs_override=env_kwargs, device=device)
        self.env = self.model.env
        self.env.reset()

        print(f"Playing '{self.model.env.name}' with {'' if deterministic else 'non-'}deterministic policy.")

        if fps is None:
            fps = 15
        self.fps = fps

        try:
            self.action_meanings = self.env.env.get_action_meanings()
            ocenv = self.env.env
            if isinstance(ocenv, HackAtari):
                ocenv = ocenv.env
            self.keys2actions = ocenv.get_keys_to_action()
            print(f"DEBUG: keys2actions: {self.keys2actions}")
        except Exception:
            print(yellow("Info: No key-to-action mapping found for this env. No manual user control possible."))
            self.action_meanings = None
            self.keys2actions = {}
        self.current_keys_down = set()

        self.nsfr_reasoner = self.model.actor
        # self.nsfr_reasoner.print_program()
        print_program(self.model)
        self.predicates = self.nsfr_reasoner.prednames

        self._init_pygame()

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.reset = False
        self.takeover = False

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Environment")
        frame = self.env.env.render()
        self.env_render_shape = frame.shape[:2]
        window_shape = list(self.env_render_shape)
        if self.render_predicate_probs:
            window_shape[0] += PREDICATE_PROBS_COL_WIDTH
        self.window = pygame.display.set_mode(window_shape, pygame.SCALED)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Calibri', 24)

    def run(self):
        length = 0
        ret = 0

        obs, _ = self.env.reset()

        while self.running:
            self.reset = False
            self._handle_user_input()

            if not self.running:
                break  # outer game loop

            if self.takeover:  # human plays game manually
                action = self._get_action()
                self.model.act(th.unsqueeze(th.tensor(obs), 0))  # update the model's internals
            else:  # AI plays the game
                action, _ = self.model.act(th.unsqueeze(th.tensor(obs), 0))
                action = self.predicates[action.item()]

            if not self.paused:
                (new_obs, _), reward, done = self.env.step(action, is_mapped=self.takeover)

                self._render()
                ret += reward

                if self.takeover and float(reward) != 0:
                    print(f"Reward {reward:.2f}")

                if self.reset:
                    done = True
                    new_obs, _ = self.env.reset()
                    self._render()

                obs = new_obs
                length += 1

                if done:
                    print(f"Return: {ret} - Length {length}")
                    ret = 0
                    length = 0
                    self.env.reset()

        pygame.quit()

    def _get_action(self):
        if self.keys2actions is None:
            return 0  # NOOP
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            return self.keys2actions[pressed_keys]
        else:
            return 0  # NOOP

    def _handle_user_input(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                self.running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_p:  # 'P': pause/resume
                    self.paused = not self.paused

                elif event.key == pygame.K_r:  # 'R': reset
                    self.reset = True

                elif event.key == pygame.K_f:  # 'F': fast forward
                    self.fast_forward = not(self.fast_forward)

                elif event.key == pygame.K_t:  # 'T': trigger takeover
                    if self.takeover:
                        print("AI takeover")
                    else:
                        print("Human takeover")
                    self.takeover = not self.takeover

                elif event.key == pygame.K_c:  # 'C': capture screenshot
                    file_name = f"{datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')}.png"
                    save_path = Path(SCREENSHOTS_BASE_PATH) / file_name
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    pygame.image.save(self.window, str(save_path))
                    print(f"Screenshot saved to {save_path}")

                else:
                    mapped_key = self._map_key(event.key)
                    if mapped_key is not None:
                        # print(f"DEBUG: Key down: {pygame.key.name(event.key)} -> {mapped_key}")
                        self.current_keys_down.add(mapped_key)

            elif event.type == pygame.KEYUP:  # keyboard key released
                mapped_key = self._map_key(event.key)
                if mapped_key in self.current_keys_down:
                    self.current_keys_down.remove(mapped_key)

    def _map_key(self, pygame_key):
        """Maps a pygame key to the format expected by the environment's keys2actions."""
        if not self.keys2actions:
            return None
            
        # Determine if mapping uses strings or integers
        first_key_tuple = next(iter(self.keys2actions.keys()), None)
        if not first_key_tuple:
            return None
        use_strings = isinstance(first_key_tuple[0], str)
        
        if use_strings:
            # Standard WASD mapping for Atari string-based envs
            arrow_map = {
                pygame.K_UP: 'w',
                pygame.K_DOWN: 's',
                pygame.K_LEFT: 'a',
                pygame.K_RIGHT: 'd',
                pygame.K_SPACE: 'e',
                pygame.K_f: 'e' # Sometimes F is used for fire
            }
            if pygame_key in arrow_map:
                mapped = arrow_map[pygame_key]
            else:
                mapped = pygame.key.name(pygame_key)
                
            # Check if this mapped key exists in any of the valid action tuples
            for k_tuple in self.keys2actions.keys():
                if mapped in k_tuple:
                    return mapped
            return None
        else:
            # Integer-based mapping (standard Gym/OCAtari)
            if (pygame_key,) in self.keys2actions.keys() or any(pygame_key in k for k in self.keys2actions.keys()):
                return pygame_key
            return None

    def _render(self):
        self.window.fill((20, 20, 20))  # clear the entire window
        self._render_env()
        if self.render_predicate_probs:
            self._render_predicate_probs()

        pygame.display.flip()
        pygame.event.pump()
        if not self.fast_forward:
            self.clock.tick(self.fps)

    def _render_env(self):
        frame = self.env.env.render()
        frame_surface = pygame.Surface(self.env_render_shape)
        pygame.pixelcopy.array_to_surface(frame_surface, frame)
        self.window.blit(frame_surface, (0, 0))
        
        # DRAW TAKEOVER STATUS
        if hasattr(self, 'takeover') and self.takeover:
            takeover_font = pygame.font.SysFont('Calibri', 32, bold=True)
            text = takeover_font.render("HUMAN TAKEOVER", True, (255, 50, 50))
            text_rect = text.get_rect()
            text_rect.topleft = (10, 10)
            # Add a semi-transparent background for readability
            bg_rect = text_rect.copy()
            bg_rect.inflate_ip(10, 5)
            pygame.draw.rect(self.window, (0, 0, 0, 150), bg_rect)
            self.window.blit(text, text_rect)

    def _render_predicate_probs(self):
        anchor = (self.env_render_shape[0] + 10, 25)

        nsfr = self.nsfr_reasoner
        pred_vals = {pred: nsfr.get_predicate_valuation(pred, initial_valuation=False) for pred in nsfr.prednames}
        i_max = np.argmax(list(pred_vals.values()))
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
