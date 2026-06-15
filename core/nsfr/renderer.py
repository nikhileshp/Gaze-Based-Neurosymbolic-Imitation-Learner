from datetime import datetime
from typing import Union

import numpy as np
import torch as th
import pygame

from .agents.logic_agent import NsfrActorCritic
from .agents.neural_agent import ActorCritic
from .utils import load_model, yellow, print_program
from .rule_explain import rule_attributions, order_for_display

from ocatari.core import OCAtari
from hackatari.core import HackAtari


SCREENSHOTS_BASE_PATH = "out/screenshots/"
PREDICATE_PROBS_COL_WIDTH = 300
NUM_PANEL_COLUMNS = 2  # rule panel flows into a 2nd column when it overflows
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
            self._normalize_keys()
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

    def _normalize_keys(self):
        """Normalizes keys2actions to use integers instead of strings/characters."""
        new_keys2actions = {}
        for keys, action in self.keys2actions.items():
            new_key_tuple = []
            for k in keys:
                if isinstance(k, str):
                    try:
                        # k might be ' ' or 'w' or 'space' or 'up'
                        code = pygame.key.key_code(k)
                        new_key_tuple.append(code)
                    except ValueError:
                        new_key_tuple.append(k) # Fallback
                else:
                    new_key_tuple.append(k)
            new_keys2actions[tuple(sorted(new_key_tuple))] = action
        self.keys2actions = new_keys2actions

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Environment")
        frame = self.env.env.render()
        self.env_render_shape = frame.shape[:2]
        window_shape = list(self.env_render_shape)
        if self.render_predicate_probs:
            window_shape[0] += PREDICATE_PROBS_COL_WIDTH * NUM_PANEL_COLUMNS
        # Draw onto a fixed-size logical surface, then scale it to fit the actual
        # display in _render(). Keeps the whole window visible even when the logical
        # size exceeds the (VNC/Xvfb) screen, instead of clipping the right/bottom.
        self.window = pygame.Surface(window_shape)
        try:
            desktop_w, desktop_h = pygame.display.get_desktop_sizes()[0]
            scale = min(desktop_w / window_shape[0], desktop_h / window_shape[1])
        except Exception:
            scale = 1.0
        screen_size = (max(1, round(window_shape[0] * scale)),
                       max(1, round(window_shape[1] * scale)))
        self.screen = pygame.display.set_mode(screen_size)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Calibri', 24)

    def run(self):
        length = 0
        ret = 0
        self.taken_head = None  # head name of the rule whose action was executed

        obs, _ = self.env.reset()

        while self.running:
            self.reset = False
            self._handle_user_input()

            if not self.running:
                break  # outer game loop

            if self.takeover:  # human plays game manually
                action = self._get_action()
                self.model.act(th.unsqueeze(th.tensor(obs), 0))  # update the model's internals
                self.taken_head = None
            else:  # AI plays the game
                action, _ = self.model.act(th.unsqueeze(th.tensor(obs), 0))
                action = self.predicates[action.item()]
                self.taken_head = action

            # Render the state the decision was made on (env not yet stepped), so the
            # overlay frame and the panel's valuation (V_0 of `obs`) are the same frame.
            self._render()

            if not self.paused:
                (new_obs, _), reward, done = self.env.step(action, is_mapped=self.takeover)
                ret += reward

                if self.takeover and float(reward) != 0:
                    print(f"Reward {reward:.2f}")

                if self.reset:
                    done = True
                    new_obs, _ = self.env.reset()

                obs = new_obs
                length += 1

                if done:
                    print(f"Return: {ret} - Length {length}")
                    ret = 0
                    length = 0
                    obs, _ = self.env.reset()

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
                    pygame.image.save(self.window, SCREENSHOTS_BASE_PATH + file_name)

                elif (event.key,) in self.keys2actions.keys():  # env action
                    self.current_keys_down.add(event.key)

            elif event.type == pygame.KEYUP:  # keyboard key released
                if (event.key,) in self.keys2actions.keys():
                    self.current_keys_down.remove(event.key)

                # elif event.key == pygame.K_f:  # 'F': fast forward
                #     self.fast_forward = False

    def _render(self):
        self.window.fill((20, 20, 20))  # clear the logical surface
        self._render_env()
        if self.render_predicate_probs:
            self._render_predicate_probs()

        # Scale the logical surface to the physical (display-fitted) window.
        if self.window.get_size() == self.screen.get_size():
            self.screen.blit(self.window, (0, 0))
        else:
            self.screen.blit(
                pygame.transform.smoothscale(self.window, self.screen.get_size()), (0, 0))
        pygame.display.flip()
        pygame.event.pump()
        if not self.fast_forward:
            self.clock.tick(self.fps)

    def _render_env(self):
        frame = self.env.env.render()
        frame_surface = pygame.Surface(self.env_render_shape)
        pygame.pixelcopy.array_to_surface(frame_surface, frame)
        self.window.blit(frame_surface, (0, 0))

    def _render_predicate_probs(self):
        """Rule panel: every clause with its probability, and below each the body
        atoms of its winning grounding (the ground predicates that produce that
        probability). The executed clause is pinned to the top and highlighted.
        Flows into a second column when it overflows the window height."""
        attributions = rule_attributions(self.nsfr_reasoner)
        if not attributions:
            self._render_predicate_probs_fallback()
            return

        taken_head = getattr(self, "taken_head", None)
        ordered = order_for_display(attributions, taken_head) if taken_head else list(attributions)
        taken_clause_idx = (
            ordered[0].clause_idx
            if taken_head and ordered and ordered[0].head == taken_head
            else None
        )

        base_x = self.env_render_shape[0] + 10
        col_width = PREDICATE_PROBS_COL_WIDTH
        top_y, header_h, body_h = 25, 30, 20
        max_y = self.env_render_shape[1] - body_h
        small_font = pygame.font.SysFont("Calibri", 16)

        col, y = 0, top_y
        for attr in ordered:
            block_h = header_h + body_h * len(attr.body) + 6
            if col < NUM_PANEL_COLUMNS - 1 and y + block_h > max_y:
                col, y = col + 1, top_y
            x = base_x + col * col_width

            is_taken = attr.clause_idx == taken_clause_idx
            p = float(np.clip(attr.prob, 0.0, 1.0))
            if is_taken:
                color = CELL_BACKGROUND_SELECTED
            else:
                color = p * CELL_BACKGROUND_HIGHLIGHT + (1 - p) * CELL_BACKGROUND_DEFAULT
            color = np.clip(np.asarray(color), 0, 255).astype(int).tolist()
            pygame.draw.rect(self.window, color, [x - 2, y - 2, col_width - 12, 26])

            label = f"{attr.prob:.2f}  {attr.head}" + ("  [TAKEN]" if is_taken else "")
            self.window.blit(self.font.render(label, True, "white", None), (x, y))
            y += header_h

            for atom_str, val in attr.body:
                line = small_font.render(f"   {atom_str} = {val:.2f}", True, "white", None)
                self.window.blit(line, (x, y))
                y += body_h
            y += 6

    def _render_predicate_probs_fallback(self):
        """Old per-predname panel, used only when attribution data is unavailable
        (e.g. before the first forward pass populates the valuation)."""
        anchor = (self.env_render_shape[0] + 10, 25)
        nsfr = self.nsfr_reasoner
        pred_vals = {pred: nsfr.get_predicate_valuation(pred, initial_valuation=False)
                     for pred in nsfr.prednames}
        if not pred_vals:
            return
        i_max = int(np.argmax(list(pred_vals.values())))
        for i, (pred, val) in enumerate(pred_vals.items()):
            if i == i_max:
                color = CELL_BACKGROUND_SELECTED
            else:
                color = val * CELL_BACKGROUND_HIGHLIGHT + (1 - val) * CELL_BACKGROUND_DEFAULT
            color = np.clip(np.asarray(color), 0, 255).astype(int).tolist()
            pygame.draw.rect(self.window, color, [
                anchor[0] - 2, anchor[1] - 2 + i * 35, PREDICATE_PROBS_COL_WIDTH - 12, 28])
            text = self.font.render(f"{100*val:.2f} - {pred}", True, "white", None)
            self.window.blit(text, (self.env_render_shape[0] + 10, 25 + i * 35))
