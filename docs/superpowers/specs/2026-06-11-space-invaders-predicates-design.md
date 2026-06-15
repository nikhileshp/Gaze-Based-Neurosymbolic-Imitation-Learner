# Space Invaders Input Representation & Neural Predicates — Design

**Date:** 2026-06-11
**Branch:** `feature/space-invaders-predicates` (off `refactor`)
**Status:** Proposed (pending user approval)

## Problem

The repo wants a neurosymbolic Space Invaders env consistent with freeway/seaquest:
an object-centric logic state + differentiable neural predicates over it. `main`
has a draft `in/envs/space_invaders/valuation.py` + predicate declarations, but
the env cannot run:

- **No `env.py`** → `NSFRBaseEnv.from_name("space_invaders")` falls back to
  `core/envs/space_invaders/env.py` and raises `FileNotFoundError` (confirmed by
  running `play_il_gui.py -g space_invaders`).
- **Incomplete logic dir** — missing `consts.txt` and `bk.txt`.
- **Non-standard typing** — the draft predicates use per-category arg types
  (`oplayer, oalien, …`), which `ValuationModule.ground_to_tensor` cannot resolve
  to object slots (it only handles `obj\d+` names or `dtype=='object'`). Every
  working env (freeway, seaquest, getout) uses the uniform `object` scheme.

## Goal (this round)

A **good input representation**: an `extract_logic_state` + neural predicate set
that runs end-to-end and is validated visually via `play_il_gui.py` over VNC
(untrained model — the GUI explicitly supports "Running with untrained model"),
showing OCAtari object boxes + the live firing-predicate panel. Iterate the
representation until objects and predicates look correct.

## Out of scope

Real behavioral `init_clauses` (action←predicate rules) and any training. The
placeholder rules (`action(X):-in_image,in_image`) stay; they don't affect the
neural-predicate values shown in the GUI.

## Grounded facts (from OCAtari probe, grail env)

- Action meanings: `['NOOP','FIRE','RIGHT','LEFT','RIGHTFIRE','LEFTFIRE']`.
- `env.objects` is a fixed list of **44** slots in stable order: Player(0),
  Shield(1–3), Bullet(4–6), Satellite(7), Alien(8–43). Absent entries are a
  `NoObject` placeholder with `x=y=w=h=0`.
- Native 160×210 coords, top-left `(x,y)` + `(w,h)`. Player y≈185, Aliens y≈31
  (descend over time), Bullets w=1. `ram` and `vision` modes both yield the
  44-slot structure (live alien count varies; rest are `NoObject`).

## Approach (chosen): uniform `object` scheme + internal type gating

Reuse `main`'s geometric predicate bodies, but type the predicate args as
`object` (not category) and gate each predicate internally on a one-hot type
block in the object row. No changes to core NSFR infrastructure; consistent with
all other envs.

### Logic-state layout — `(44, 6)` int tensor

Per-object row: `[present, x, y, w, h, type_id]` (seaquest convention). Position
block (0–4) matches `main`'s valuation (`_cx=o[1]+o[3]/2`, `_cy=o[2]+o[4]/2`);
`type_id` (index 5) is an int matching the `type:` consts order
(player=0, alien=1, shield=2, bullet=3, satellite=4). The standard
`type(O, type_oh)` predicate compares `O[5] == type_oh.argmax()` (exactly
seaquest's), and geometric predicates gate on category via `_is(o, id)` helpers.

`extract_logic_state(raw_state)` uses category offsets (like seaquest):
`MAX_NB_OBJECTS` (Player1,Shield3,Bullet3,Satellite1,Alien36) → slot offsets
Player 0, Shield 1, Bullet 4, Satellite 7, Alien 8; each object goes to
`offset[cat] + count[cat]`. `NoObject`/irrelevant categories are skipped
(row stays `present=0`).

### Components

| File | Action | Responsibility |
|------|--------|----------------|
| `in/envs/space_invaders/env.py` | **create** | `NSFREnv(NSFRBaseEnv)`: `name="space_invaders"`, `pred2action={noop:0,fire:1,move_right:2,move_left:3,fire_right:4,fire_left:5}`, OCAtari `ALE/SpaceInvaders-v5` (mode from `oc_mode`, `render_oc_overlay` passthrough), `n_objects=44`, `n_features=10`, `extract_logic_state`, `extract_neural_state`, `get_rgb_frame` (`unwrapped.ale.getScreenRGB`). Modeled on `core/envs/seaquest/env.py`. |
| `in/envs/space_invaders/valuation.py` | **graft + adapt** | `main`'s predicates, retyped to `object` args, each gated on the type one-hot. Add `type(O, category)`. Keep the geometric logic + thresholds (`ALIGN_PX`, etc.) as tunable module constants. |
| `…/logic/space_invaders_root/consts.txt` | **create** | `object:obj0..obj43` / `type:player,alien,shield,bullet,satellite` / `image:img` (type order matches the one-hot column order). |
| `…/neural_preds.txt` | **adapt** | retype all args to `object`; add `type:2:object,type`. |
| `…/preds.txt`, `…/init_clauses.txt` | keep | placeholder action rules (out of scope). |
| `…/bk.txt` | **create** | background facts as required by the language loader (mirror seaquest's `bk.txt`). |
| `scripts/diagnostics/si_check_state.py` | **create** | headless: run the SI env N steps (grail env), print each frame's populated slots (slot, category, present, x,y,w,h) and the firing neural predicates (val>0.5), as a fast cross-check independent of the GUI. |

### Neural predicate set (adapted from main; all gated on type)

Unary existence: `visible_alien`, `visible_bullet`, `visible_satellite`.
Player↔alien: `left_of_alien`, `right_of_alien`, `aligned_with_alien` (smooth),
`slightly_left_of_alien`, `slightly_right_of_alien`. Player↔satellite:
`aligned_with_satellite`. Threats: `close_by_bullet` (smooth),
`bullet_above_player`, `bullet_left_of_player`, `bullet_right_of_player`,
`bullet_aligned_player`, `bullet_threatens_shield`, `bullet_above_alien`.
Defense: `behind_shield`. Plus generic `type(O, category)`.

Start with all 44 slots (Player/Shield/Bullet/Satellite + 36 aliens). Seaquest
(47 objects) already runs in `play_il_gui` in real time, so 44 is tractable. If
the GUI is sluggish, the first pruning step is to cap/reduce alien slots (e.g.
front alien per column) — decided from the visual iteration.

## Data flow / iteration loop

1. OCAtari objects → `extract_logic_state` → `(44,10)` logic state.
2. `ImitationAgent('space_invaders','space_invaders_root')` (untrained) →
   FactsConverter runs `valuation.py` → neural predicate values (V_0).
3. `play_il_gui.py` over VNC renders OCAtari **boxes** + the **firing-predicate
   panel**; keyboard takeover lets us drive specific scenarios.
4. Observe → edit `extract_logic_state` / `valuation.py` thresholds → relaunch →
   repeat. `si_check_state.py` gives a fast headless cross-check.

**Launch (grail env, matches prior VNC flow):**
```
./scripts/play/remote_gui.sh serve
# laptop: ssh -L 5900:localhost:5900 nick@<server> ; vncviewer localhost:5900
./scripts/play/remote_gui.sh /home/nick/miniconda3/envs/grail/bin/python \
    scripts/play/play_il_gui.py -g space_invaders -r space_invaders_root --oc_mode vision
```

## Validation

- `si_check_state.py` asserts: Player in slot 0 with `present=1` and y≈185; alien
  slots (8+) populate with y near the top early; all coords within `[0,160]×[0,210]`;
  existence/`type` predicates fire for the right categories.
- Visual confirmation in VNC: boxes align with sprites; the predicate panel shows
  sensible atoms (e.g. `aligned_with_alien(obj0, objK)` when the player sits under
  an alien column).
- Env build smoke test: `from_name("space_invaders")` + one `reset/step` returns
  a `(44,10)` state without error (run with grail env).

## Testing

Lightweight, run with the grail env (full stack):
- `tests/test_space_invaders_env.py`: `extract_logic_state` shape `(44,10)`,
  player slot/type, `NoObject`→present 0, coords in-range, `pred2action` matches
  `get_action_meanings()` order.
- A predicate sanity check: on a hand-built state (player under an alien), the
  right neural predicates exceed 0.5 and wrong-category pairs stay ~0.01.

## Notes

- Environment: use **`grail`** for the GUI/env (full stack incl. pygame +
  HackAtari + OCAtari); `sam3` for any pure nsfr unit tests.
- Branch isolation: all work on `feature/space-invaders-predicates`; `refactor`
  stays at `6884e3c`.
