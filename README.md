# Gaze-Based Neurosymbolic Imitation Learner (grail)

A framework for training a Neuro-Symbolic Forward Reasoning (NSFR) agent on Atari Seaquest via imitation learning from human expert demonstrations, with optional gaze data as privileged information.

---

## Table of Contents
1. [Setup](#1-setup)
2. [Project Structure](#2-project-structure)
3. [Create a Dataset (.pt)](#3-create-a-dataset-pt)
4. [Train the NSFR Model](#4-train-the-nsfr-model)
5. [Train a Gaze Prediction Model](#5-train-a-gaze-prediction-model)
6. [Run Baselines (BC / AGIL)](#6-run-baselines-bc--agil)
7. [Evaluate a Model](#7-evaluate-a-model)
8. [Visualize & Play](#8-visualize--play)
9. [Key Design Notes](#9-key-design-notes)

---

## 1. Setup

### Clone and install

```bash
git clone <repo-url>
cd Gaze-Based-Neurosymbolic-Imitation-Learner
git checkout refactor   # Moved to mai later

conda create -n grail -python=3.9
pip install -r requirements.txt
pip install -e .        # installs all deps + registers grail-* CLI commands
```

> **Note:** Atari ROMs are installed automatically via `gymnasium[atari,accept-rom-license]`.

### Running scripts

Scripts must be run in one of two ways only (both require `pip install -e .` first):

```bash
# Option 1 — CLI entry points (recommended)
grail-train --env seaquest --rules new --dataset data/seaquest/dataset.pt

# Option 2 — Python module mode from the project root
python -m scripts.training.train_il --env seaquest --rules new --dataset data/seaquest/dataset.pt
```

> **Do not** run scripts directly as `python scripts/training/train_il.py` — imports will fail
> because the project root won't be in `sys.path`.

### Available CLI commands

After `pip install -e .`, the following commands are available in your shell:

| Command | Script |
|---|---|
| `grail-train` | `scripts/training/train_il.py` |
| `grail-train-bc` | `scripts/training/train_bc.py` |
| `grail-train-bc-pt` | `scripts/training/train_bc_pt.py` |
| `grail-train-gaze` | `scripts/gaze/train_gaze_predictor_gabril.py` |
| `grail-eval` | `scripts/evaluation/evaluate_model.py` |
| `grail-eval-bc` | `scripts/evaluation/evaluate_bc_model.py` |
| `grail-convert` | `scripts/preprocess/convert_trajectories_to_pt.py` |
| `grail-preprocess` | `scripts/preprocess/preprocess_dataset.py` |
| `grail-precompute` | `scripts/preprocess/precompute_valuations.py` |
| `grail-gen-atoms` | `scripts/preprocess/generate_valuation_atoms.py` |
| `grail-play` | `scripts/play/play_il_gui.py` |
| `grail-visualize` | `scripts/visualization/visualize_trajectory.py` |

---

## 2. Project Structure

```
.
├── core/
│   ├── nsfr/           # Neuro-Symbolic Forward Reasoner (logic engine, agents, env)
│   ├── ocatari/        # Object-Centric Atari (OC-Atari) RAM extraction
│   ├── envs/           # Per-game environment wrappers & NSFR logic rules
│   ├── utils/          # Shared utilities (gaze, CNN models, email, etc.)
│   └── baselines/      # Baseline CNN model implementations
├── data/               # Expert datasets (.pt files, CSV trajectories, gaze masks)
├── scripts/            # All runnable scripts (see scripts/README.md)
│   ├── training/
│   ├── evaluation/
│   ├── preprocess/
│   ├── gaze/
│   ├── visualization/
│   ├── play/
│   ├── search/
│   ├── baselines/
│   └── shell/
├── tests/              # Debugging and evaluation test scripts
├── results/            # CSV result files from training runs
├── trained_models/     # Saved model checkpoints
├── logs/               # Training log directories
├── setup.py            # Single-file install (all packages + CLI entry points)
└── requirements.txt    # Full dependency list
```

---

## 3. Create a Dataset (.pt)

Convert raw expert-play trajectories (PNG frames + CSV gaze/action file) into a self-contained `.pt` dataset:

```bash
grail-convert \
    --traj_dir data/seaquest/trajectories \
    --output   data/seaquest/dataset.pt \
    --gaze_sigma 5.0
```

The `.pt` file contains: `observations`, `gaze_information`, `gaze_image`, `logic_state`, `actions`, `episode_number`, `terminateds`, `steps`.

---

## 4. Train the NSFR Model

### Without gaze (baseline symbolic IL)

```bash
grail-train \
    --env seaquest --rules new \
    --dataset data/seaquest/dataset.pt \
    --epochs 50 --lr 0.001 --batch_size 128 \
    --device cuda --eval_interval 5
```

### With expert gaze (privileged training)

```bash
grail-train \
    --env seaquest --rules new \
    --dataset data/seaquest/dataset.pt \
    --use_gaze \
    --epochs 50 --lr 0.001 --device cuda
```

### With live gaze predictor

```bash
grail-train \
    --env seaquest --rules new \
    --dataset data/seaquest/dataset.pt \
    --use_gazemap \
    --gaze_model_path trained_models/gaze_predictor/seaquest_gaze_predictor.pth \
    --gaze_threshold 50.0 \
    --epochs 50 --device cuda
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | — | Path to `.pt` dataset |
| `--rules` | `new` | Ruleset under `core/envs/seaquest/logic/` |
| `--epochs` | `16` | Training epochs |
| `--lr` | `0.001` | Learning rate |
| `--batch_size` | `32` | Batch size |
| `--num_episodes` | all | Limit episodes loaded (for sample-efficiency studies) |
| `--use_gaze` | off | Use gaze heatmaps from the `.pt` dataset during training |
| `--use_gazemap` | off | Use a live gaze predictor model at training & eval time |
| `--valuation_path` | auto | Path to pre-computed `valuation.pt` |
| `--eval_interval` | `5` | Evaluate in-game every N epochs |
| `--send_email` | off | Send periodic email updates during training |

Checkpoints saved to: `trained_models/nsfr/seaquest/{gaze|no_gaze}/{N}_ep/`

---

## 5. Train a Gaze Prediction Model

The `HumanGazeNet` CNN learns to predict human fixation heatmaps from stacked game frames.

```bash
grail-train-gaze \
    --game_name seaquest \
    --dataset   data/seaquest/dataset.pt \
    --epochs    30 --batch_size 64 --lr 1e-3
```

Pre-compute gaze valuations (strongly recommended for speed — ~10× faster training):

```bash
grail-precompute \
    --dataset  data/seaquest/dataset.pt \
    --env      seaquest \
    --use_gaze \
    --gaze_model_path trained_models/gaze_predictor/seaquest_gaze_predictor.pth \
    --output   trained_models/nsfr/seaquest/gaze/valuation.pt \
    --device   cuda
```

---

## 6. Run Baselines (BC / AGIL)

```bash
# Plain Behavior Cloning
grail-train-bc-pt \
    --dataset data/seaquest/dataset.pt \
    --env seaquest --epochs 50 --batch_size 256 --lr 1e-3 --device cuda

# AGIL — gaze-mask overlay
grail-train-bc-pt \
    --dataset data/seaquest/dataset.pt \
    --env seaquest --gaze_method mask --epochs 50 --device cuda
```

Sweep experiments:

```bash
bash scripts/shell/run_bc_mask_incremental.sh
bash scripts/shell/run_sample_efficiency_sweep.sh
```

---

## 7. Evaluate a Model

```bash
# NSFR agent
grail-eval \
    --env seaquest --rules new \
    --model trained_models/nsfr/seaquest/gaze/full_ep/best.pth \
    --num_episodes 10

# BC / AGIL agent
grail-eval-bc \
    --model trained_models/bc/seaquest/best.pth \
    --env seaquest --num_episodes 10 --device cuda
```

---

## 8. Visualize & Play

```bash
# Watch the NSFR agent play with live rule-probability overlay
grail-play \
    --env seaquest --rules new \
    --model trained_models/nsfr/seaquest/gaze/full_ep/best.pth

# Replay a recorded trajectory
grail-visualize --dataset data/seaquest/dataset.pt --episode 0

# Side-by-side gaze comparison video
python scripts/gaze/compare_gaze_predictions.py \
    --dataset data/seaquest/dataset.pt \
    --model_weights trained_models/gaze_predictor/seaquest_gaze_predictor.pth \
    --output gaze_comparison.mp4
```

---

## 9. Key Design Notes

### Rulesets

Logic rules live in `core/envs/seaquest/logic/`. Each subfolder is a ruleset:

```
core/envs/seaquest/logic/
├── new/          # Primary ruleset (used by default with --rules new)
│   ├── clauses.txt
│   ├── bk.txt
│   ├── preds.txt
│   ├── consts.txt
│   └── neural_preds.txt
```

To use a different ruleset: `--rules <subfolder_name>`.

### Gaze valuation

The gaze mechanism lives in `core/envs/seaquest/valuation.py`. It computes normalized gaze attention per symbolic object:
1. Sums the gaze heatmap mass inside each object's bounding box
2. Normalizes across all present objects (attention sums to 1)
3. Clips to a minimum of `0.1` so unattended objects remain partially visible

This modulates predicate probabilities like `visible_enemy`, `visible_diver`, etc.

### Module imports

The project uses a single `pip install -e .` (via `setup.py`) which registers `core/nsfr`, `core/ocatari`, and all `scripts/` subpackages. Scripts add `core/` to `sys.path` for legacy compatibility.
