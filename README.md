# Gaze-Based Neurosymbolic Imitation Learner (GBNIL)

A framework for training a Neuro-Symbolic Forward Reasoning (NSFR) agent on Seaquest via imitation learning from human expert demonstrations, optionally exploiting gaze data as privileged information.

---

## Table of Contents
1. [Setup](#1-setup)
2. [Create a Dataset (.pt)](#2-create-a-dataset-pt)
3. [Train a Gaze Prediction Model](#3-train-a-gaze-prediction-model)
4. [Train a Probabilistic Facts Prediction Model](#4-train-a-probabilistic-facts-prediction-model)
5. [Train the NSFR Model](#5-train-the-nsfr-model)
6. [Run Baselines (BC / AGIL)](#6-run-baselines-bc--agil)
7. [Visualize Gaze on Game Video](#7-visualize-gaze-on-game-video)
8. [Interactive GUI Play (`play_il_gui`)](#8-interactive-gui-play-play_il_gui)
9. [Other Important Scripts](#9-other-important-scripts)

---

## 1. Setup

### Clone and install dependencies

```bash
git clone <repo-url>
cd Gaze-Based-Neurosymbolic-Imitation-Learner
git checkout refactor   # or main / v2
```

Install core libraries as editable packages (required for imports to work):

```bash
pip install -e core/nsfr/
pip install -e core/nudge/
pip install -e core/ocatari/
```

Install project dependencies:

```bash
pip install -r requirements.txt
# or
pip install -e .
```

> **Note:** Atari ROMs are required. `gymnasium[atari,accept-rom-license]` in requirements handles this automatically.

### Directory overview

```
core/          Library code (nsfr, nudge, ocatari, envs, config, baselines, models)
data/          Expert datasets (.pt files, CSV trajectories, gaze masks)
scripts/       All runnable scripts (training, evaluation, preprocessing, shell scripts)
tests/         Evaluation and debugging scripts
results/       CSV result files from training runs
logs/          Training log directories (saved at runtime)
```

---

## 2. Create a Dataset (.pt)

The dataset is created from raw trajectory folders that contain PNG game frames and a CSV/TXT gaze+action file per episode.

### Expected input structure

```
data/seaquest/trajectories/
├── episode_001/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── gaze_actions.csv      # columns: frame_id, episode_id, score, duration,
│                             #           unclipped_reward, action, gaze_x0, gaze_y0, ...
├── episode_002/
│   └── ...
```

### Convert trajectories to `.pt`

```bash
cd scripts
python convert_trajectories_to_pt.py \
    --traj_dir ../data/seaquest/trajectories \
    --output   ../data/seaquest/dataset.pt \
    --gaze_sigma 5.0           # Gaussian sigma for 84x84 gaze heatmap
    # --max_episodes 20        # optional: limit number of episodes
```

### Output format

The `.pt` file is a dict containing:

| Key | Shape | Description |
|---|---|---|
| `observations` | `(N, 84, 84)` uint8 | Grayscale game frames |
| `gaze_information` | `(N, 3)` float64 | `[x_norm, y_norm, global_step]` |
| `gaze_image` | `(N, 84, 84)` float32 | Gaussian gaze heatmap |
| `logic_state` | `(N, 47, 5)` int32 | Symbolic object state (OC-Atari vision) |
| `actions` | `(N,)` int32 | Expert action index per frame |
| `episode_number` | `(N,)` int32 | Which episode each frame belongs to |
| `episode-rewards` | `(N,)` float64 | Reward per frame |
| `terminateds` | `(N,)` bool | True when episode ended |
| `steps` | `(E,)` int32 | Number of steps per episode |

---

## 3. Train a Gaze Prediction Model

The gaze predictor (`Human_Gaze_Predictor`) is a CNN that learns to predict where a human player is looking given a stack of game frames. It is used at inference time to supply gaze information when playing without a human observer.

```bash
cd scripts
python gaze_predictor.py \
    --train \
    --game_name seaquest \
    --dataset   ../data/seaquest/dataset.pt \
    --gaze_masks ../data/seaquest/gaze_masks.pt \
    --epochs    20 \
    --batch_size 64 \
    --lr        1.0 \
    --frame_stack 4            # number of frames stacked as input
    # --model_weights path/to/existing.pth   # optional: fine-tune from checkpoint
```

The trained model is saved as a `.pth` file (default: `seaquest_gaze_predictor.pth`).

To use pre-generated gaze masks (recommended for speed):

```bash
# Generate gaze masks first
python generate_full_gaze_tensor.py \
    --dataset ../data/seaquest/dataset.pt \
    --output  ../data/seaquest/gaze_masks.pt
```

---

## 4. Train a Probabilistic Facts Prediction Model

Pre-computing valuations (atom probabilities from symbolic states) is optional but **highly recommended** — it caches the NSFR forward pass per frame, making training ~10× faster.

```bash
cd scripts
python preprocess_dataset.py \
    --dataset_path ../data/seaquest/dataset.pt \
    --env          seaquest \
    --output       train_atoms.pkl \
    --device       cuda

# With gaze (uses gaze predictor to compute gaze-conditioned valuations):
python preprocess_dataset.py \
    --dataset_path  ../data/seaquest/dataset.pt \
    --env           seaquest \
    --use_gaze \
    --gaze_threshold 50.0 \
    --gaze_model_path core/models/gaze_predictor/seaquest_gaze_predictor_sigma_10.pth \
    --output        train_atoms_gaze.pkl \
    --device        cuda
```

The output `.pkl` is a dict mapping `frame_id → atom_probs_tensor` and is loaded automatically by `train_il.py` if placed at `models/nsfr/seaquest/{gaze|_no_gaze}/valuation.pt`.

---

## 5. Train the NSFR Model

The main training script. All scripts must be run from the **project root**.

### Without gaze (baseline symbolic IL)

```bash
python scripts/train_il.py \
    --env     seaquest \
    --rules   new \
    --dataset data/seaquest/dataset.pt \
    --epochs  50 \
    --lr      0.001 \
    --batch_size 128 \
    --device  cuda \
    --eval_interval 5
```

### With gaze (privileged information)

```bash
python scripts/train_il.py \
    --env           seaquest \
    --rules         new \
    --dataset       data/seaquest/dataset.pt \
    --use_gazemap \
    --gaze_model_path core/models/gaze_predictor/seaquest_gaze_predictor_2.pth \
    --gaze_threshold  50.0 \
    --epochs          50 \
    --lr              0.001 \
    --batch_size      128 \
    --device          cuda \
    --eval_interval   5
```

> `--use_gazemap` activates the live gaze predictor for both training-time valuation and evaluation. Without it, gaze heatmaps from the `.pt` expert data are used (`--use_gaze`).

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | — | Path to `.pt` dataset |
| `--rules` | `new` | Ruleset under `core/envs/seaquest/logic/` |
| `--epochs` | `16` | Training epochs |
| `--lr` | `0.001` | Learning rate |
| `--batch_size` | `32` | Batch size |
| `--num_episodes` | all | Limit episodes loaded (sample efficiency) |
| `--sort_by` | — | Sort episodes by `length` or `reward_per_step` |
| `--valuation_path` | auto | Path to pre-computed `valuation.pt` (skip if auto-detected) |
| `--eval_interval` | `5` | Evaluate in-game every N epochs |
| `--eval_max_steps` | `10000` | Max game steps per eval episode |
| `--send_email` | off | Send periodic email updates during training |

Saved checkpoints: `core/models/nsfr/seaquest/{gaze|no_gaze}/{N}_ep/epoch_X.pth`

### Evaluating a trained NSFR model

```bash
python tests/evaluate_model.py \
    --env       seaquest \
    --rules     new \
    --model     core/models/nsfr/seaquest/gaze/full_ep/best.pth \
    --num_episodes 10
```

---

## 6. Run Baselines (BC / AGIL)

Plain Behavior Cloning and AGIL (Attention-Guided IL) baselines.

### BC / AGIL from `.pt` dataset

```bash
python scripts/train_bc_pt.py \
    --dataset     data/seaquest/dataset.pt \
    --env         seaquest \
    --epochs      50 \
    --batch_size  256 \
    --lr          1e-3 \
    --device      cuda

# AGIL: add gaze mask overlay
python scripts/train_bc_pt.py \
    --dataset     data/seaquest/dataset.pt \
    --env         seaquest \
    --gaze_method mask \
    --epochs      50 \
    --device      cuda
```

`--gaze_method` options: `None` (plain BC), `mask` (AGIL — multiplies gaze mask onto frames).

### Incremental (sample efficiency) sweep

```bash
# Train separate models for N=1,2,...,10 episodes
bash scripts/run_bc_mask_incremental.sh
bash scripts/run_sample_efficiency_sweep.sh         # NSFR no-gaze
bash scripts/run_sample_efficiency_sweep_no_gaze.sh # BC no-gaze
```

### Evaluate a BC/AGIL model

```bash
python tests/evaluate_bc_model.py \
    --model  core/models/bc/seaquest/best.pth \
    --env    seaquest \
    --num_episodes 10 \
    --device cuda
```

---

## 7. Visualize Gaze on Game Video

Generate a side-by-side video comparing ground-truth gaze heatmaps vs. model-predicted gaze heatmaps.

```bash
cd scripts
python compare_gaze_predictions.py \
    --dataset      ../data/seaquest/dataset.pt \
    --game_name    seaquest \
    --model_weights ../core/models/gaze_predictor/seaquest_gaze_predictor_2.pth \
    --output       ../gaze_comparison.mp4 \
    --num_frames   3000 \
    --multiplier   2.0    # upscale factor for output resolution
```

To visualize gaze segmentation and goal-extraction from raw trajectories:

```bash
python visualize_gaze_segmentation.py \
    --data_dir data/seaquest/trajectories/episode_001 \
    --json_file skill_discovery/gaze_segments.json
```

---

## 8. Interactive GUI Play (`play_il_gui`)

Watch the trained NSFR agent play the game in real time with a visual overlay showing rule probabilities.

```bash
python scripts/play_il_gui.py \
    --env    seaquest \
    --rules  new \
    --model  core/models/nsfr/seaquest/gaze/full_ep/best.pth
    # --use_gazemap    # enable live gaze predictor during play
    # --gaze_model_path core/models/gaze_predictor/seaquest_gaze_predictor_2.pth
```

The GUI shows:
- The live game render
- Per-rule probability bars (color-coded: yellow = active rule)
- The currently selected action and predicate clause

---

## 9. Other Important Scripts

| Script | Location | Purpose |
|---|---|---|
| `precompute_valuations.py` | `scripts/` | Batch-precompute NSFR atom probabilities (faster training) |
| `evaluate_model.py` | `tests/` | Evaluate a trained NSFR agent in the game |
| `evaluate_bc_model.py` | `tests/` | Evaluate a trained BC/AGIL agent |
| `eval_loop.py` | `tests/` | Sweep evaluation across multiple saved checkpoints |
| `eval_fewer_objs.py` | `tests/` | Evaluate generalization with fewer objects in the game |
| `compare_valuation_models.py` | `scripts/` | Compare two trained NSFR models' clause weights |
| `compare_valuation_preds.py` | `scripts/` | Compare predicate probabilities between two models |
| `inspect_env.py` | `scripts/` | Inspect raw game objects and logic states frame-by-frame |
| `mine_rules.py` | `scripts/` | Mine new logic rules via beam search |
| `atari_env_manager.py` | `scripts/` | Utility for managing Atari ROM environments |
| `generate_valuation_atoms.py` | `scripts/` | Generate atom valuations for a single episode |

### Rulesets

Logic rules live in `core/envs/seaquest/logic/`. Each subfolder is a ruleset:

```
core/envs/seaquest/logic/
├── new/          # Primary ruleset (used by default)
│   ├── clauses.txt
│   ├── bk.txt
│   ├── preds.txt
│   ├── consts.txt
│   └── neural_preds.txt
└── ...
```

To use a different ruleset: `--rules <subfolder_name>`.

### Gaze valuation

The core gaze mechanism lives in `core/envs/seaquest/valuation.py`. It computes normalized gaze attention per symbolic object using `gaze_object_attention_normalized()`, which:
1. Sums the gaze heatmap mass inside each object's bounding box
2. Normalizes across all present objects (so attention sums to 1)
3. Clips to a minimum of `0.1` so unattended objects remain partially visible

This scored attention directly modulates predicate probabilities like `visible_enemy`, `visible_diver`, etc.
