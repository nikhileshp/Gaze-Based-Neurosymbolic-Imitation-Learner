# Scripts

This directory contains all executable scripts for the **Gaze-Based Neurosymbolic Imitation Learner (grail)** project, organised into thematic subfolders.

Each subfolder is a Python package (`__init__.py` present). Scripts must be run
in one of two ways after `pip install -e .` from the project root:

```bash
# Option 1 — CLI entry points (recommended)
grail-train --env seaquest --rules new --dataset data/seaquest/dataset.pt --epochs 50
grail-eval  --env seaquest --rules new --model trained_models/nsfr/seaquest/gaze/full_ep/best.pth

# Option 2 — Python module mode (from project root)
python -m scripts.training.train_il --env seaquest ...
```

> **Do not** invoke scripts directly as `python scripts/<subfolder>/<script>.py` —
> the project root will not be in `sys.path` and imports will fail.

---

## Folder Overview

| Folder | Purpose |
|---|---|
| [`training/`](#-training) | Model training scripts |
| [`evaluation/`](#-evaluation) | Model evaluation & comparison |
| [`preprocess/`](#-preprocess) | Data collection, conversion & preprocessing |
| [`gaze/`](#-gaze) | Gaze processing, prediction & analysis |
| [`visualization/`](#-visualization) | Trajectory & gaze visualizations |
| [`play/`](#-play) | Interactive human play & environment wrappers |
| [`search/`](#-search) | Rule / clause search algorithms |
| [`baselines/`](#-baselines) | Baseline model architectures & utilities |
| [`shell/`](#-shell) | Bash scripts for running experiments |

---

## 🧠 training/

Training scripts for the neurosymbolic IL pipeline and neural baselines.

| File | CLI alias | Description |
|---|---|---|
| `train_il.py` | `grail-train` | **Main** NSFR imitation learning trainer. Supports `.pt` dataset, gaze-guided training, and per-epoch evaluation. |
| `train_bc.py` | `grail-train-bc` | Behaviour Cloning (GABRIL-style) with multiple gaze methods (BC, AGIL, ViSaRL, Mask, Reg, Contrastive, GRIL). |
| `train_bc_pt.py` | `grail-train-bc-pt` | Simplified BC training directly from a `.pt` dataset. |
| `train_valuation_cnn.py` | — | Train a CNN to regress NSFR atom valuations from raw pixel observations. |

### Key arguments — `train_il.py`

```bash
python scripts/training/train_il.py \
  --env seaquest --rules new \
  --dataset data/seaquest/<file>.pt \
  --epochs 50 --lr 0.001 \
  --use_gaze --use_gazemap \
  --num_episodes 28
```

---

## 📊 evaluation/

Scripts for evaluating and comparing trained models.

| File | CLI alias | Description |
|---|---|---|
| `evaluate_model.py` | `grail-eval` | Evaluate a trained NSFR agent in-game over N episodes. |
| `evaluate_bc_model.py` | `grail-eval-bc` | Evaluate a trained BC/AGIL agent in-game. |
| `eval_loop.py` | — | Sweep evaluation across multiple saved checkpoints. |
| `eval_fewer_objs.py` | — | Evaluate generalization with fewer objects in the game. |
| `show_rules.py` | — | Print learned NSFR symbolic rules from a trained model. |

---

## 💾 preprocess/

Data collection, format conversion and preprocessing utilities.

| File | CLI alias | Description |
|---|---|---|

| `convert_trajectories_to_pt.py` | `grail-convert` | Convert raw CSV+image trajectories into a single self-contained `.pt` dataset file. |
| `precompute_valuations.py` | `grail-precompute` | Pre-compute NSFR atom valuations over a dataset and cache to disk. |
| `generate_valuation_atoms.py` | `grail-gen-atoms` | Generate ground-truth atom probabilities from a `.pt` dataset using the NSFR engine. |


---

## 👁️ gaze/

Everything related to gaze data processing, prediction, and analysis.

| File | CLI alias | Description |
|---|---|---|
| `gaze_predictor.py` | — | `HumanGazeNet` CNN model and trainer. Entry point for training gaze heatmap predictors. |
| `train_gaze_predictor_gabril.py` | `grail-train-gaze` | Train a GABRIL-style AutoEncoder-based gaze predictor. |
| `extract_gaze_goals.py` | — | Segment gaze trajectories into goal-directed episodes based on fixation patterns. |
| `compare_gaze_predictions.py` | — | Produce a side-by-side video comparing ground-truth vs. predicted gaze heatmaps. |
| `cluster_goals.py` | — | Cluster gaze goal segments using K-Means to discover high-level behaviour patterns. |
| `label_segments.py` | — | Assign human-readable labels to gaze clusters. |
| `mine_rules.py` | — | Mine PDDL-style skill preconditions/effects from labelled gaze segments. |

### Key arguments — `gaze_predictor.py`

```bash
python scripts/gaze/gaze_predictor.py \
  --game_name seaquest \
  --dataset data/seaquest/<file>.pt \
  --epochs 30 --batch_size 64
```

---

## 🎥 visualization/

Scripts for visualizing game trajectories, gaze data, and model behaviour.

| File | CLI alias | Description |
|---|---|---|
| `visualize_pt.py` | — | Replay frames from a `.pt` dataset with optional gaze heatmap overlay. |
| `visualize_trajectory.py` | `grail-visualize` | Replay a recorded trajectory video with overlaid game state info. |
| `visualize_gaze_segmentation.py` | — | Visualize gaze segmentation results on game frames. |
| `make_trajectory_video.py` | — | Export a game trajectory as an MP4 video. |

---

## 🎮 play/

Interactive play scripts and environment wrappers.

| File | CLI alias | Description |
|---|---|---|
| `play_il_gui.py` | `grail-play` | GUI for watching the trained IL agent play, with side-by-side gaze visualization. |
| `play.py` | — | Human play mode with gaze and action recording. |
| `inspect_env.py` | — | Quick environment action/observation space inspection. |
| `atari_env_manager.py` | — | Atari environment builder. Imported by training and evaluation scripts. |

---

## 🔍 search/

Symbolic rule and clause search algorithms.

| File | Description |
|---|---|
| `beam_search.py` | Beam search for NSFR clause generation given a rollout buffer. |
| `naive_search.py` | Brute-force / naive rule search baseline. |

---

## 📐 baselines/

Neural baseline model implementations by other repositories

| File | Description |
|---|---|
| `agil.py` | AGIL (Attention-Guided Imitation Learning) network architecture reference. |

---

## 🐚 shell/

Bash scripts for batch experiments and sweeps.

| File | Description |
|---|---|
| `run_sample_efficiency_sweep.sh` | Sample-efficiency sweep (NSFR + gaze, varying number of episodes). |
| `run_sample_efficiency_sweep_no_gaze.sh` | Same sweep without gaze. |
| `run_bc_incremental_sweep.sh` | BC incremental training sweep (add one episode at a time). |
| `run_bc_independent_sweep.sh` | BC independent-episode training sweep. |
| `run_bc_fewer_objs.sh` | BC training sweep with reduced object set. |
| `run_bc_mask_fewer_objs.sh` | BC + gaze-mask training with fewer objects. |
| `run_bc_mask_incremental.sh` | BC + gaze-mask incremental sweep. |
| `run_agil_fewer_objs.sh` | AGIL baseline with fewer objects. |
| `retrain_agil_1_50ep.sh` | Retrain AGIL model #1 for 50 epochs. |
| `retrain_agil_2_50ep.sh` | Retrain AGIL model #2 for 50 epochs. |

---

## Notes

- Scripts must be run via `grail-*` commands or `python -m scripts.<subfolder>.<script>`. Do **not** invoke them directly as `python scripts/...`.
- Always activate the conda environment first: `conda activate nesy-il`
