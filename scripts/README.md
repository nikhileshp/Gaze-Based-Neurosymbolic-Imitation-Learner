# Scripts

This directory contains all executable scripts for the **Gaze-Based Neurosymbolic Imitation Learner** project, organised into thematic subfolders.

Each subfolder is a Python package (`__init__.py` present) and each script adds the `scripts/` directory to `sys.path` at startup so that cross-folder imports always resolve correctly.  
Run any script from the **project root**:

```bash
conda activate nesy-il
python scripts/<subfolder>/<script>.py [args]
```

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
| [`utils/`](#-utils) | Shared utilities (email, eval helpers, etc.) |
| [`shell/`](#-shell) | Bash scripts for running experiments |

---

## training/

Training scripts for the neurosymbolic IL pipeline and neural baselines.

| File | Description |
|---|---|
| `train_il.py` | **Main** NSFR-based imitation learning trainer. Supports `.pt` dataset, gaze-guided training, and per-epoch evaluation. |
| `train_il_main.py` | IL training from the `main` branch (legacy variant). |
| `train_il_new.py` | Experimental IL training variant. |
| `train_bc.py` | Behaviour Cloning (GABRIL-style) with multiple gaze methods (BC, AGIL, ViSaRL, Mask, Reg, Contrastive, GRIL). |
| `train_bc_pt.py` | Simplified BC training directly from a `.pt` dataset. |
| `train_gaze_predictor_gabril.py` | Train a GABRIL-style AutoEncoder-based gaze predictor. |
| `train_valuation_cnn.py` | Train a CNN to regress NSFR atom valuations from raw pixel observations. |
| `train.py` | Generic PPO / on-policy training entry point. |
| `ppo.py` | PPO reinforcement learning agent (standalone). |

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

| File | Description |
|---|---|
| `compare_valuation_models.py` | Compare two trained NSFR valuation models side-by-side. |
| `compare_valuation_preds.py` | Compare valuation predictions between model checkpoints. |
| `show_rules.py` | Display learned NSFR symbolic rules from a trained model. |
| `precompute_valuations.py` | Pre-compute NSFR atom valuations over a dataset and cache to disk. |
| `generate_valuation_atoms.py` | Generate ground-truth atom probabilities from a `.pt` dataset using the NSFR engine. |

---

## 💾 preprocess/

Data collection, format conversion and preprocessing utilities.

| File | Description |
|---|---|
| `collect_data.py` | Play the game with a scripted/human policy and record observations, actions, and gaze to disk. |
| `convert_trajectories_to_pt.py` | Convert raw CSV+image trajectories into a single self-contained `.pt` dataset file. |
| `convert_seaquest.py` | Seaquest-specific conversion of legacy data formats. |
| `preprocess.py` | General preprocessing utilities. |
| `preprocess_dataset.py` | Preprocess a `.pt` dataset (action filtering, episode splitting, etc.). |
| `preprocess_focus.py` | Preprocess gaze focus/segment data: maps JSON segments to CSV trajectories. |
| `data_utils.py` | **Central dataset utilities**: `PtDataset`, `ExpertDataset`, `load_gaze_predictor_data`. Imported by most training scripts. |
| `load_data.py` | Load legacy CSV/image-based trajectory data (`Dataset` class for the Atari-HEAD format). |

---

## 👁️ gaze/

Everything related to gaze data processing, prediction, and analysis.

| File | Description |
|---|---|
| `gaze_predictor.py` | `HumanGazeNet` CNN model and `Human_Gaze_Predictor` trainer/predictor class. Entry point for training gaze heatmap predictors. |
| `extract_gaze_goals.py` | Segment gaze trajectories into goal-directed episodes based on fixation patterns. |
| `compare_gaze_predictions.py` | Produce a side-by-side video comparing ground-truth vs. predicted gaze heatmaps. | Moved to test
| `compare_gaze_predictors.py` | Quantitatively compare multiple gaze predictor checkpoints on a `.pt` dataset. | Moved to test
| `cluster_goals.py` | Cluster gaze goal segments using K-Means or hierarchical clustering to discover high-level behaviour patterns. |
| `label_segments.py` | Assign human-readable labels to gaze clusters from `cluster_goals.py` output. |
| `mine_rules.py` | Mine PDDL-style skill preconditions/effects from labelled gaze segments. |

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

| File | Description |
|---|---|
| `visualize_pt.py` | Replay frames from a `.pt` dataset with optional gaze heatmap overlay. |
| `visualize_trajectory.py` | Replay a recorded trajectory video with overlaid game state info. |
| `visualize_gaze_segmentation.py` | Visualize gaze segmentation results on game frames. |
| `visualize_gaze_segments.py` | Plot gaze segment timelines and spatial patterns. |
| `visualize_gaze_segments_backup.py` | Backup/legacy version of the gaze segment visualizer. |
| `make_trajectory_video.py` | Export a game trajectory as an MP4 video. |
| `vizu_ppo.py` | Visualise a trained PPO agent playing the game. |

---

## 🎮 play/

Interactive play scripts and environment wrappers.

| File | Description |
|---|---|
| `play.py` | Human play mode with gaze and action recording. |
| `play_gui.py` | Minimal GUI launcher for human play. |
| `play_il_gui.py` | GUI for watching the trained IL agent play, with side-by-side gaze visualization. |
| `bigfish_play.py` | Human play in the BigFish environment. |
| `coinjump_play.py` | Human play in the CoinJump environment. |
| `loot_play.py` | Human play in the Loot environment. |
| `inspect_env.py` | Quick environment action/observation space inspection. |
| `atari_env_manager.py` | Atari environment builder (frame-stack, action repeat, recording wrappers). Imported by training and evaluation scripts. |

---

## 🔍 search/

Symbolic rule and clause search algorithms.

| File | Description |
|---|---|
| `beam_search.py` | Beam search for NSFR clause generation given a rollout buffer. |
| `naive_search.py` | Brute-force / naive rule search baseline. |

---

## 📐 baselines/

Neural baseline model implementations.

| File | Description |
|---|---|
| `agil.py` | AGIL (Attention-Guided Imitation Learning) Keras/TensorFlow network architecture reference. |
| `linear_models.py` | CNN `Encoder`, `Decoder`, `AutoEncoder`, and `VectorQuantizer` modules used by BC and AGIL training. |
| `gabril_utils.py` | GABRIL dataset loading (`load_dataset`), evaluation loop, and gaze mask generation utilities. |

---

## 🔧 utils/

Shared utilities used across the project.

| File | Description |
|---|---|
| `utils.py` | General evaluation helper (`evaluate`), gaze-mask utilities (`GazeToMask`, `apply_gmd_dropout`), and seeding. |
| `email_me.py` | Send email notifications during long training runs (SMTP helper). |
| `find_warning_frames.py` | Scan a `.pt` dataset for frames with anomalous states or missing gaze data. |

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
| `run_bc_stack_4.sh` | BC training with frame-stack size 4. |
| `run_agil_fewer_objs.sh` | AGIL baseline with fewer objects. |
| `run_agil_incremental_train_bc_pt.sh` | AGIL incremental training from `.pt` dataset. |
| `retrain_agil_1_50ep.sh` | Retrain AGIL model #1 for 50 epochs. |
| `retrain_agil_2_50ep.sh` | Retrain AGIL model #2 for 50 epochs. |

---

## Notes

- **`gaze_comparison.mp4`** — Sample gaze comparison video; kept in the `scripts/` root for reference.
- All scripts that cross-import each other use a `sys.path` shim at the top of the file to ensure imports resolve correctly regardless of working directory.
- Always activate the conda environment before running: `conda activate nesy-il`
