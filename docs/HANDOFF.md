# Handoff Guide

## Contents

1. Prerequisites
2. First-Time Setup
3. WOMD Setup
4. Environment Verification
5. PPO Smoke Run
6. PPO Training
7. Resuming Training
8. Visualization
9. Weights & Biases (W&B)
10. Outputs
11. Troubleshooting
12. Remaining Research Work

## 1. Prerequisites

- Linux (verified on WSL2 Ubuntu 24.04) with an NVIDIA GPU + CUDA driver
- Conda / Miniconda
- Access to WOMD (Waymo Open Motion Dataset) v1.3.1 under the Waymo Open
  Dataset license, and Google Cloud auth (`gcloud`) to download shards

## 2. First-Time Setup

```bash
conda create -n its-merge python=3.10
conda activate its-merge
pip install -r requirements.txt
```

Verify JAX sees the GPU:

```bash
python -c "import jax; print(jax.devices())"
# expect: [CudaDevice(id=0)]
```

## 3. WOMD Setup

The final MERGE Dataset's runtime loader needs only the raw WOMD shards
referenced by `configs/dataset.yaml` (TRAIN) and
`configs/dataset_validation.yaml` (VALIDATION) — the canonical
`data/manifests/` artifacts (tracked in git) already encode which
maneuvers/candidates exist; only the underlying physical scene data must
be downloaded locally.

```bash
mkdir -p data/womd/training data/womd/validation

# TRAIN shards (10): training_tfexample.tfrecord-00000-of-01000 .. -00009-of-01000
gcloud storage cp \
  gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord-0000{0..9}-of-01000 \
  data/womd/training/

# VALIDATION shards (6): validation_tfexample.tfrecord-00000-of-00150 .. -00005-of-00150
gcloud storage cp \
  gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord-0000{0..5}-of-00150 \
  data/womd/validation/
```

Raw WOMD shards are git-ignored (`data/*`) and must never be committed.

### 20-second WOMD preparation (separate from the frozen merge runtime)

Store new large scenario datasets and derived artifacts on Storage (E:),
mounted as `/mnt/e`. Keep source code and the Conda environment in Ubuntu.
The downloaded v1.3.1 source is
`/mnt/e/Datasets/WOMD/v_1_3_1/scenario/training_20s` (1000 shards).
Do not point the existing 91-frame `tf.Example` training config at these
Scenario-protobuf files. The following standalone commands prepare and inspect
the new data without modifying that runtime:

```bash
conda activate its-merge
cd /home/autonav/2026-AD-Capstone

# Full source index: TFRecord CRC, protobuf parsing and structural validation.
# Repeating the command resumes shards whose size/mtime and index version match.
python scripts/prepare_womd20.py index

# Convert the first three scenarios, keeping every source frame and object.
python scripts/prepare_womd20.py inspect

# Replay all frames and render sampled frames as GIFs (default every 5th frame).
python scripts/prepare_womd20.py replay
```

All outputs default to
`/mnt/e/Datasets/WOMD/v_1_3_1/derived/training_20s/`:

- `index/index.sqlite`: resumable per-shard index; only run one index writer.
- `index/scenarios.csv`, `index/shards.csv`, `index/summary.json`: exported
  metadata, errors and full-scan status. `full_scan_ok` requires all 1000 shards,
  no missing sources, no failed shards/records, and no duplicate scenario IDs.
- `inspect/report.json`: unbatched Waymax conversion reports.
- `replay/report.json`: numeric replay checks. Each scene also has
  `replay_frames.csv`, `replay.gif`, and `last_frame.png` in its own subdirectory.

Use `index --limit-shards 1` for a pilot; a pilot's `full_scan_ok: false` is
expected. Use `inspect --shard-index 1 --record-indices 0 1 2` (or the same flags
with `replay`) for other scenes. `replay --render-stride 2` produces smoother
GIFs; `--no-gif` runs numeric checks only. Every source frame is checked
regardless of rendering stride. `--root` and `--output` override defaults.
The CLI defaults JAX to CPU and disables TensorFlow GPU use for these tasks.

The adapter uses the source's actual timestamps (rebased to relative int32
microseconds), tracks, validity, dimensions, map vertices and per-lane signal
states. It neither pads trajectories nor limits scenes to 64/128 objects.
The original proto retains map topology. Driveways have geometry but use
Waymax's UNKNOWN map type; SDC paths/routes are not loaded/reconstructed.
Replay uses `update_state_by_log`, checks the final trajectory against the log,
and checks signal state alignment and termination at the actual last frame.
This verifies data loading and playback, not learned control, PD tracking,
reactive traffic, fixed-shape training batches or train/validation split safety.
Future logs remain simulator data, not a policy observation specification.

Targeted verification:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest -q tests/scenarios/test_womd20.py
```

## 4. Environment Verification

```bash
PYTHONPATH=. pytest tests/ -m "not slow"
```

This runs the full targeted regression suite (fast representative-sample
tests + real-environment checks), excluding the exhaustive
`@pytest.mark.slow` full-dataset (1485 real Waymax reset) tests. Never
run plain `pytest` with no marker filter as the default verification
command.

To run the exhaustive slow suite manually (only when doing a full
manual/clean-clone verification):

```bash
PYTHONPATH=. pytest tests/ -m slow
```

## 5. PPO Smoke Run

Verifies the full Dataset -> 14D Observation -> Reward -> PPO -> Checkpoint
pipeline end-to-end on a tiny slice of TRAIN, without performing any real
training:

```bash
PYTHONPATH=. WANDB_MODE=offline python scripts/smoke_ppo.py \
    --max-maneuvers 2 --num-updates 2 --max-episode-steps 30
```

Writes a checkpoint to `outputs/checkpoints/smoke_seed<seed>_<timestamp>.pkl`.

## 6. PPO Training

```bash
PYTHONPATH=. python scripts/train_ppo.py \
    --ppo-config configs/ppo/train.yaml \
    --max-maneuvers <N> \
    --num-updates <N> \
    --max-episode-steps <N>
```

Key flags: `--seed` (override config seed), `--maneuver-ids` (explicit
comma-separated maneuver_id list, overrides `--max-maneuvers`),
`--checkpoint-path` (default
`outputs/checkpoints/train_seed<seed>_<timestamp>.pkl`),
`--dataset-config-path` (default `configs/dataset.yaml`).

## 7. Resuming Training

```bash
PYTHONPATH=. python scripts/train_ppo.py \
    --ppo-config configs/ppo/train.yaml \
    --resume outputs/checkpoints/<checkpoint>.pkl \
    --num-updates <N>
```

`--resume` continues `global_env_step`/`ppo_update_step` from the
checkpoint rather than restarting from zero. The checkpoint's recorded
`dataset_schema_version` must be `MERGE_DATASET_SCHEMA` ("merge_decision");
checkpoints produced before this repository's cleanup are not expected to
satisfy this and are not supported for resume.

## 8. Visualization

```bash
PYTHONPATH=. python scripts/visualize_ppo.py \
    --checkpoint outputs/checkpoints/<checkpoint>.pkl \
    --scope checkpoint
```

`--scope` accepts `checkpoint` (maneuvers the checkpoint was
trained/evaluated on) or `explicit` (with `--maneuver-ids`). Use
`--scan-only` for a fast outcome-only pass (writes `outcome_index.csv`,
`selected_episodes.json`, `manifest.json`, no rendering) or `--render-all`
to render every scanned episode. Output goes to
`outputs/visualizations/<run-id>/` (defaults to the checkpoint filename
stem).

## 9. Weights & Biases (W&B)

Set `WANDB_MODE=offline` to disable network calls (used for smoke runs
and CI), or `WANDB_MODE=online` (default in `configs/ppo/train.yaml`) to
log to a real project (`wandb_project` in the PPO config). Do not commit
any W&B API key — authenticate locally via `wandb login` or the
`WANDB_API_KEY` environment variable; never record credentials in this
repository or in any config file.

`wandb/` local run directories are git-ignored.

## 10. Outputs

`outputs/` is generated-artifact-only:

```
outputs/
├── checkpoints/       # scripts/train_ppo.py, scripts/smoke_ppo.py
├── visualizations/    # scripts/visualize_ppo.py
└── metrics/
```

All contents except `.gitkeep` are git-ignored. Nothing under `outputs/`
is a runtime input to any canonical entrypoint.

## 11. Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`** — prefix the command
  with `PYTHONPATH=.` (all canonical scripts are invoked this way).
- **`FailedPreconditionError: Failed to allocate scratch buffer for
  device 0`** during tests — usually a leftover process from a previous
  run still holding the GPU. Check `ps aux | grep pytest` (or the
  relevant script) for a stale process and stop it before re-running.
- **`ValueError: Unknown shard: source_shard='validation_...' does not
  match the basename of any physical path`** — a VALIDATION-split
  maneuver was reset against a TRAIN-only `MergeEnvironment`. Waymax
  binds exactly one physical split per `DatasetExpansionConfig`; use
  `configs/dataset_validation.yaml` (a separate `MergeEnvironment`
  instance) for VALIDATION-split resets.
- **Checkpoint fails to resume with a dataset-schema error** — the
  checkpoint was produced before the final MERGE Dataset contract. Only
  checkpoints produced by the current `scripts/train_ppo.py`/
  `scripts/smoke_ppo.py` (schema `merge_decision`) are supported.

## 12. Remaining Research Work

- Full PPO training / hyperparameter tuning has not been run — only
  pipeline-verification smoke runs.
- A formal final rule-based FSM baseline (for comparison against PPO) is
  not implemented.
- The dataset-construction reproduction chain has not yet been
  reorganized under a single dedicated `scripts/dataset/` path with
  fully functional naming; the scripts currently live at the top level
  of `scripts/` alongside the canonical training/smoke/visualization
  entrypoints.
