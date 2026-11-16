# ITS-Merge-Decision

Research title:

- Korean: Waymax 기반 자율주행 합류 상황에서 PPO 기반 행동 의사결정 연구
- English: PPO-Based Behavior Decision-Making for Autonomous Vehicle Merging in Waymax

## 1. Overview

This repository implements a PPO-based high-level behavior decision policy
(KEEP / FOLLOW / MERGE / STOP) for autonomous-vehicle lane-merge scenarios,
built on real WOMD (Waymo Open Motion Dataset) logged scenes and the Waymax
simulator. Behavior decisions are executed by a common downstream
Frenet-frame planner and an LTV-MPC tracking controller, so the PPO policy
only ever needs to reason over a compact discrete action space, not raw
acceleration/steering.

## 2. Architecture

```
WOMD (raw logged scenes)
    ↓
MERGE Dataset construction (scene/maneuver detection, tiering, splitting)
    ↓
Final MERGE Dataset (frozen: data/manifests/)
    ↓
14D Observation (src/environment/observation_builder.py)
    ↓
PPO policy: KEEP / FOLLOW / MERGE / STOP  (src/policies/ppo/)
    ↓
Reward: Terminal + Safety + Progress + Decision  (src/rewards/)
    ↓
Frenet Planner -> LTV-MPC  (src/planning/, src/control/)
    ↓
Waymax simulation step
    ↓
Checkpoint / W&B / Visualization
```

## 3. Repository Structure

```
ITS-Merge-Decision/
├── configs/
│   ├── dataset.yaml                # WOMD TRAIN shards (final PPO runtime)
│   ├── dataset_validation.yaml     # WOMD VALIDATION shards (tests/tooling only)
│   ├── merge.yaml                  # merge-scenario / lane-assignment config
│   ├── observation.yaml            # frozen 14D observation contract
│   ├── downstream.yaml             # Frenet + LTV-MPC downstream config
│   ├── reward.yaml                 # final PPO reward
│   └── ppo/
│       ├── train.yaml              # canonical training config
│       └── smoke.yaml              # pipeline-verification smoke config
├── data/
│   ├── womd/                       # raw WOMD shards (git-ignored, user-provided)
│   └── manifests/                  # final MERGE Dataset artifacts (tracked)
├── src/
│   ├── environment/                # MergeEnvironment, observation builder, dataset loading
│   ├── scenarios/                  # merge/lane-assignment/scenario feature extraction
│   ├── rewards/                    # reward computation
│   ├── policies/ppo/               # PPO networks + train state
│   ├── training/                   # trainer, config loading, rollout
│   ├── planning/                   # Frenet planner
│   ├── control/                    # LTV-MPC tracking controller
│   ├── tracking/                   # object tracking helpers
│   └── visualization/              # rendering utilities
├── scripts/
│   ├── train_ppo.py                # canonical PPO training entrypoint
│   ├── smoke_ppo.py                # canonical PPO smoke-verification entrypoint
│   ├── visualize_ppo.py            # canonical checkpoint visualization entrypoint
│   └── (dataset construction)      # minimal raw-WOMD -> frozen-dataset reproduction chain
├── tests/                          # pytest suite (see Section 7 of docs/HANDOFF.md)
├── outputs/                        # generated artifacts only (checkpoints/visualizations/metrics)
├── docs/
│   └── HANDOFF.md                  # setup, training, resume, visualization, troubleshooting
├── requirements.txt
└── README.md
```

## 4. Environment

Verified working environment:

- WSL2, Ubuntu 24.04
- Conda environment: `its-merge`, Python 3.10
- NVIDIA RTX 4060, JAX with CUDA support (`jax.devices()` -> `[CudaDevice(id=0)]`)
- Waymax (GitHub main-based install, pinned commit in `requirements.txt`)
- WOMD Motion Dataset v1.3.1
- Key dependencies: `jax`, `flax`, `optax`, `scipy`, `tensorflow`, `waymax`,
  `numpy`, `pyyaml`, `matplotlib`, `pillow`, `wandb`, `pytest`

See [docs/HANDOFF.md](docs/HANDOFF.md) for full setup instructions.

For branch, pull request, review, and validation rules, see
[CONTRIBUTING.md](CONTRIBUTING.md).

The new PPO–SAC–PD interface foundation is documented separately in
[the schema 3 planning contract](docs/contracts/planning.md).
Its initial fixed-provider control loop and smoke command are described in
[the 20-second WOMD control guide](docs/architecture/womd-control.md).

## 5. Dataset

The final MERGE Dataset is built from real WOMD logged scenes: 10 TRAIN
shards (`training_tfexample.tfrecord-00000-of-01000` .. `-00009-of-01000`)
and 6 VALIDATION shards (`validation_tfexample.tfrecord-00000-of-00150` ..
`-00005-of-00150`). Candidate maneuvers are detected, geometrically
classified, tiered (A/B/C/INELIGIBLE), and filtered to a decision-context
eligible CORE/SUPPORT set.

Frozen final counts (never expected to change without an explicit new
freeze): **TRAIN = 1097**, **VALIDATION = 388**.

Canonical tracked artifacts (`data/manifests/`):

- `merge_maneuvers.csv`, `merge_decision_manifest.csv`, `merge_split.csv` —
  maneuver/manifest/split tables
- `evidence_training.jsonl`, `evidence_validation.jsonl` — per-candidate
  decision-context evidence, keyed by `candidate_id`
- `freeze.json` — provenance record (dataset version, git SHA, filter
  config hash, canonical manifest hash, candidate counts, known
  limitations)

The sole runtime loader is `load_decision_dataset_maneuver_specs` in
[src/environment/full_split_evaluator.py](src/environment/full_split_evaluator.py),
which filters by Tier A/B, `dataset_role` CORE/SUPPORT, and
`merge_context_status == MERGE_CONTEXT_ELIGIBLE`, and enforces
`MERGE_DATASET_SCHEMA = "merge_decision"`.

The minimal raw-WOMD -> frozen-dataset reproduction chain is kept under
`scripts/` (construction-only scripts, e.g. `build_merge_manifest.py`,
`extract_merge_v2_evidence.py`) for provenance/reproducibility; it is not
required for training, smoke-testing, or visualization, which only ever
read the frozen `data/manifests/` artifacts above.

## 6. Quick Start

```bash
conda activate its-merge

# Verify the pipeline end-to-end with a tiny smoke run (fresh checkpoint, WANDB offline)
PYTHONPATH=. WANDB_MODE=offline python scripts/smoke_ppo.py \
    --max-maneuvers 2 --num-updates 2 --max-episode-steps 30

# Visualize a checkpoint's rollout outcomes
PYTHONPATH=. python scripts/visualize_ppo.py \
    --checkpoint outputs/checkpoints/<checkpoint>.pkl --scope checkpoint
```

See [docs/HANDOFF.md](docs/HANDOFF.md) for full training/resume/visualization
commands.

## 7. PPO

- Observation: 14D vector (`src/environment/observation_builder.py`),
  frozen contract in `configs/observation.yaml`.
- Action space: `KEEP / FOLLOW / MERGE / STOP`
  (`src/environment/behavior_action.py`), executed downstream through a
  shared Frenet planner + LTV-MPC (`downstream_mode: "frenet_mpc"`).
- Network: 14 -> 256 -> 64 -> 32 -> {4 logits | 1 value}, tanh activations;
  separate policy/value stacks (no shared parameters).
- Hyperparameters (`configs/ppo/train.yaml`): `learning_rate=3e-4`,
  `gamma=0.99`, `gae_lambda=0.95`, `clip_epsilon=0.2`, `value_coef=0.5`,
  `entropy_coef=0.01`, `max_grad_norm=0.5`, `ppo_epochs=4`,
  `num_minibatches=4`.
- `configs/ppo/smoke.yaml` inherits the same architecture/hyperparameters
  and only overrides run-size knobs (`num_minibatches=1`,
  `max_episode_steps=60`) for pipeline verification.

## 8. Reward

```
R_t = R_terminal,t + w_s * R_safety,t + w_p * R_progress,t + w_d * R_decision,t
```

- **Terminal**: success `+1.0`, collision `-1.0`, offroad `-1.0`, horizon
  truncation `-0.5`, none `0.0`.
- **Safety** (`weight=0.02`): target-front TTC/gap only
  (`ttc_danger_s=2.5`, `ttc_safe_s=9.0`, `gap_sufficient_m=20.0`).
- **Progress** (`weight=0.5`): merge-distance progress toward `merge_end_s`.
- **Decision** (`weight=0.02`): switching-only regularizer.

The legacy flat per-decision-step `decision_cost` term is present in
`configs/reward.yaml` only as a disabled (`enabled: false`) dataclass
field for backward field-compatibility; it always contributes exactly
`0.0` to the total and is excluded from `R_t` above.

## 9. Outputs

`outputs/` contains only generated artifacts, never runtime inputs:

```
outputs/
├── checkpoints/       # scripts/train_ppo.py, scripts/smoke_ppo.py
├── visualizations/     # scripts/visualize_ppo.py
└── metrics/
```

All three are git-ignored except a `.gitkeep` per directory. W&B local run
directories (`wandb/`) are also git-ignored.

## 10. Current Research Status

- Final MERGE Dataset: **complete / frozen** (TRAIN=1097, VALIDATION=388)
- Final Reward: **implemented**
- PPO pipeline (Dataset -> 14D Observation -> Reward -> PPO -> Checkpoint):
  **smoke validated**
- Frenet Planner + LTV-MPC downstream execution: **integrated**
- Full PPO training / hyperparameter tuning: **NOT completed**
- Formal final rule-based FSM baseline: **NOT implemented**

## 11. Development History

- **Dataset V1** (`merge_interaction_v2` / `DATASET_SCHEMA_V2`, CONFIRMED_MERGE-
  only contract) — superseded by the decision-context Dataset V2.
- **Dataset V2** (`MERGE_DATASET_SCHEMA = "merge_decision"`) — adopted as
  the final MERGE Dataset.
- **Reward V0** (flat per-decision-step `decision_cost`) — replaced.
- **Reward V1** (Terminal + Safety + Progress + Decision) — adopted as the
  final reward.
- Historical Phase/P5/P6 naming has been removed from the canonical runtime,
  configs, and user-facing entrypoints. Some dataset-construction utilities
  retain historical v1/v2/phase terminology where it is part of the frozen
  dataset reproduction chain.
