# `sim_python/` — Planar Monte-Carlo Contour-Tracking Simulator

This is the Layer-1 (Python, 2D-plane) simulator used to verify the
contour-tracking conditions (C1)/(C2) and Theorem 1 of the accompanying
paper, and to map out where a planar AUV can hold a constant standoff
from a curved wall. It is a fast, fully deterministic Monte-Carlo
verification harness: each sample draws a configuration, runs one
closed-loop simulation, and records whether the vehicle stayed on the
wall.

## What it models

A planar torpedo-class AUV tracks a 2D wall at a fixed lateral standoff
`d*` using line-of-sight (LOS) guidance with a curvature feed-forward
term. The vehicle carries a sonar with a finite field of view (FOV) and
noisy lateral-distance measurements. A 4-state finite-state machine
manages tracking and loss recovery:

| State | Meaning |
|-------|---------|
| `T`   | Tracking — nominal closed-loop LOS + κ̂ feed-forward |
| `L_N` | Lost-Noise — the χ² innovation gate tripped (noise-driven loss); dwell capped by the noise-driven dwell `T*_N` |
| `L_G` | Lost-Geometry — the wall left the FOV cone (occlusion); dwell capped by the geometric-occlusion dwell `T*_G` |
| `R`   | Re-acquire — maximum-likelihood prediction + Archimedean spiral search |

Key quantities the harness exercises:

- the **sliding-window LS curvature estimator** with the exact closed
  form σ_κ̂² = 720 · σ_η² / (L⁵ s̄⁴) (the constant 720 is exact);
- the two closed-form dwell budgets **`T*_N`** (noise-driven) and
  **`T*_G`** (geometric-occlusion), with the joint dwell `max(T*_N, T*_G)`;
- the bandwidth condition **(C2)** in the (`f_s`, `τ_d`) plane and the
  kinematic feasibility condition **(C1)**;
- the **gate-feasibility "island"** and the per-run loss frequencies
  **Lost-N** / **Lost-G**.

## Package layout

```
sim_python/
├─ README.md                  this file
├─ environment.yml            conda environment (pinned)
├─ models/                    torpedo_kinematic_2d (L0 kinematic hull + rudder gate)
├─ scenarios/                 wall_generator (straight / arc / cubic-spline walls)
├─ controllers/               los_controller, curvature_estimator, mode_manager (FSM), reacquire_planner
├─ mc/                        sampler (Latin-Hypercube), dispatcher (parallel execution)
│  └─ analysis/               aggregator, phase_diagram, check_phenomena, dual_sonar_compare
├─ configs/                   YAML batch configs (one per Monte-Carlo batch)
└─ tests/                     pytest unit + smoke tests
```

The package is self-contained and does not import from any companion
higher-fidelity simulator.

## Installation

The simulator ships a pinned conda environment named `sci`:

```bash
conda env create -f sim_python/environment.yml
conda activate sci
```

This installs Python 3.10 plus numpy, scipy, pandas, pyarrow,
matplotlib, pyyaml, pytest, tqdm, and joblib at the versions used to
produce the published results. Run all commands below from the
repository root (the parent of `sim_python/`).

## Running the tests

```bash
python -m pytest sim_python/tests/ -q
```

The suite covers the hull model, the controllers, the wall generators,
the sampler, the dispatcher, and the phenomenon-detection analysis.

## Running the Monte-Carlo batches

Each batch is described by a YAML file under `sim_python/configs/`.
The dispatcher Latin-Hypercube samples the configuration axes, runs one
closed-loop simulation per sample (optionally several noise
realisations), and writes a per-run trajectory parquet plus a summary:

```bash
python -m sim_python.mc.dispatcher \
    --config sim_python/configs/stage_03_smoke.yaml \
    --out-dir sim_python/results/stage_03_smoke \
    --jobs 4
```

- `--config` — a batch YAML (LHS axes, sample count, seed, sim defaults,
  hull and LOS blocks).
- `--out-dir` — directory for the per-run output.
- `--jobs` — number of parallel worker processes (joblib).

All randomness flows through seeded `numpy.random.Generator` instances,
so a fixed `(config, seed)` reproduces a batch exactly.

The provided configs are:

| Config | Purpose |
|--------|---------|
| `stage_01_smoke.yaml` | Open-loop hull smoke (rudder gate + yaw-rate ramp) |
| `stage_02_smoke.yaml` | Closed-loop LOS smoke on a straight wall and an arc |
| `stage_03_smoke.yaml` | N=100 LHS Monte-Carlo smoke |
| `stage_04_main_smoke_n500.yaml`, `stage_04_main_n2000.yaml`, `stage_04_refine_n500.yaml` | Main 5-axis phase-diagram sweep + boundary refinement |
| `stage_05_scanning_n2000.yaml`, `stage_05_forward_n2000.yaml` | Dual-sonar batches (scanning ≈360° FOV vs forward 90° FOV) |
| `stage_05b_scanning_ctrl_n2000.yaml` | Controlled scanning batch that matches the forward batch's sensing and differs only in field of view |

The repository also ships a dual-sonar driver, `scripts/run_stage_05_dual_sonar.py`,
which runs the scanning and forward batches and feeds their outputs to
the comparison helpers; the dispatcher CLI above is the core entry point
that script wraps.

## Outputs

For each run the dispatcher writes
`<out-dir>/run_<hash>/trajectory.parquet` (time series of mode, lateral
distance, heading error, surge, yaw rate, curvature estimate, loss
counts, collision flag, termination reason; `engine="py"`) and a
`summary.json`. Per-run summaries are aggregated into
`<out-dir>/_summary.parquet`.

## How the results map to the paper

- **Gate-feasibility island** — `mc.analysis.check_phenomena`
  detects the interior optimum in the (`κ_max`, `v*`) slice;
  `mc.analysis.phase_diagram` renders it.
- **(f_s, τ_d) bandwidth slice** — the same module checks that the (C2)
  feasibility boundary follows the hyperbola `τ_d + 1/(2 f_s) = const`.
- **Dual-sonar T\*_G comparison** — `mc.analysis.dual_sonar_compare`
  pairs the scanning and forward batches and shows that the
  geometric-occlusion dwell `T*_G` (and hence Lost-G) is driven by the
  field of view: a ≈360° scanning sonar yields Lost-G ≈ 0, while the
  90° forward sonar shows a finite `T*_G` that grows as `κ'_max` shrinks.
  `stage_05b_scanning_ctrl_n2000.yaml` holds the sensing parameters at
  the forward batch's values so the contrast is attributable to the FOV
  alone.

## Reproducibility notes

- Every batch records its git commit, LHS seed, and engine label in the
  per-run `summary.json`.
- The curvature-variance constant 720, the dwell closed forms, and the
  χ² gate thresholds are fixed in code; they are not tuned per batch.
- Large per-run outputs are git-ignored; see `.gitignore` for the
  retention policy.
