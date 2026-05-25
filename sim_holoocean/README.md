# `sim_holoocean/` — Layer 2: HoloOcean Photorealistic Sim-to-Sim Verification

## Purpose

This is the Layer-2 simulator: an independent, photorealistic re-validation of the
contour-tracking conditions (C1)(C2) and Theorem 1 from the paper, built on the
HoloOcean 2.3 underwater simulator with GPU-rendered ImagingSonar / ProfilingSonar
returns. Where the lightweight Layer-1 Python simulator (`sim_python/`) uses a 2D
ray-cast sonar model, Layer-2 drives the *same* contour-tracking controller against
HoloOcean's photorealistic acoustic rendering of a real harbor wall.

The deliverable is a **sim-to-sim consistency check**: for a shared configuration
grid, the two engines should agree in sign on the key tracking outcomes
(lost-frequency, collide-rate, empirical guaranteed-loss time). This supports the
manuscript's bounded-tracking and dual-sonar results (Section VI, Fig. 6).

Layer-2 is fully self-contained: nothing in `sim_holoocean/` imports from
`sim_python/`, and vice versa. The ported controllers (`controllers/`) re-implement
the Layer-1 control law byte-for-byte so the two engines compare the *same*
controller; only the sensor input adapter differs.

## What it demonstrates

- **Photorealistic bounded tracking** on the PierHarbor harbor wall: the torpedo AUV
  holds a bounded standoff distance while tracking the wall contour, matching the
  Lemma-1 bounded-tracking prediction.
- **The w_FOV → 360° rigidity limit** (the dual-sonar contrast): a narrow forward
  sonar fan can lose the wall at a sharp convex corner (the wall tangent rotates out
  of the field of view), driving the controller into a guaranteed-loss episode; a
  360° mechanically-scanning sonar keeps the wall visible on every sweep, so the
  visibility predicate never drops and the loss episode never triggers. This is the
  photorealistic realization of the scanning-vs-forward sonar contrast in the paper.

## Requirements

HoloOcean 2.3 cannot render on a headless machine without the engine binary: it
requires a machine with the HoloOcean 2.3 engine and its world packages installed
(GPU-rendered). The scenarios in this folder build HoloOcean `scenario_cfg` dicts and
must be executed on such a machine.

Create the environment from the pinned spec:

```bash
conda env create -f sim_holoocean/environment.yml
conda activate holoocean
```

`environment.yml` pins HoloOcean 2.3.0 on Python 3.10 plus the numeric stack
(numpy / scipy / pandas / pyarrow / pyyaml). The HoloOcean `Ocean` world package
(which ships `OpenWater`, `PierHarbor`, `Dam`, etc.) must be installed alongside the
engine; HoloOcean fetches its world packages on first use.

## How to run

All scenarios are runnable as modules from the repository root with
`PYTHONPATH=.`:

```bash
# Minimal end-to-end check: load a world, tick ~100 frames, write a tiny parquet.
PYTHONPATH=. python -m sim_holoocean.scenarios.empty_world

# Static-target sonar range-fidelity check (forward + scanning archetypes).
PYTHONPATH=. python -m sim_holoocean.scenarios.static_target_smoke \
    --config sim_holoocean/configs/stage_02_smoke.yaml

# Closed-loop contour tracking in PierHarbor (full demo).
PYTHONPATH=. python -m sim_holoocean.mc.run_tracking \
    --config sim_holoocean/configs/stage_03_pierharbor.yaml

# Dual-sonar contrast — narrow forward fan (loses the wall at a convex corner):
PYTHONPATH=. python -m sim_holoocean.mc.run_tracking \
    --config sim_holoocean/configs/stage_03b_armA_forward60.yaml
# vs. 360° scanning sonar (wall stays visible on every sweep):
PYTHONPATH=. python -m sim_holoocean.mc.run_tracking \
    --config sim_holoocean/configs/stage_03b_armB_mss360.yaml
```

Add `--smoke` to `run_tracking` for a short verification run. Per-run output (parquet
trajectory + sonar-frame snapshots + a JSON summary) lands under
`sim_holoocean/results/`.

## Directory layout

```
sim_holoocean/
├─ README.md                  this file
├─ environment.yml            conda env pin (holoocean 2.3.0, py3.10, pyarrow, ...)
├─ __init__.py
├─ scenarios/                 HoloOcean scenario builders + range-fidelity drivers
│   ├─ empty_world.py         minimal smoke: load Ocean world, tick, write parquet
│   ├─ static_target_smoke.py static-target sonar range-fidelity check
│   └─ wall_tracking_world.py PierHarbor torpedo + sonar scenario builder
├─ models/                    torpedo hull config + forward / scanning / MSS sonar
├─ configs/                   per-scenario YAML run configs
├─ controllers/              ported LOS + curvature + FSM + re-acquire controllers
├─ mc/                        closed-loop tracking driver
├─ results/                   per-run parquet / npz / summary output
└─ logs/                      run logs
```

## Configs and models

- `models/torpedo_config.yaml` — the Fossen torpedo dynamics + actuator config for the
  TorpedoAUV (validated REMUS-100-class baseline; CG/CB tuning, fin envelope).
- `models/forward_sonar.yaml` — the "forward" imaging sonar archetype (HoloOcean
  `ImagingSonar`, 90° azimuth fan, 256 beams), a direct 1:1 mapping of the paper's
  forward-sonar parameters.
- `models/scanning_sonar.yaml` — the "scanning" archetype (HoloOcean `ProfilingSonar`,
  wide forward fan). HoloOcean's sonars are fixed-fan, so the 360° mechanical-scanning
  sonar is reconstructed in software instead (see `models/mss_sonar.py`).
- `models/mss_sonar.py` — the mechanically-scanning-sonar (MSS) reconstruction: a
  single narrow ImagingSonar beam steered over a full circle via `sensor.rotate()`,
  with returns accumulated into a polar map to assemble a 360° sweep.
- `configs/` — YAML run configs binding a world, spawn pose, sonar parameters and
  controller gains for each scenario.

## Output schema

Per-run parquet from `sim_holoocean/results/` shares column names and dtypes with the
Layer-1 `sim_python/results/` output, with an added `engine` field set to `"ho"`
(vs `"py"` for Layer-1), so the two engines can be compared directly.
