# Sonar-Based Contour-Tracking Measurement Systems — Reproducibility Package

Code and complete mathematical derivations accompanying the paper

> **Sufficient Conditions for Sonar-Based Contour-Tracking Measurement
> Systems: Field-of-View Geometry and a Scanning-versus-Forward
> Comparison.**

The paper derives analytic performance limits for an underactuated
autonomous underwater vehicle (AUV) following a wall at a controlled
standoff with a single sonar, and validates them in two simulation
layers. This repository contains:

- **`derivations/`** — the full, self-contained derivations of every
  lemma and the main theorem (the paper carries proof *sketches* only).
- **`sim_python/`** — **Layer 1**: a planar dynamics-plus-sensor
  Monte-Carlo simulator that produces the quantitative results
  (the gate-feasibility island, the bandwidth-condition slice, and the
  dual-sonar geometric-occlusion comparison).
- **`sim_holoocean/`** — **Layer 2**: a photorealistic
  [HoloOcean](https://holoocean.byu.edu) 2.3 pipeline that renders sonar
  returns from textured 3-D scenes and tracks the wall of the built-in
  *PierHarbor* world under the same guidance law.

---

## Derivations ↔ paper

Each file in `derivations/` opens with a header naming the paper section,
lemma/theorem, and equation(s) it derives. Summary:

| Derivation file | Paper section | Result | Key equation(s) |
|---|---|---|---|
| `01-tracking-conditions.md` | III | Lemma 1 (tracking-mode ISS) | static condition **(C1)**, bandwidth condition **(C2)**, Table I budget terms |
| `02-curvature-uncertainty.md` | IV | curvature-estimator uncertainty | $\sigma_{\hat\kappa}^{2}=720\,\sigma_\eta^{2}/(L^{5}\bar s^{4})$ |
| `03-loss-dwell.md` | IV | Lemma 2 (loss-mode dwell) | $T^{\star}_N=\sqrt{2(d^\star-d_{\min})\beta_N/(\sigma_{\hat\kappa}\underline u^{2})}$; $T^{\star}_G=\underline u^{-1}\sqrt{w_{\mathrm{FOV}}/\kappa'_{\max}}$ |
| `04-reacquisition.md` | IV | Lemma 3 (re-acquisition) | $P[\text{re-acquire}]\ge 1-e^{-\lambda}$ + control-barrier-function safety |
| `05-joint-trackability.md` | V | Theorem 1 (joint trackability) | finite-horizon bound $\delta(\mathcal T)\le(\mathcal T/T_D)(\delta_L+e^{-\lambda}+\delta_G)$ |

See `derivations/README.md` for the full correspondence table and a
symbol glossary. Math renders on GitHub via MathJax.

---

## Code usage

The two layers are independent; each ships a conda `environment.yml`.

### Layer 1 — planar Monte-Carlo (`sim_python/`)

```bash
conda env create -f sim_python/environment.yml
conda activate sci

# unit tests (117)
PYTHONPATH=. python -m pytest sim_python/tests/ -q

# a single Monte-Carlo batch (N samples, one noise realisation)
PYTHONPATH=. python -m sim_python.mc.dispatcher \
    --config sim_python/configs/stage_04_main_n2000.yaml \
    --out-dir results/main --jobs 8

# the dual-sonar comparison (N x M = 6000 runs per sonar)
PYTHONPATH=. python sim_python/scripts/run_stage_05_dual_sonar.py \
    --config sim_python/configs/stage_05_forward_n2000.yaml \
    --out-dir results/forward --jobs 12
PYTHONPATH=. python sim_python/scripts/run_stage_05_dual_sonar.py \
    --config sim_python/configs/stage_05b_scanning_ctrl_n2000.yaml \
    --out-dir results/scanning_ctrl --jobs 12
```

Each run writes a per-configuration `trajectory.parquet` and a batch
`_summary.parquet`. See `sim_python/README.md` for the configuration
table and the per-run output schema.

### Layer 2 — HoloOcean photorealistic (`sim_holoocean/`)

```bash
conda env create -f sim_holoocean/environment.yml
conda activate holoocean
# HoloOcean 2.3 and its world packages must be installed on a machine
# with the rendering engine; see sim_holoocean/README.md.

python sim_holoocean/scenarios/wall_tracking_world.py
```

`sim_holoocean/README.md` documents the torpedo and sonar models
(`models/`), the scenarios (`scenarios/`), and the mechanically-scanning
sonar reconstruction.

---

## Reproducing the paper's headline results

| Paper artefact | Produced by |
|---|---|
| Gate-feasibility island, $(\kappa_{\max}, v^*)$ | `sim_python` `stage_04_main_n2000` batch |
| Bandwidth-condition slice, $(f_s, \tau_d)$ | `sim_python` `stage_04_main_n2000` batch |
| Dual-sonar $T^\star_G$ comparison; Lost-$G$ $0.000\%$ (scanning) vs $99.900\%$ (forward) | `sim_python` `stage_05_forward_n2000` + `stage_05b_scanning_ctrl_n2000` (identical sensing, differing only in field of view) |
| $\sigma_{\hat\kappa}^2$, $T^\star_N$, $T^\star_G$ closed-form agreement (<1%) | `sim_python` verification subset (see `sim_python/README.md`) |
| Photorealistic bounded tracking + scanning-sonar rigidity | `sim_holoocean` PierHarbor scenario |

---

## License and citation

Released under the MIT License (see `LICENSE`).

If you use this code or the derivations, please cite:

```bibtex
@article{Yan2026ContourTracking,
  author  = {Yan, Zheping and Min, Xuyu and Chen, Tao and Chen, Hailun},
  title   = {Sufficient Conditions for Sonar-Based Contour-Tracking
             Measurement Systems: Field-of-View Geometry and a
             Scanning-versus-Forward Comparison},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  note    = {Submitted},
  year    = {2026}
}
```
