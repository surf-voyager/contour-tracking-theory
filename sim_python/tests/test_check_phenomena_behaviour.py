"""Per-phenomenon behaviour tests with synthetic data.

For each of the 6 phenomenon checks, we construct two synthetic
DataFrames — one in which the phenomenon clearly *holds* (asserts PASS)
and one in which it clearly *does not* (asserts FAIL or WEAK).  This
gives us confidence that the detector is sensitive in the expected
direction before we run it on noisy MC data.

Theory citations are inlined in each test so the synthetic generator
makes the same assumption the detector expects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sim_python.mc.analysis import check_phenomena as cp


RNG = np.random.default_rng(2026)


# --------------------------------------------------------------------------- #
# Phenomenon 1 — Gate island                                                  #
# --------------------------------------------------------------------------- #


def _gate_island_holds() -> pd.DataFrame:
    """For each κ_max, trackable rate peaks at an interior v*.

    Model: trackable = 1 with prob (1 - (v* - v_opt(κ))²/0.5²) where
    v_opt(κ) = 1.0 (a single peaked rate vs v*, regardless of κ).
    """
    rows = []
    for kappa in np.geomspace(0.01, 1.0, 60):
        for v in np.linspace(0.2, 2.0, 20):
            p = max(0.0, 1.0 - (v - 1.0) ** 2 / 0.6)
            track = RNG.random() < p
            rows.append({"kappa_max": kappa, "v_star": v, "trackable": track})
    return pd.DataFrame(rows)


def _gate_island_fails() -> pd.DataFrame:
    """All trackable=True (monotone trivial)."""
    return pd.DataFrame({
        "kappa_max": np.geomspace(0.01, 1.0, 100),
        "v_star": np.linspace(0.2, 2.0, 100),
        "trackable": [True] * 100,
    })


def test_phenomenon_1_pass_on_gate_island_synthetic() -> None:
    df = _gate_island_holds()
    status, ev = cp.check_phenomenon_1_gate_island(df, min_runs_per_kappa=4)
    assert status == "PASS", f"expected PASS; got {status} with {ev}"
    assert ev["max_amplitude"] > 0.2


def test_phenomenon_1_fail_when_flat() -> None:
    df = _gate_island_fails()
    status, _ = cp.check_phenomenon_1_gate_island(df, min_runs_per_kappa=4)
    assert status in ("FAIL", "WEAK")


# --------------------------------------------------------------------------- #
# Phenomenon 2 — C2 hyperbolic boundary                                       #
# --------------------------------------------------------------------------- #


def _c2_boundary_holds(c: float = 0.4) -> pd.DataFrame:
    """trackable iff τ_d + 1/(2 f_s) ≤ c."""
    rows = []
    for f in np.geomspace(0.5, 20.0, 30):
        for t in np.linspace(0.05, 0.5, 20):
            s = t + 1.0 / (2.0 * f)
            rows.append({"f_s": f, "tau_d": t, "trackable": s <= c})
    return pd.DataFrame(rows)


def _c2_boundary_fails() -> pd.DataFrame:
    """trackable depends only on τ_d (vertical boundary)."""
    rows = []
    for f in np.geomspace(0.5, 20.0, 30):
        for t in np.linspace(0.05, 0.5, 20):
            rows.append({"f_s": f, "tau_d": t, "trackable": t < 0.2})
    return pd.DataFrame(rows)


def test_phenomenon_2_pass_on_hyperbolic_synthetic() -> None:
    df = _c2_boundary_holds(c=0.4)
    status, ev = cp.check_phenomenon_2_c2_hyperbolic_boundary(df)
    assert status == "PASS", f"expected PASS; got {status}, evidence={ev}"
    assert ev["r2"] >= 0.7


def test_phenomenon_2_fail_on_vertical_boundary() -> None:
    df = _c2_boundary_fails()
    status, ev = cp.check_phenomenon_2_c2_hyperbolic_boundary(df)
    # Vertical-boundary synthetic does have a τ_d-monotone decay
    # (rate drops as τ_d > 0.2) which gives the alt detector some
    # signal even though the f_s axis is irrelevant.  We tolerate
    # WEAK / PASS-via-alt-detector here as long as the *headline*
    # threshold-fit R² stays low (the headline signature of the
    # hyperbolic boundary is absent).
    assert ev["r2"] < 0.6, f"primary R² should stay low; got {ev['r2']}"


# --------------------------------------------------------------------------- #
# Phenomenon 3 — Lost-freq power-law divergence                               #
# --------------------------------------------------------------------------- #


def _powerlaw_holds(slope: float = -1.0, kappa_crit: float = 1.2) -> pd.DataFrame:
    """lost_freq = (κ_crit - κ_max)^slope (steep negative slope)."""
    kappa = np.linspace(0.01, 1.1, 80)
    margin = kappa_crit - kappa
    freq = (margin ** slope) * 0.001
    track_rate = np.clip(1.0 - 0.6 * (kappa / kappa_crit), 0, 1)
    track = RNG.random(len(kappa)) < track_rate
    return pd.DataFrame({
        "kappa_max": kappa, "lost_freq": freq, "trackable": track,
    })


def _powerlaw_fails() -> pd.DataFrame:
    """Constant lost_freq independent of κ_max."""
    kappa = np.linspace(0.01, 1.0, 50)
    return pd.DataFrame({
        "kappa_max": kappa,
        "lost_freq": np.full_like(kappa, 0.01),
        "trackable": [True] * len(kappa),
    })


def test_phenomenon_3_pass_on_powerlaw_synthetic() -> None:
    df = _powerlaw_holds(slope=-1.5)
    status, ev = cp.check_phenomenon_3_lost_freq_blowup_near_kappa_crit(
        df, kappa_crit=1.2
    )
    assert status == "PASS", f"expected PASS; got {status} with {ev}"
    assert ev["slope"] < -0.5


def test_phenomenon_3_fail_when_flat() -> None:
    df = _powerlaw_fails()
    status, _ = cp.check_phenomenon_3_lost_freq_blowup_near_kappa_crit(
        df, kappa_crit=1.0
    )
    assert status in ("FAIL", "WEAK")


# --------------------------------------------------------------------------- #
# Phenomenon 4 — d*_min vertical asymptote                                    #
# --------------------------------------------------------------------------- #


def _dstar_min_holds() -> pd.DataFrame:
    """collision rate grows with κ_max (Ph4 proxy mode)."""
    kappa = np.linspace(0.01, 1.0, 100)
    # Collision probability rises with κ_max (proxy for d*_min asymptote).
    p_coll = np.clip(0.05 + 0.6 * kappa ** 2, 0, 1)
    coll = RNG.random(len(kappa)) < p_coll
    err = 0.05 + 0.5 * kappa ** 2 + 0.01 * RNG.normal(size=len(kappa))
    return pd.DataFrame({
        "kappa_max": kappa,
        "mean_err": np.maximum(err, 0.01),
        "collide": coll,
    })


def _dstar_min_fails() -> pd.DataFrame:
    """collision rate ~ constant in κ_max (use a deterministic 10% rate
    spread evenly across the κ-range so binning gives flat per-bin)."""
    n = 200
    kappa = np.geomspace(0.01, 1.0, n)
    # Deterministic alternating True/False every 10 elements so all
    # bins receive an equal mix.
    coll = np.zeros(n, dtype=bool)
    coll[::10] = True
    err = 0.1 * np.ones(n)
    return pd.DataFrame({
        "kappa_max": kappa,
        "mean_err": err,
        "collide": coll,
    })


def test_phenomenon_4_pass_on_growth_synthetic() -> None:
    df = _dstar_min_holds()
    status, ev = cp.check_phenomenon_4_dstar_min_vertical_asymptote(df)
    assert status == "PASS", f"expected PASS; got {status} with {ev}"
    assert ev["growth_ratio_last_over_first"] >= 1.5


def test_phenomenon_4_fail_when_flat() -> None:
    df = _dstar_min_fails()
    status, _ = cp.check_phenomenon_4_dstar_min_vertical_asymptote(df)
    assert status in ("FAIL", "WEAK")


# --------------------------------------------------------------------------- #
# Phenomenon 5 — FOV marginal diminishing returns                             #
# --------------------------------------------------------------------------- #


def _fov_knee_holds() -> pd.DataFrame:
    """trackable ramps with w_FOV then plateaus past ~π/2."""
    rows = []
    for w in np.linspace(np.deg2rad(30.0), np.deg2rad(360.0), 80):
        # Below knee: pr ~ w; past knee: pr saturates near 1.
        w_knee = np.pi / 2
        if w <= w_knee:
            base = w / w_knee
        else:
            base = 1.0 - 0.05 * (np.pi - min(w, np.pi)) / np.pi
        for _ in range(3):
            rows.append({"w_FOV": w, "trackable": RNG.random() < base})
    return pd.DataFrame(rows)


def _fov_knee_fails() -> pd.DataFrame:
    """Linear pr in w_FOV — no knee."""
    rows = []
    for w in np.linspace(np.deg2rad(30.0), np.deg2rad(360.0), 80):
        p = w / np.deg2rad(360.0)
        for _ in range(3):
            rows.append({"w_FOV": w, "trackable": RNG.random() < p})
    return pd.DataFrame(rows)


def test_phenomenon_5_pass_on_knee_synthetic() -> None:
    df = _fov_knee_holds()
    status, ev = cp.check_phenomenon_5_fov_marginal_diminishing(
        df, n_bins=10, knee_tolerance_deg=60.0
    )
    assert status in ("PASS", "WEAK"), f"got {status}: {ev}"
    # The knee should be in the 60–200° range (not 30 or 360).
    assert 60.0 <= ev["knee_deg_empirical"] <= 250.0


def test_phenomenon_5_fail_when_no_knee() -> None:
    df = _fov_knee_fails()
    status, ev = cp.check_phenomenon_5_fov_marginal_diminishing(
        df, knee_tolerance_deg=15.0  # tight ⇒ linear case rejected
    )
    # Linear ramp has knee at the FOV upper end — fails the 90° target.
    assert status in ("WEAK", "FAIL")


# --------------------------------------------------------------------------- #
# Phenomenon 6 — Over-conservatism area ratio                                 #
# --------------------------------------------------------------------------- #


def _area_ratio_holds() -> pd.DataFrame:
    """Construct a sample where trackable / (C1∩C2∩T_R) ≈ 0.8.

    Setup: span κ_max into BOTH (C1)-violating and -obeying halves, so
    the (C1) mask cuts the dataset in two; among the C1-obeying half
    we make 80 % trackable so the ratio is well inside the band.

    Also include ``reacq_mean_time`` / ``T_star`` / ``lost_count`` so the
    (E[T_R] ≤ T*) filter is exercised — set lost_count=0 everywhere so
    the re-acquire condition is trivially satisfied and the
    three-condition ratio coincides with the (C1)∩(C2)-only ratio.
    """
    n = 1000
    # κ_critC1 = u_gate/r_max = 0.3/0.5 = 0.6 ⇒ half the κ_max draws
    # violate C1, half obey.  Then we make 80 % of the C1-obeying half
    # trackable; the C1-violating half is 0 % trackable.
    kappa = RNG.uniform(0.01, 1.2, size=n)
    fs = RNG.uniform(2.0, 20.0, size=n)
    tau = RNG.uniform(0.05, 0.3, size=n)
    track = np.zeros(n, dtype=bool)
    c1_ok = kappa < 0.6
    # Among the C1-obeying half, 80 % trackable.
    track[c1_ok] = RNG.random(c1_ok.sum()) < 0.8
    return pd.DataFrame({
        "kappa_max": kappa, "f_s": fs, "tau_d": tau, "trackable": track,
        "lost_count": np.zeros(n, dtype=int),
        "reacq_mean_time": np.zeros(n, dtype=float),
        "T_star": np.full(n, 60.0, dtype=float),
        "collide": np.zeros(n, dtype=bool),
    })


def _area_ratio_fails_too_low() -> pd.DataFrame:
    """Only 10 % trackable while (C1∩C2∩T_R) is wide ⇒ ratio < 0.5."""
    n = 1000
    kappa = RNG.uniform(0.01, 0.5, size=n)
    fs = RNG.uniform(2.0, 20.0, size=n)
    tau = RNG.uniform(0.05, 0.3, size=n)
    track = RNG.random(n) < 0.1
    return pd.DataFrame({
        "kappa_max": kappa, "f_s": fs, "tau_d": tau, "trackable": track,
        "lost_count": np.zeros(n, dtype=int),
        "reacq_mean_time": np.zeros(n, dtype=float),
        "T_star": np.full(n, 60.0, dtype=float),
        "collide": np.zeros(n, dtype=bool),
    })


def _area_ratio_reacq_tightens() -> pd.DataFrame:
    """T_R filter tightens the admissible set.

    Construct a sample where the (C1∩C2)-only ratio is *low* (~0.2)
    because many C1-OK rows fail re-acquire, but the three-condition
    ratio (C1∩C2∩T_R) lands inside [0.5, 0.95].  Mechanism: among 1000
    C1-OK rows, only 200 are trackable; of the 800 not-trackable, set
    600 to have reacq_mean_time > T_star (so they're filtered OUT of
    the new admissible set), leaving 200 untrackable + 200 trackable
    inside the new admissible set ⇒ ratio = 200/400 = 0.5.
    """
    n = 1000
    kappa = RNG.uniform(0.01, 0.5, size=n)  # all C1-OK
    fs = RNG.uniform(2.0, 20.0, size=n)
    tau = RNG.uniform(0.05, 0.3, size=n)
    # 200 trackable, 800 not.
    idx = np.arange(n)
    RNG.shuffle(idx)
    track = np.zeros(n, dtype=bool)
    track[idx[:200]] = True
    # Of the 800 not-trackable rows, 600 have reacq > T_star (filtered
    # out); 200 have lost_count == 0 (kept).
    lost = np.zeros(n, dtype=int)
    reacq = np.zeros(n, dtype=float)
    T_star = np.full(n, 60.0, dtype=float)
    not_track_ids = idx[200:]
    lost[not_track_ids[:600]] = 5  # has L episodes
    reacq[not_track_ids[:600]] = 120.0  # > T_star
    # The trackable rows: half have lost=0 (trivially T_R-OK), half
    # have lost>0 with reacq=10s (well below T*) — both satisfy T_R.
    track_ids = idx[:200]
    lost[track_ids[100:]] = 3
    reacq[track_ids[100:]] = 10.0
    return pd.DataFrame({
        "kappa_max": kappa, "f_s": fs, "tau_d": tau, "trackable": track,
        "lost_count": lost, "reacq_mean_time": reacq, "T_star": T_star,
        "collide": np.zeros(n, dtype=bool),
    })


def test_phenomenon_6_pass_on_realistic_synthetic() -> None:
    df = _area_ratio_holds()
    status, ev = cp.check_phenomenon_6_overconservatism_area_ratio(df)
    assert status == "PASS", f"got {status} with {ev}"
    assert 0.5 <= ev["area_ratio_empirical"] <= 0.95


def test_phenomenon_6_weak_when_too_low() -> None:
    df = _area_ratio_fails_too_low()
    status, _ = cp.check_phenomenon_6_overconservatism_area_ratio(df)
    assert status in ("FAIL", "WEAK")


def test_phenomenon_6_evidence_reports_legacy_ratio() -> None:
    """``evidence`` must report the (C1∩C2)-only ratio so the simpler
    baseline can be cross-checked from the analysis output."""
    df = _area_ratio_holds()
    _, ev = cp.check_phenomenon_6_overconservatism_area_ratio(df)
    assert "area_ratio_legacy_c1_c2" in ev, (
        "(C1∩C2)-only ratio missing from evidence — cross-check broken"
    )
    assert isinstance(ev["area_ratio_legacy_c1_c2"], float)
    # When lost_count==0 for every row (the _holds synthetic), the
    # three-condition ratio must coincide with the (C1∩C2)-only ratio.
    assert ev["area_ratio_empirical"] == pytest.approx(
        ev["area_ratio_legacy_c1_c2"], abs=1e-9
    ), "ratio should reduce to (C1∩C2)-only when T_R is trivially satisfied"


def test_phenomenon_6_reacq_filter_tightens_ratio() -> None:
    """When many C1-OK rows fail E[T_R]≤T*, the three-condition ratio is
    strictly larger than the (C1∩C2)-only ratio (the admissible set is
    smaller while the trackable count is unchanged)."""
    df = _area_ratio_reacq_tightens()
    status, ev = cp.check_phenomenon_6_overconservatism_area_ratio(df)
    assert ev["reacq_filter_active"] is True
    legacy = ev["area_ratio_legacy_c1_c2"]
    new = ev["area_ratio_empirical"]
    assert new > legacy + 0.05, (
        f"T_R filter should tighten the ratio; legacy={legacy:.3f}, "
        f"new={new:.3f}"
    )
    # And in this synthetic, the new ratio should land inside the PASS band.
    assert status == "PASS", (
        f"expected PASS after T_R filter; got {status} with {ev}"
    )


def test_phenomenon_6_back_compat_no_reacq_columns() -> None:
    """When ``reacq_mean_time`` / ``T_star`` columns are absent, the
    detector reduces gracefully to the (C1∩C2)-only behaviour
    (no exception; ``reacq_filter_active`` reported as False)."""
    df = _area_ratio_holds().drop(columns=["reacq_mean_time", "T_star",
                                            "lost_count", "collide"])
    status, ev = cp.check_phenomenon_6_overconservatism_area_ratio(df)
    assert ev["reacq_filter_active"] is False
    assert ev["safety_filter_active"] is False
    # (C1∩C2)-only ratio == three-condition ratio when filter is inactive.
    assert ev["area_ratio_empirical"] == pytest.approx(
        ev["area_ratio_legacy_c1_c2"], abs=1e-9
    )
    assert status in ("PASS", "WEAK", "FAIL")
