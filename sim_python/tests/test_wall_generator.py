"""Tests for sim_python.scenarios.wall_generator.

Covers the Serret-Frenet conventions for the tangent angle and the κ
sign, plus the straight / arc / cubic-spline generators.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_python.scenarios.wall_generator import (
    Wall,
    gen_arc,
    gen_cspline,
    gen_straight,
)


# --------------------------------------------------------------------------- #
# Test 1 — straight wall: κ ≡ 0 + endpoint at (length, 0)                     #
# --------------------------------------------------------------------------- #


def test_straight_wall_shape_and_kappa() -> None:
    w = gen_straight(length=10.0, side="L")
    # Endpoint
    assert abs(w.points[-1, 0] - 10.0) < 1e-12
    assert abs(w.points[-1, 1]) < 1e-12
    # κ ≡ 0
    assert np.max(np.abs(w.kappa)) < 1e-12
    # tangent γ_p ≡ 0
    assert np.max(np.abs(w.tangent)) < 1e-12
    # side recorded
    assert w.side == "L"
    # arc length monotone
    assert np.all(np.diff(w.s) > 0)


# --------------------------------------------------------------------------- #
# Test 2 — arc wall: half-circle endpoint check + |κ| = 1/R                   #
# --------------------------------------------------------------------------- #


def test_arc_wall_half_circle_geometry() -> None:
    R = 5.0
    # Half circle: arc length = π R, left-turn from (0,0,+x) ends at (0, 2R).
    w = gen_arc(radius=R, arc_length=np.pi * R, side="L")
    last = w.points[-1]
    assert abs(last[0]) < 1e-6, f"arc end x = {last[0]} not ~0"
    assert abs(last[1] - 2.0 * R) < 1e-6, f"arc end y = {last[1]} not ~2R"
    # κ = +1/R everywhere (LEFT turn)
    assert np.allclose(w.kappa, +1.0 / R)
    # Tangent at end = π (180°)
    assert abs(w.tangent[-1] - np.pi) < 1e-6


def test_arc_wall_right_turn_sign() -> None:
    R = 4.0
    # Quarter arc to the right: tangent at end = -π/2.
    w = gen_arc(radius=R, arc_length=np.pi * R / 2.0, side="R")
    assert np.allclose(w.kappa, -1.0 / R)
    assert abs(w.tangent[-1] - (-np.pi / 2.0)) < 1e-6
    # Endpoint at (R, -R)
    last = w.points[-1]
    assert abs(last[0] - R) < 1e-6
    assert abs(last[1] - (-R)) < 1e-6


# --------------------------------------------------------------------------- #
# Test 3 — cspline: |κ| ≤ kappa_max + determinism in seed                     #
# --------------------------------------------------------------------------- #


def test_cspline_bounds_and_determinism() -> None:
    kappa_max = 0.05
    kappa_dot_max = 0.01
    w1 = gen_cspline(
        kappa_max=kappa_max,
        kappa_dot_max=kappa_dot_max,
        length=80.0,
        side="L",
        seed=2026,
    )
    w2 = gen_cspline(
        kappa_max=kappa_max,
        kappa_dot_max=kappa_dot_max,
        length=80.0,
        side="L",
        seed=2026,
    )
    # |κ| bounded by spec (post-clip)
    assert np.max(np.abs(w1.kappa)) <= kappa_max + 1e-12
    # Same seed ⇒ same wall (Generator-based determinism)
    assert np.array_equal(w1.points, w2.points)
    assert np.array_equal(w1.kappa, w2.kappa)
    # Different seeds ⇒ different walls
    w3 = gen_cspline(
        kappa_max=kappa_max,
        kappa_dot_max=kappa_dot_max,
        length=80.0,
        side="L",
        seed=99,
    )
    assert not np.array_equal(w1.kappa, w3.kappa)


# --------------------------------------------------------------------------- #
# Test 4 — Wall dataclass validates shapes + invalid side                     #
# --------------------------------------------------------------------------- #


def test_wall_validation_raises_on_bad_shapes() -> None:
    pts = np.zeros((10, 2))
    bad_tan = np.zeros(5)  # wrong shape
    with pytest.raises(ValueError, match="tangent"):
        Wall(
            points=pts,
            tangent=bad_tan,
            kappa=np.zeros(10),
            side="L",
            s=np.arange(10, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="side"):
        Wall(
            points=pts,
            tangent=np.zeros(10),
            kappa=np.zeros(10),
            side="Z",
            s=np.arange(10, dtype=np.float64),
        )


# --------------------------------------------------------------------------- #
# Test 5 — generator input validation                                         #
# --------------------------------------------------------------------------- #


def test_generator_input_validation() -> None:
    with pytest.raises(ValueError, match="length"):
        gen_straight(length=-1.0, side="L")
    with pytest.raises(ValueError, match="radius"):
        gen_arc(radius=0.0, arc_length=1.0, side="L")
    with pytest.raises(ValueError, match="arc_length"):
        gen_arc(radius=1.0, arc_length=-1.0, side="L")
    with pytest.raises(ValueError, match="kappa_max"):
        gen_cspline(kappa_max=0.0, kappa_dot_max=1.0, length=10.0, side="L", seed=0)
    with pytest.raises(ValueError, match="side"):
        gen_straight(length=10.0, side="X")


# --------------------------------------------------------------------------- #
# Test 6 — defensive: no NaN/inf in any column                                #
# --------------------------------------------------------------------------- #


def test_all_generators_produce_no_nan() -> None:
    for w in (
        gen_straight(length=10.0, side="L"),
        gen_arc(radius=3.0, arc_length=2.0 * np.pi * 3.0, side="R"),
        gen_cspline(kappa_max=0.1, kappa_dot_max=0.02, length=50.0, side="L", seed=7),
    ):
        assert np.all(np.isfinite(w.points))
        assert np.all(np.isfinite(w.tangent))
        assert np.all(np.isfinite(w.kappa))
        assert np.all(np.isfinite(w.s))


# --------------------------------------------------------------------------- #
# Test 7 — gen_cspline geometric κ overshoot check                            #
# --------------------------------------------------------------------------- #
#
# Cubic-spline interpolation can produce raw κ(s) overshooting the
# configured kappa_max bound by ~25-30 %.  The generator mitigates this
# with an a-posteriori np.clip() on the κ array.  We verify that:
#
#   (a) the stored ``Wall.kappa`` array is bounded by kappa_max within
#       float-roundoff tolerance, AND
#   (b) the *geometric* κ reconstructed from finite differences of the
#       integrated (x, y) points also stays within kappa_max × (1 + 5 %).
#
# (b) is the load-bearing check: if (b) failed but (a) passed, the
# integration would "remember" pre-clip overshoot — but because the
# integration in gen_cspline uses the clipped κ directly, (a) and (b) agree
# up to numerical-differentiation noise (~ds² ≈ 2.5e-3 for ds=0.05 m).


def _geometric_kappa(points: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Reconstruct signed κ(s) numerically from (x, y) via finite differences.

    κ = (x' y'' − y' x'') / (x'² + y'²)^{3/2}
    """
    dx = np.gradient(points[:, 0], s)
    dy = np.gradient(points[:, 1], s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)
    num = dx * ddy - dy * ddx
    den = (dx**2 + dy**2) ** 1.5
    return num / np.maximum(den, 1e-12)


@pytest.mark.parametrize("kappa_max", [0.01, 0.05, 0.2, 0.5, 1.0])
def test_gen_cspline_kappa_overshoot(kappa_max: float) -> None:
    """Actual geometric κ(s) must respect kappa_max within 5 % tol.

    10 walls per kappa_max with different seeds; geometric κ from finite
    differences must satisfy max|κ(s)| ≤ kappa_max × 1.05.
    """
    tol = 0.05  # 5 % overshoot allowed (covers ds² FD truncation)
    length = 200.0
    # kappa_dot_max scaled with kappa_max so n_ctrl ≈ const → comparable test
    # across kappa_max values (avoids one config getting orders-of-magnitude
    # more or fewer control nodes).
    kappa_dot_max = kappa_max * 0.5
    for seed in range(10):
        w = gen_cspline(
            kappa_max=kappa_max,
            kappa_dot_max=kappa_dot_max,
            length=length,
            side="L",
            seed=seed,
            ds=0.05,
        )
        # (a) stored array within roundoff of clip
        max_arr = float(np.max(np.abs(w.kappa)))
        assert max_arr <= kappa_max * (1.0 + 1e-9), (
            f"stored kappa exceeds kappa_max for seed={seed}: "
            f"max|κ|={max_arr}, kappa_max={kappa_max}"
        )
        # (b) geometric κ from positions within 5 % tolerance
        # (5 % budgets the O(ds²) finite-difference truncation; for ds=0.05 m
        # the truncation is ~0.05² · κ'' ≪ 5 %·κ_max in practice).  Trim
        # 5 samples each end (gradient edge effect).
        kg = _geometric_kappa(w.points, w.s)
        max_geo = float(np.max(np.abs(kg[5:-5])))
        bound = kappa_max * (1.0 + tol)
        assert max_geo <= bound, (
            f"geometric kappa overshoots for seed={seed}, kappa_max={kappa_max}: "
            f"max|κ_geo|={max_geo:.6f} > bound={bound:.6f} "
            f"(overshoot {100.0 * (max_geo / kappa_max - 1.0):.2f} %)"
        )
