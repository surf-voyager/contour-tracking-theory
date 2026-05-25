"""Contract tests for sim_python.mc.analysis.check_phenomena.

These tests check:
- Registry has 6 entries with the correct names.
- Each function docstring documents its phenomenon number.
- Each function tolerates ``**kwargs`` and returns the (status, evidence)
  contract — status ∈ {"PASS", "FAIL", "WEAK"} and evidence is a dict.
- Each function is callable from ``sim_python.mc.analysis.check_phenomena``.

Per-phenomenon *behavioural* tests live in ``test_check_phenomena_*.py``
(one file per phenomenon, with synthetic data designed to hit PASS / FAIL).
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from sim_python.mc.analysis import check_phenomena as cp


_FUNCS = (
    cp.check_phenomenon_1_gate_island,
    cp.check_phenomenon_2_c2_hyperbolic_boundary,
    cp.check_phenomenon_3_lost_freq_blowup_near_kappa_crit,
    cp.check_phenomenon_4_dstar_min_vertical_asymptote,
    cp.check_phenomenon_5_fov_marginal_diminishing,
    cp.check_phenomenon_6_overconservatism_area_ratio,
)


def test_registry_has_six_entries() -> None:
    assert len(cp.CHECK_PHENOMENA) == 6
    names = [fn.__name__ for fn in cp.CHECK_PHENOMENA]
    for i, name in enumerate(names, 1):
        assert name.startswith(f"check_phenomenon_{i}_"), (
            f"#{i}: name {name} does not start with check_phenomenon_{i}_"
        )


def test_each_function_documents_phenomenon_number() -> None:
    """Each function's docstring must identify its phenomenon number."""
    for i, fn in enumerate(cp.CHECK_PHENOMENA, 1):
        ds = fn.__doc__ or ""
        assert f"Phenomenon {i}" in ds, (
            f"#{i}: docstring missing 'Phenomenon {i}'"
        )


def test_each_function_signature_accepts_kwargs() -> None:
    """Each phenomenon check must accept arbitrary extra kwargs."""
    for fn in _FUNCS:
        sig = inspect.signature(fn)
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD
                   for p in sig.parameters.values()), (
            f"{fn.__name__} must accept **kwargs (for forward-compat)"
        )


@pytest.mark.parametrize("fn", _FUNCS)
def test_returns_status_and_evidence_dict_contract(fn) -> None:
    """Empty or pathological df must return (status, dict) — not raise."""
    # Empty df ⇒ should return FAIL (missing columns) or WEAK (no data),
    # but NEVER raise.
    df = pd.DataFrame({})
    result = fn(df)
    assert isinstance(result, tuple) and len(result) == 2
    status, evidence = result
    assert status in ("PASS", "FAIL", "WEAK")
    assert isinstance(evidence, dict)


def test_module_importable_from_package_root() -> None:
    """Confirm import path used by Stage-04 will work."""
    from sim_python.mc.analysis.check_phenomena import CHECK_PHENOMENA  # noqa
    assert CHECK_PHENOMENA is cp.CHECK_PHENOMENA
