"""v0.70: automatic self-weight (auto-SW).

When the toggle is on, the structural self-weight is computed in the
solver from w(x) = density * b_eff * h(x) and reported as a separate
"Self-weight" load case, factored as a permanent action (KFI*gamma_g in
ULS, sls_g in SLS, 1.0 unfactored). It shares the section assumptions of
the stiffness model, so it is exact for constant and linear-taper
sections and resolves to pure axial on vertical walls. These checks
cover the closed-form magnitude, equivalence with an equal manual Dead
Load, the off=no-contribution guard, global equilibrium, taper and wall
handling, ULS/SLS factoring, defaults, the cache key, and the
report/export plumbing.
"""
import numpy as np
import pytest

import bricos_data as data
import bricos_solver as solver
import bricos_export as export
from bricos_report import BricosReportGenerator

E_MOD = 30e6   # kN/m^2
L = 10.0       # m
B_EFF = 1.0    # m
H_SEC = 1.0    # m
DENSITY = 25.0           # kN/m^3 (reinforced concrete)
W_SW = DENSITY * B_EFF * H_SEC   # 25 kN/m for the constant 1x1 section


def _beam(n_spans=1, H=H_SEC, density=DENSITY, auto_sw=True, sw=0.0, **overrides):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam",
        "num_spans": n_spans,
        "L_list": [L] * 10,
        "Is_list": [H] * 10,
        "sw_list": [sw] * n_spans + [0.0] * (10 - n_spans),
        "E": E_MOD,
        "E_span_list": [E_MOD] * 10,
        "mesh_size": 0.5,
        "step_size": 0.5,
        "vehicle": {"loads": [], "spacing": []},
        "vehicleB": {"loads": [], "spacing": []},
        "vehicle_direction": "Forward",
        "use_shear_def": False,
        "b_eff": B_EFF,
        "phi_mode": "Manual",
        "phi": 1.0,
        "auto_selfweight": auto_sw,
        "density": density,
    })
    params.update(overrides)
    return params


def _portal(h_wall=5.0, wall_H=0.5, deck_H=H_SEC, density=DENSITY, auto_sw=True, **overrides):
    params = data.get_clear("A", "Frame")
    params.update({
        "mode": "Frame",
        "num_spans": 1,
        "L_list": [L] * 10,
        "Is_list": [deck_H] * 10,
        "sw_list": [0.0] * 10,
        "h_list": [h_wall, h_wall] + [0.0] * 9,
        "Iw_list": [wall_H] * 11,
        "E": E_MOD,
        "E_span_list": [E_MOD] * 10,
        "E_wall_list": [E_MOD] * 11,
        "mesh_size": 0.5,
        "step_size": 0.5,
        "vehicle": {"loads": [], "spacing": []},
        "vehicleB": {"loads": [], "spacing": []},
        "vehicle_direction": "Forward",
        "use_shear_def": False,
        "b_eff": B_EFF,
        "phi_mode": "Manual",
        "phi": 1.0,
        "soil": [],
        "surcharge": [],
        "auto_selfweight": auto_sw,
        "density": density,
    })
    params.update(overrides)
    return params


def _run(params):
    solver.clear_solver_cache()
    raw, _, _, err = solver.run_raw_analysis(params)
    assert err == 0
    solver.clear_solver_cache()
    return raw


def _combined(params, mode="Unfactored"):
    raw = _run(params)
    return raw, solver.combine_results(raw, params, mode)


# ==========================================
# 1. CLOSED-FORM MAGNITUDE
# ==========================================

def test_constant_section_self_weight_matches_udl_closed_form():
    raw, res = _combined(_beam())
    s1 = res["Self-weight"]["S1"]
    x = np.asarray(s1["x"])

    # Simply supported UDL fields with w = density * b_eff * H.
    np.testing.assert_allclose(s1["M"], W_SW * x * (L - x) / 2.0, atol=1e-6)
    np.testing.assert_allclose(s1["V"], W_SW * (L / 2.0 - x), atol=1e-6)
    np.testing.assert_allclose(s1["N"], 0.0, atol=1e-6)

    # Self-weight is its own static case; its reactions balance wL upward
    # (the raw "Reactions" dict tracks the manual Dead Load case only).
    eq = raw["Equilibrium"]["Self-weight"]
    assert eq["reactions_y"] == pytest.approx(W_SW * L, rel=1e-6)


def test_self_weight_scales_with_density_and_height():
    # Doubling either density or the section height doubles the weight.
    _, base = _combined(_beam())
    _, dense = _combined(_beam(density=2 * DENSITY))
    _, tall = _combined(_beam(H=2 * H_SEC))

    m0 = np.asarray(base["Self-weight"]["S1"]["M"])
    np.testing.assert_allclose(dense["Self-weight"]["S1"]["M"], 2.0 * m0, atol=1e-6)
    np.testing.assert_allclose(tall["Self-weight"]["S1"]["M"], 2.0 * m0, atol=1e-6)


# ==========================================
# 2. EQUIVALENCE WITH MANUAL DEAD LOAD
# ==========================================

def test_auto_self_weight_equals_equal_manual_dead_load():
    _, auto = _combined(_beam(auto_sw=True, sw=0.0))
    _, manual = _combined(_beam(auto_sw=False, sw=W_SW))

    np.testing.assert_allclose(auto["Self-weight"]["S1"]["M"],
                               manual["Dead Load"]["S1"]["M"], atol=1e-6)
    # The total envelope is identical whether the same magnitude is the
    # auto self-weight or a manual dead load.
    np.testing.assert_allclose(auto["Total Envelope"]["S1"]["M_max"],
                               manual["Total Envelope"]["S1"]["M_max"], atol=1e-6)
    np.testing.assert_allclose(auto["Total Envelope"]["S1"]["M_min"],
                               manual["Total Envelope"]["S1"]["M_min"], atol=1e-6)


def test_self_weight_and_manual_dead_load_superpose():
    # With auto-SW on AND a manual dead load, the total permanent moment is
    # the sum of both (each a UDL of W_SW); midspan = 2*W_SW*L^2/8.
    _, res = _combined(_beam(auto_sw=True, sw=W_SW))
    s1 = res["Total Envelope"]["S1"]
    assert np.max(s1["M_max"]) == pytest.approx(2.0 * W_SW * L ** 2 / 8.0, rel=1e-6)


# ==========================================
# 3. OFF = NO CONTRIBUTION
# ==========================================

def test_off_yields_no_self_weight_case():
    raw, res = _combined(_beam(auto_sw=False, sw=W_SW))
    assert raw["Self-weight"] == {}
    assert res["Self-weight"] == {}
    # The manual dead load is unaffected.
    assert np.max(res["Dead Load"]["S1"]["M"]) == pytest.approx(W_SW * L ** 2 / 8.0, rel=1e-6)


# ==========================================
# 4. GLOBAL EQUILIBRIUM
# ==========================================

def test_self_weight_equilibrium_residual_near_zero():
    raw = _run(_beam())
    eq = raw["Equilibrium"]["Self-weight"]
    assert eq["applied_y"] == pytest.approx(-W_SW * L, rel=1e-6)
    scale = max(abs(eq["applied_y"]), 1.0)
    assert abs(eq["residual_y"]) <= 1e-6 * scale + 1e-6
    assert abs(eq["residual_x"]) <= 1e-6 * scale + 1e-6


def test_linear_taper_total_weight_exact():
    # Linear taper h: h0 -> h1. The exact weight is density*b*L*(h0+h1)/2,
    # which the per-sub-element integration reproduces regardless of mesh.
    h0, h1 = 1.5, 0.5
    p = _beam(span_geom_0={'type': 1, 'shape': 1, 'vals': [h0, (h0 + h1) / 2.0, h1]})
    raw = _run(p)
    eq = raw["Equilibrium"]["Self-weight"]
    expected = DENSITY * B_EFF * L * (h0 + h1) / 2.0
    assert eq["applied_y"] == pytest.approx(-expected, rel=1e-6)
    scale = max(expected, 1.0)
    assert abs(eq["residual_y"]) <= 1e-6 * scale + 1e-6


# ==========================================
# 5. WALL SELF-WEIGHT (PORTAL)
# ==========================================

def test_wall_self_weight_in_vertical_reactions_and_axial():
    p = _portal()
    raw = _run(p)

    # Self-weight reactions carry the deck plus both walls.
    expected = DENSITY * B_EFF * (H_SEC * L + 2.0 * 0.5 * 5.0)
    eq = raw["Equilibrium"]["Self-weight"]
    assert eq["applied_y"] == pytest.approx(-expected, rel=1e-6)
    assert eq["reactions_y"] == pytest.approx(expected, rel=1e-6)
    assert abs(eq["residual_y"]) <= 1e-6 * max(expected, 1.0) + 1e-6

    # The walls' own weight resolves to axial compression that grows toward
    # the base (vertical members, is_gravity decomposition).
    res = solver.combine_results(raw, p, "Unfactored")
    for wid in ("W1", "W2"):
        n_wall = np.asarray(res["Self-weight"][wid]["N"])
        assert np.min(n_wall) < -1e-3  # compression present


# ==========================================
# 6. PARTIAL FACTORS
# ==========================================

def test_uls_factor_applies_kfi_times_gamma_g():
    kfi, gamma_g = 1.1, 1.35
    raw, res = _combined(_beam(KFI=kfi, gamma_g=gamma_g), "Design (ULS)")
    s1 = res["Self-weight"]["S1"]
    x = np.asarray(s1["x"])
    f = kfi * gamma_g
    np.testing.assert_allclose(s1["M"], f * W_SW * x * (L - x) / 2.0, rtol=1e-6, atol=1e-5)


def test_sls_factor_applies_sls_g():
    sls_g = 0.9
    raw, res = _combined(_beam(sls_g=sls_g), "Characteristic (SLS)")
    s1 = res["Self-weight"]["S1"]
    x = np.asarray(s1["x"])
    np.testing.assert_allclose(s1["M"], sls_g * W_SW * x * (L - x) / 2.0, rtol=1e-6, atol=1e-5)


# ==========================================
# 7. DEFAULTS, CACHE KEY AND PLUMBING
# ==========================================

def test_defaults_have_auto_selfweight_off():
    for params in (data.get_def(), data.get_clear("A", "Frame"), data.get_clear("B", "Beam")):
        assert params["auto_selfweight"] is False
        assert params["density"] == pytest.approx(25.0)


def test_cache_key_includes_auto_sw_inputs():
    # Toggling the feature or changing density must invalidate the solver
    # cache, so both keys belong in the cache parameter set.
    filtered = solver.solver_cache_params(data.get_def())
    assert "auto_selfweight" in filtered
    assert "density" in filtered


def test_report_and_export_recognize_self_weight_case():
    on, off = {"auto_selfweight": True}, {"auto_selfweight": False}

    assert BricosReportGenerator._system_has_component(on, "Self-weight") is True
    assert BricosReportGenerator._system_has_component(off, "Self-weight") is False
    assert export.system_has_case(on, "Self-weight") is True
    assert export.system_has_case(off, "Self-weight") is False
    assert "Self-weight" in export.EXPORT_CASES
