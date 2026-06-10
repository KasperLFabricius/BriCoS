import numpy as np
import pytest

import bricos_data as data
import bricos_solver as solver
from bricos_report import BricosReportGenerator


def _clear_solver_cache():
    fn = getattr(solver, "_run_raw_analysis_cached", None)
    if fn is not None and hasattr(fn, "clear"):
        fn.clear()


def _frame_params(**overrides):
    params = data.get_clear("A", "Frame")
    params.update({
        "mode": "Frame",
        "num_spans": 2,
        "L_list": [4.0, 30.0] + [0.0] * 8,
        "Is_list": [0.8] * 10,
        "sw_list": [5.0, 5.0] + [0.0] * 8,
        "h_list": [6.0, 6.0, 6.0] + [0.0] * 8,
        "Iw_list": [0.7] * 11,
        "E": 30e6,
        "E_span_list": [30e6] * 10,
        "E_wall_list": [30e6] * 11,
        "mesh_size": 2.0,
        "step_size": 1.0,
        "phi_mode": "Calculate",
        "vehicle": {"loads": [10.0], "spacing": [0.0]},
        "vehicle_direction": "Forward",
    })
    params.update(overrides)
    return params


def _run(params):
    _clear_solver_cache()
    raw, nodes, _, err = solver.run_raw_analysis(params)
    assert err == 0
    _clear_solver_cache()
    return raw


# --- phi curve (DK NA A.2.3.5(2)) ---

def test_phi_from_length_curve():
    assert solver.phi_from_length(2.0) == pytest.approx(1.25)
    assert solver.phi_from_length(5.0) == pytest.approx(1.25)
    assert solver.phi_from_length(27.5) == pytest.approx(1.25 - 22.5 / 225.0)
    assert solver.phi_from_length(50.0) == pytest.approx(1.05)
    assert solver.phi_from_length(120.0) == pytest.approx(1.05)


# --- L_inf methodologies ---

def test_determinant_mode_is_uniform_and_matches_table_62():
    params = _frame_params(phi_linf_mode="Determinant")
    raw = _run(params)

    # Components: left leg 6, spans 4 + 30, right leg 6 -> n=4, k=1.4
    lengths = [6.0, 4.0, 30.0, 6.0]
    L_phi = max(1.4 * sum(lengths) / 4.0, max(lengths))
    assert raw["phi_calc"] == pytest.approx(solver.phi_from_length(L_phi))
    assert raw["Phi Members"] == {}


def test_span_mode_produces_per_member_phi():
    params = _frame_params(phi_linf_mode="Span", phi_application="Per member")
    raw = _run(params)

    phi_s1 = solver.phi_from_length(4.0)   # 1.25 (short span)
    phi_s2 = solver.phi_from_length(30.0)  # 1.1389
    members = raw["Phi Members"]

    assert members["S1"] == pytest.approx(phi_s1)
    assert members["S2"] == pytest.approx(phi_s2)
    # Walls: max of adjacent spans.
    assert members["W1"] == pytest.approx(phi_s1)
    assert members["W2"] == pytest.approx(max(phi_s1, phi_s2))
    assert members["W3"] == pytest.approx(phi_s2)
    # Governing value reported as phi_calc.
    assert raw["phi_calc"] == pytest.approx(max(phi_s1, phi_s2))


def test_span_mode_governing_application_collapses_to_uniform_max():
    params = _frame_params(phi_linf_mode="Span", phi_application="Governing")
    raw = _run(params)

    assert raw["Phi Members"] == {}
    assert raw["phi_calc"] == pytest.approx(solver.phi_from_length(4.0))


def test_manual_phi_ignores_linf_mode():
    params = _frame_params(phi_mode="Manual", phi=1.17, phi_linf_mode="Span")
    raw = _run(params)

    assert raw["Phi Members"] == {}
    assert raw["phi_calc"] == pytest.approx(1.17)


# --- combination-stage behaviour ---

def _combine_raw(phi_members, phi_calc=1.25):
    env_a = np.array([10.0, 20.0])
    env_keys = (
        "M_max", "M_min", "V_max", "V_min", "N_max", "N_min",
        "def_x_max", "def_x_min", "def_y_max", "def_y_min",
    )
    base = {
        "x": np.array([0.0, 1.0]),
        "M": np.zeros(2), "V": np.zeros(2), "N": np.zeros(2),
        "def_x": np.zeros(2), "def_y": np.zeros(2),
        "loads": [], "L": 1.0, "cx": 1.0, "cy": 0.0,
        "ni": (0.0, 0.0), "nj": (1.0, 0.0), "ni_id": 200, "nj_id": 201,
    }
    env = {
        "S1": {**{k: env_a for k in env_keys}, "base": dict(base)},
        "S2": {**{k: env_a for k in env_keys}, "base": dict(base)},
    }
    return {
        "Selfweight": {}, "Soil": {}, "Surcharge": {},
        "Vehicle Envelope A": env, "Vehicle Envelope B": {},
        "phi_calc": phi_calc, "phi_log": [],
        "Phi Members": phi_members,
    }, env_a


def _combine_params(**overrides):
    params = {
        "KFI": 1.0, "gamma_g": 1.0, "gamma_j": 1.0,
        "gamma_veh": 1.0, "gamma_vehB": 1.0,
        "phi_mode": "Calculate",
        "combine_surcharge_vehicle": False,
        "phi_sls_mode": "Same",
    }
    params.update(overrides)
    return params


def test_combine_applies_per_member_phi_factors():
    raw, env_a = _combine_raw({"S1": 1.25, "S2": 1.10}, phi_calc=1.25)

    results = solver.combine_results(raw, _combine_params(), "Design (ULS)")

    np.testing.assert_allclose(results["Vehicle Envelope"]["S1"]["M_max"], env_a * 1.25)
    np.testing.assert_allclose(results["Vehicle Envelope"]["S2"]["M_max"], env_a * 1.10)
    assert results["f_vehA_map"]["S1"] == pytest.approx(1.25)
    assert results["f_vehA_map"]["S2"] == pytest.approx(1.10)


def test_sls_reduction_halves_dynamic_increment_per_member():
    raw, env_a = _combine_raw({"S1": 1.25, "S2": 1.10}, phi_calc=1.25)
    params = _combine_params(phi_sls_mode="Reduced")

    results = solver.combine_results(raw, params, "Characteristic (SLS)")

    np.testing.assert_allclose(results["Vehicle Envelope"]["S1"]["M_max"], env_a * 1.125)
    np.testing.assert_allclose(results["Vehicle Envelope"]["S2"]["M_max"], env_a * 1.05)
    assert results["phi_sls_mode_applied"] == "Reduced"


def test_sls_without_reduction_keeps_full_phi():
    raw, env_a = _combine_raw({}, phi_calc=1.20)

    results = solver.combine_results(raw, _combine_params(), "Characteristic (SLS)")

    np.testing.assert_allclose(results["Vehicle Envelope"]["S1"]["M_max"], env_a * 1.20)
    assert results["phi_sls_mode_applied"] == "Same"


def test_sls_reduction_applies_to_manual_phi():
    raw, env_a = _combine_raw({}, phi_calc=1.0)
    params = _combine_params(phi_mode="Manual", phi=1.20, phi_sls_mode="Reduced")

    results = solver.combine_results(raw, params, "Characteristic (SLS)")

    np.testing.assert_allclose(results["Vehicle Envelope"]["S1"]["M_max"], env_a * 1.10)


def test_sls_manual_value_replaces_member_values_uniformly():
    raw, env_a = _combine_raw({"S1": 1.25, "S2": 1.10}, phi_calc=1.25)
    params = _combine_params(phi_sls_mode="Manual", phi_sls=1.08)

    sls = solver.combine_results(raw, params, "Characteristic (SLS)")
    uls = solver.combine_results(raw, params, "Design (ULS)")

    # SLS: uniform user-defined value; ULS keeps per-member phi.
    np.testing.assert_allclose(sls["Vehicle Envelope"]["S1"]["M_max"], env_a * 1.08)
    np.testing.assert_allclose(sls["Vehicle Envelope"]["S2"]["M_max"], env_a * 1.08)
    np.testing.assert_allclose(uls["Vehicle Envelope"]["S1"]["M_max"], env_a * 1.25)
    np.testing.assert_allclose(uls["Vehicle Envelope"]["S2"]["M_max"], env_a * 1.10)
    assert sls["phi_sls_mode_applied"] == "Manual"


def test_combine_uses_manual_per_span_members_from_solver():
    raw, env_a = _combine_raw({"S1": 1.30, "S2": 1.05}, phi_calc=1.30)
    params = _combine_params(phi_mode="Manual", phi=1.0, phi_manual_scope="Per span")

    results = solver.combine_results(raw, params, "Design (ULS)")

    np.testing.assert_allclose(results["Vehicle Envelope"]["S1"]["M_max"], env_a * 1.30)
    np.testing.assert_allclose(results["Vehicle Envelope"]["S2"]["M_max"], env_a * 1.05)


def test_no_dynamic_factor_mode_ignores_phi_and_reduction():
    raw, env_a = _combine_raw({"S1": 1.25, "S2": 1.10}, phi_calc=1.25)
    params = _combine_params(phi_sls_mode="Reduced")

    results = solver.combine_results(raw, params, "Characteristic (No Dynamic Factor)")

    np.testing.assert_allclose(results["Vehicle Envelope"]["S1"]["M_max"], env_a)
    np.testing.assert_allclose(results["Vehicle Envelope"]["S2"]["M_max"], env_a)


def test_uls_unaffected_by_sls_settings():
    raw, env_a = _combine_raw({}, phi_calc=1.20)
    for sls_overrides in (dict(phi_sls_mode="Reduced"), dict(phi_sls_mode="Manual", phi_sls=1.02)):
        params = _combine_params(**sls_overrides)

        results = solver.combine_results(raw, params, "Design (ULS)")

        np.testing.assert_allclose(results["Vehicle Envelope"]["S1"]["M_max"], env_a * 1.20)


# --- manual per-span phi (solver) ---

def test_manual_per_span_phi_builds_member_map():
    params = _frame_params(
        phi_mode="Manual", phi_manual_scope="Per span",
        phi_span_list=[1.30, 1.07] + [1.0] * 8,
    )
    raw = _run(params)

    members = raw["Phi Members"]
    assert members["S1"] == pytest.approx(1.30)
    assert members["S2"] == pytest.approx(1.07)
    assert members["W1"] == pytest.approx(1.30)
    assert members["W2"] == pytest.approx(1.30)
    assert members["W3"] == pytest.approx(1.07)
    assert raw["phi_calc"] == pytest.approx(1.30)


def test_manual_global_phi_keeps_uniform_map():
    params = _frame_params(phi_mode="Manual", phi=1.17, phi_manual_scope="Global")
    raw = _run(params)

    assert raw["Phi Members"] == {}
    assert raw["phi_calc"] == pytest.approx(1.17)


# --- cache key and report display ---

def test_cache_key_excludes_sls_settings_but_includes_phi_inputs():
    params = data.get_def()
    filtered = solver.solver_cache_params(params)

    assert "phi_sls_mode" not in filtered
    assert "phi_sls" not in filtered
    assert filtered["phi_linf_mode"] == params["phi_linf_mode"]
    assert filtered["phi_application"] == params["phi_application"]
    assert filtered["phi_manual_scope"] == params["phi_manual_scope"]
    assert filtered["phi_span_list"] == params["phi_span_list"]


def test_report_phi_display_text_shows_range_for_per_member_values():
    p = {"phi_mode": "Calculate"}
    raw = {"Phi Members": {"S1": 1.25, "S2": 1.10}, "phi_calc": 1.25}
    assert BricosReportGenerator._phi_display_text(p, raw) == "Phi[1.10-1.25]"

    raw_uniform = {"Phi Members": {}, "phi_calc": 1.18}
    assert BricosReportGenerator._phi_display_text(p, raw_uniform) == "1.18"

    p_manual = {"phi_mode": "Manual", "phi": 1.05}
    assert BricosReportGenerator._phi_display_text(p_manual, None) == "1.05"

    # Manual per-span values flow through Phi Members like calculated ones.
    raw_manual_span = {"Phi Members": {"S1": 1.30, "S2": 1.05}, "phi_calc": 1.30}
    assert BricosReportGenerator._phi_display_text(p_manual, raw_manual_span) == "Phi[1.05-1.30]"
