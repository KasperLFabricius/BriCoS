import numpy as np
import pytest

import bricos_data as data
import bricos_solver as solver


def _clear():
    solver.clear_solver_cache()


def _beam_params(n_spans=2, **overrides):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam",
        "num_spans": n_spans,
        "L_list": [10.0] * 10,
        "Is_list": [0.8] * 10,
        "sw_list": [0.0] * 10,
        "E": 30e6,
        "E_span_list": [30e6] * 10,
        "mesh_size": 0.5,
        "step_size": 0.5,
        "phi_mode": "Manual",
        "phi": 1.0,
    })
    params.update(overrides)
    return params


def _run(params):
    _clear()
    raw, nodes, _, err = solver.run_raw_analysis(params)
    assert err == 0
    _clear()
    return raw


# --- helper ---

def test_udl_line_load_helper():
    # Direct line load [kN/m], like selfweight: the user accounts for the
    # effective width in the input.
    assert data.udl_line_load({"udl_q": 7.5}) == pytest.approx(7.5)
    # inactive / invalid definitions never enter the analysis
    assert data.udl_line_load({"udl_q": 0.0}) == 0.0
    assert data.udl_line_load({"udl_q": -1.0}) == 0.0
    assert data.udl_line_load({"udl_q": float("nan")}) == 0.0
    assert data.udl_line_load({}) == 0.0


def test_validation_rejects_negative_udl_definitions():
    params = _beam_params(udl_q=-1.0)
    errors = data.validate_analysis_inputs(params, "System A")
    assert any("Traffic UDL line load" in e for e in errors)

    params = _beam_params(udl_q=1.0, udl_gap=-5.0)
    errors = data.validate_analysis_inputs(params, "System A")
    assert any("distance to vehicle" in e for e in errors)


# --- adverse envelope correctness ---

def test_udl_single_span_matches_full_udl_closed_form():
    q = 2.5
    params = _beam_params(n_spans=1, udl_q=q)
    raw = _run(params)
    env = raw["Traffic UDL"]["S1"]

    # Simply supported: every segment contributes sagging (positive M in
    # the engineering sign convention, v0.58), so the adverse max envelope
    # equals the fully loaded closed form and the min envelope stays zero.
    assert np.max(env["M_max"]) == pytest.approx(q * 10.0**2 / 8.0, rel=1e-6)
    assert np.min(env["M_min"]) == pytest.approx(0.0, abs=1e-9)


def test_udl_two_span_adverse_envelope_bounds_span_patterns():
    q = 2.5
    udl_raw = _run(_beam_params(udl_q=q))
    env = udl_raw["Traffic UDL"]

    # Pattern cases through the selfweight path (same load type/mesh).
    pat_s1 = _run(_beam_params(sw_list=[q, 0.0] + [0.0] * 8))["Selfweight"]
    pat_s2 = _run(_beam_params(sw_list=[0.0, q] + [0.0] * 8))["Selfweight"]
    pat_both = _run(_beam_params(sw_list=[q, q] + [0.0] * 8))["Selfweight"]

    for eid in ("S1", "S2"):
        pattern_max = np.maximum.reduce([
            np.zeros_like(pat_s1[eid]["M"]),
            pat_s1[eid]["M"], pat_s2[eid]["M"], pat_both[eid]["M"],
        ])
        pattern_min = np.minimum.reduce([
            np.zeros_like(pat_s1[eid]["M"]),
            pat_s1[eid]["M"], pat_s2[eid]["M"], pat_both[eid]["M"],
        ])
        # Segment-level adverse loading can only be at least as adverse as
        # any span-level pattern.
        assert np.all(env[eid]["M_max"] >= pattern_max - 1e-9)
        assert np.all(env[eid]["M_min"] <= pattern_min + 1e-9)

    # For bending the influence sign changes only at supports, so the
    # segment-level envelope must EQUAL the best span-level pattern.
    # Sagging (positive M, engineering convention) in S1 is worst with
    # only S1 loaded; hogging (negative) over the interior support is
    # worst with both spans loaded (-qL^2/8 for two equal spans).
    np.testing.assert_allclose(
        np.max(env["S1"]["M_max"]), np.max(pat_s1["S1"]["M"]), rtol=1e-6)
    np.testing.assert_allclose(
        np.min(env["S1"]["M_min"]), np.min(pat_both["S1"]["M"]), rtol=1e-6)
    assert np.min(env["S1"]["M_min"]) == pytest.approx(-q * 10.0**2 / 8.0, rel=1e-6)


def test_udl_inactive_when_q_zero():
    raw = _run(_beam_params(udl_q=0.0))
    assert raw["Traffic UDL"] == {}
    assert raw["udl_line_load"] == 0.0


# --- combination-stage behaviour ---

def _combine(raw, result_mode, **param_overrides):
    params = {
        "KFI": 1.0, "gamma_g": 1.0, "gamma_j": 1.0,
        "gamma_veh": 1.0, "gamma_vehB": 1.0,
        "phi_mode": "Manual", "phi": 1.0,
        "combine_surcharge_vehicle": False,
        "gamma_udl": 0.56, "sls_udl": 0.40,
    }
    params.update(param_overrides)
    return solver.combine_results(raw, params, result_mode)


def test_udl_factoring_per_result_mode_and_no_phi():
    raw = _run(_beam_params(udl_q=2.5))
    base = raw["Traffic UDL"]["S1"]["M_max"]

    uls = _combine(raw, "Design (ULS)", KFI=1.1, gamma_udl=0.56, phi=1.30)
    np.testing.assert_allclose(uls["Traffic UDL"]["S1"]["M_max"], base * 1.1 * 0.56)
    assert uls["f_udl"] == pytest.approx(1.1 * 0.56)

    sls = _combine(raw, "Characteristic (SLS)", sls_udl=0.40, phi=1.30)
    np.testing.assert_allclose(sls["Traffic UDL"]["S1"]["M_max"], base * 0.40)

    nodyn = _combine(raw, "Characteristic (No Dynamic Factor)")
    np.testing.assert_allclose(nodyn["Traffic UDL"]["S1"]["M_max"], base)


def test_total_envelope_couples_udl_with_vehicle_steps():
    # v0.59: the traffic term is the EXACT coupled envelope - per step,
    # vehicle + adverse UDL outside that step's window - enveloped with
    # the vehicle-absent full adverse UDL. The former independent
    # superposition (vehicle envelope + full adverse UDL) counted the
    # UDL share of the window region the vehicle displaces.
    raw = _run(_beam_params(
        udl_q=2.5,
        vehicle={"loads": [10.0], "spacing": [0.0]},
        vehicle_direction="Forward",
    ))
    res = _combine(raw, "Design (ULS)", KFI=1.1, gamma_veh=1.4, gamma_udl=0.56)

    eid = "S1"
    f_v = res["f_vehA_map"].get(eid, res["f_vehA"])
    f_u = res["f_udl"]
    steps = raw["Vehicle Steps A"]
    cands_max = [res["Traffic UDL"][eid]["M_max"]]  # vehicle-absent
    for s in steps:
        r = s["res"][eid]
        cands_max.append(r["M"] * f_v + r["M_udl_max"] * f_u)
    expected_max = (res["Selfweight"][eid]["M_max"]
                    + np.maximum.reduce(cands_max))
    np.testing.assert_allclose(
        res["Total Envelope"][eid]["M_max"], expected_max, atol=1e-9)

    # And strictly below the former superposition where the window bites.
    old_bound = (res["Selfweight"][eid]["M_max"]
                 + res["Vehicle Envelope"][eid]["M_max"]
                 + res["Traffic UDL"][eid]["M_max"])
    assert np.all(res["Total Envelope"][eid]["M_max"] <= old_bound + 1e-9)
    assert np.max(old_bound - res["Total Envelope"][eid]["M_max"]) > 1e-6


def test_cache_key_excludes_udl_factors_but_includes_geometry():
    params = data.get_def()
    filtered = solver.solver_cache_params(params)

    assert "gamma_udl" not in filtered
    assert "sls_udl" not in filtered
    assert filtered["udl_q"] == params["udl_q"]
    assert filtered["udl_gap"] == params["udl_gap"]
