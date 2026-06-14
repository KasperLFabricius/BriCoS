"""Engineering sign convention (v0.58): sagging-positive bending,
tension-positive axial force, global-axes deflections - and nodal
reactions staying in the global CCW-positive convention."""
import numpy as np
import pytest

import bricos_data as data
import bricos_results_ui as results_ui
import bricos_solver as solver


def _run(params):
    solver.clear_solver_cache()
    raw, nodes, _, err = solver.run_raw_analysis(params)
    assert err == 0
    solver.clear_solver_cache()
    return raw, nodes


def _ss_beam_params(**overrides):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam", "num_spans": 1,
        "L_list": [10.0] + [0.0] * 9,
        "Is_list": [0.8] * 10,
        "sw_list": [10.0] + [0.0] * 9,
        "E": 30e6, "E_span_list": [30e6] * 10,
        "mesh_size": 0.5, "step_size": 1.0,
        "phi_mode": "Manual", "phi": 1.0,
    })
    params.update(overrides)
    return params


def _frame_params():
    params = data.get_clear("A", "Frame")
    params.update({
        "mode": "Frame", "num_spans": 1,
        "L_list": [10.0] + [0.0] * 9,
        "Is_list": [0.8] * 10,
        "sw_list": [10.0] + [0.0] * 9,
        "h_list": [6.0, 6.0] + [0.0] * 9,
        "Iw_list": [0.7] * 11,
        "E": 30e6, "E_span_list": [30e6] * 10, "E_wall_list": [30e6] * 11,
        "mesh_size": 0.5, "step_size": 1.0,
        "phi_mode": "Manual", "phi": 1.0,
    })
    return params


def test_simply_supported_beam_signs():
    raw, _ = _run(_ss_beam_params())
    sw = raw["Dead Load"]["S1"]
    i_mid = len(sw["x"]) // 2

    # Sagging positive: closed form wL^2/8 = +125 kNm.
    assert sw["M"][i_mid] == pytest.approx(10.0 * 10.0**2 / 8.0, rel=1e-6)
    # Classical shear: V(0+) = +R = +wL/2.
    assert sw["V"][0] == pytest.approx(50.0, rel=1e-6)
    # Deflection in global axes: downwards negative.
    assert sw["def_y"][i_mid] < 0.0
    # No axial force in a simply supported beam under gravity.
    assert abs(sw["N"][i_mid]) < 1e-6


def test_moving_load_envelope_sagging_is_max():
    params = _ss_beam_params(
        vehicle={"loads": [10.0], "spacing": [0.0]},
        vehicle_direction="Forward",
    )
    raw, _ = _run(params)
    env = raw["Vehicle Envelope A"]["S1"]

    # Single moving axle on a simply supported span: max sagging PL/4
    # lands in M_max (positive); the min envelope stays ~zero.
    P = 10.0 * 9.81
    assert np.max(env["M_max"]) == pytest.approx(P * 10.0 / 4.0, rel=1e-6)
    assert np.min(env["M_min"]) > -1e-6


def test_frame_compression_and_corner_hogging_signs():
    raw, _ = _run(_frame_params())
    w1 = raw["Dead Load"]["W1"]
    s1 = raw["Dead Load"]["S1"]
    i_mid = len(s1["x"]) // 2

    # Walls carry the deck: compression negative (~ -wL/2).
    assert w1["N"][len(w1["N"]) // 2] == pytest.approx(-50.0, rel=1e-6)
    # Deck: sagging positive at midspan, hogging negative at the corners.
    assert s1["M"][i_mid] > 0.0
    assert s1["M"][0] < 0.0
    assert s1["M"][-1] < 0.0
    # Deck deflection downward negative.
    assert s1["def_y"][i_mid] < 0.0


def test_reaction_envelope_moments_keep_nodal_convention():
    # The reaction tables derive Mz from the (now sagging-positive) M field;
    # the result must still equal the solver's nodal reactions (global
    # CCW-positive convention) computed from the element end forces.
    params = _frame_params()
    raw, nodes = _run(params)
    res = solver.combine_results(raw, params, "Unfactored")

    env = results_ui.get_reaction_envelope(
        res["Total Envelope"], nodes, "Frame", raw.get("Restrained Nodes"))
    legacy = raw["Reactions"]

    checked = 0
    for nid in raw.get("Restrained Nodes", []):
        assert env[nid]["Mz_max"] == pytest.approx(legacy[nid][2], abs=1e-6)
        assert env[nid]["Mz_min"] == pytest.approx(legacy[nid][2], abs=1e-6)
        # Ry keeps its historical member-end-force sign convention in the
        # reaction tables (it derives from N/V, untouched by the M flip);
        # only the magnitude is compared here.
        assert abs(env[nid]["Ry_max"]) == pytest.approx(abs(legacy[nid][1]), abs=1e-6)
        checked += 1
    assert checked >= 2
