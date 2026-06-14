import numpy as np
import pytest

import bricos_data as data
import bricos_solver as solver


def _clear():
    solver.clear_solver_cache()


def _beam_params(**overrides):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam",
        "num_spans": 1,
        "L_list": [10.0] * 10,
        "Is_list": [1.0] * 10,
        "sw_list": [10.0] + [0.0] * 9,
        "E": 30e6,
        "E_span_list": [30e6] * 10,
        "mesh_size": 0.5,
        "step_size": 0.5,
        "phi_mode": "Manual",
        "phi": 1.0,
    })
    params.update(overrides)
    return params


def _frame_params(**overrides):
    params = data.get_clear("A", "Frame")
    params.update({
        "mode": "Frame",
        "num_spans": 1,
        "L_list": [10.0] + [0.0] * 9,
        "Is_list": [0.8] * 10,
        "sw_list": [10.0] + [0.0] * 9,
        "h_list": [6.0, 6.0] + [0.0] * 9,
        "Iw_list": [0.7] * 11,
        "E": 30e6,
        "E_span_list": [30e6] * 10,
        "E_wall_list": [30e6] * 11,
        "mesh_size": 0.5,
        "step_size": 0.5,
        "phi_mode": "Manual",
        "phi": 1.0,
        "soil": [
            {"wall_idx": 0, "face": "L", "h": 6.0, "q_top": 0.0, "q_bot": 20.0},
        ],
        "surcharge": [
            {"wall_idx": 0, "face": "R", "q": 5.0, "h": 6.0},
        ],
    })
    params.update(overrides)
    return params


def _run(params):
    _clear()
    raw, nodes, _, err = solver.run_raw_analysis(params)
    assert err == 0
    _clear()
    return raw


# --- equilibrium check values ---

def test_equilibrium_selfweight_beam_udl():
    raw = _run(_beam_params())
    eq = raw["Equilibrium"]["Dead Load"]

    # q = 10 kN/m over 10 m, acting downward.
    assert eq["applied_y"] == pytest.approx(-100.0, rel=1e-9)
    assert eq["reactions_y"] == pytest.approx(100.0, rel=1e-6)
    assert abs(eq["residual_x"]) < 1e-6
    assert abs(eq["residual_y"]) < 1e-6


def test_equilibrium_soil_and_surcharge_on_frame():
    raw = _run(_frame_params())

    eq_soil = raw["Equilibrium"]["Soil"]
    # Triangular pressure on the LEFT face: (20+0)/2 * 6 = 60 kN pushing
    # the wall toward +x; reactions must balance it.
    assert eq_soil["applied_x"] == pytest.approx(60.0, rel=1e-9)
    assert eq_soil["reactions_x"] == pytest.approx(-60.0, rel=1e-6)
    assert abs(eq_soil["residual_x"]) < 1e-6
    assert abs(eq_soil["residual_y"]) < 1e-6

    eq_surch = raw["Equilibrium"]["Surcharge"]
    # UI surcharge convention: face 'R' carries sign +1, i.e. it pushes the
    # same way as soil on face 'L' (+x, toward the bridge interior for the
    # left abutment): 5 * 6 = 30 kN toward +x.
    assert eq_surch["applied_x"] == pytest.approx(30.0, rel=1e-9)
    assert eq_surch["reactions_x"] == pytest.approx(-30.0, rel=1e-6)
    assert abs(eq_surch["residual_x"]) < 1e-6

    eq_sw = raw["Equilibrium"]["Dead Load"]
    assert eq_sw["applied_y"] == pytest.approx(-100.0, rel=1e-9)
    assert abs(eq_sw["residual_y"]) < 1e-6


def test_equilibrium_symmetric_soil_cancels_but_reports_has_loads():
    # Mirrored earth pressure on both walls cancels in the global x-sum.
    # The has_loads flag must still mark the case as loaded so the report
    # shows a PASS row instead of "(no loads)".
    params = _frame_params(
        soil=[
            {"wall_idx": 0, "face": "L", "h": 6.0, "q_top": 0.0, "q_bot": 20.0},
            {"wall_idx": 1, "face": "R", "h": 6.0, "q_top": 0.0, "q_bot": 20.0},
        ],
        surcharge=[],
    )
    raw = _run(params)

    eq_soil = raw["Equilibrium"]["Soil"]
    assert eq_soil["has_loads"] is True
    assert abs(eq_soil["applied_x"]) < 1e-9
    assert abs(eq_soil["residual_x"]) < 1e-6
    assert abs(eq_soil["residual_y"]) < 1e-6
    # v0.60: the opposing parts are reported separately so the report can
    # show the applied loading despite the zero net sum: each wall carries
    # (20+0)/2 * 6 = 60 kN, in opposite directions.
    assert eq_soil["applied_x_pos"] == pytest.approx(60.0, rel=1e-9)
    assert eq_soil["applied_x_neg"] == pytest.approx(-60.0, rel=1e-9)

    assert raw["Equilibrium"]["Surcharge"]["has_loads"] is False
    assert raw["Equilibrium"]["Dead Load"]["has_loads"] is True


def test_equilibrium_inclined_span_selfweight():
    # Inclined span: gravity totals must still balance vertically.
    params = _beam_params()
    params["span_geom_0"] = {
        "type": 1, "shape": 0, "vals": [1.0, 1.0, 1.0],
        "align_type": 1, "incline_mode": 0, "incline_val": 10.0,
    }
    raw = _run(params)
    eq = raw["Equilibrium"]["Dead Load"]

    # q applies per metre of (inclined) member length.
    member_len = np.hypot(10.0, 1.0)
    assert eq["applied_y"] == pytest.approx(-10.0 * member_len, rel=1e-6)
    assert abs(eq["residual_x"]) < 1e-6
    assert abs(eq["residual_y"]) < 1e-6


# --- single-factorization batching ---

def test_single_solve_matches_chunked_fallback(monkeypatch):
    params = _beam_params(
        vehicle={"loads": [10.0, 12.0], "spacing": [0.0, 2.0]},
        vehicle_direction="Both",
    )

    _clear()
    raw_single, _, _, err = solver.run_raw_analysis(params)
    assert err == 0

    monkeypatch.setattr(solver, "MAX_SINGLE_SOLVE_RHS", 0)
    _clear()
    raw_chunked, _, _, err = solver.run_raw_analysis(params)
    assert err == 0
    _clear()

    env_s = raw_single["Vehicle Envelope A"]
    env_c = raw_chunked["Vehicle Envelope A"]
    assert set(env_s.keys()) == set(env_c.keys())
    for eid in env_s:
        for key in ("M_max", "M_min", "V_max", "V_min", "N_max", "N_min"):
            np.testing.assert_allclose(env_s[eid][key], env_c[eid][key], rtol=1e-9, atol=1e-9)

    steps_s = raw_single["Vehicle Steps A"]
    steps_c = raw_chunked["Vehicle Steps A"]
    assert len(steps_s) == len(steps_c)
    for sf, sl in zip(steps_s, steps_c):
        assert sf["x"] == pytest.approx(sl["x"])
        for eid in sl["res"]:
            np.testing.assert_allclose(sf["res"][eid]["M"], sl["res"][eid]["M"], rtol=1e-9, atol=1e-9)


# --- session-state result cache ---

def test_solver_cache_returns_same_object_and_clears():
    params = _beam_params()

    _clear()
    first = solver.run_raw_analysis(params)
    second = solver.run_raw_analysis(params)
    # Cache hit: identical object, no pickle round-trip.
    assert first is second

    solver.clear_solver_cache()
    third = solver.run_raw_analysis(params)
    assert third is not first
    np.testing.assert_allclose(
        third[0]["Dead Load"]["S1"]["M"], first[0]["Dead Load"]["S1"]["M"]
    )
    _clear()


def test_solver_cache_distinguishes_solver_relevant_params():
    params = _beam_params()
    _clear()
    base = solver.run_raw_analysis(params)

    renamed = dict(params)
    renamed["name"] = "Renamed"
    assert solver.run_raw_analysis(renamed) is base  # cosmetic: cache hit

    changed = dict(params)
    changed["sw_list"] = [20.0] + [0.0] * 9
    assert solver.run_raw_analysis(changed) is not base  # solver-relevant: miss
    _clear()
