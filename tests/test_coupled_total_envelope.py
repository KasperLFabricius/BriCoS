"""Exact coupled Total Envelope (v0.59): the traffic term envelopes each
vehicle step together with the adverse UDL outside that step's window,
plus the vehicle-absent alternative (full adverse UDL alone). Static and
footprint applications reproduce the former conservative superposition."""
import numpy as np
import pytest

import bricos_data as data
import bricos_solver as solver


def _run(params):
    solver.clear_solver_cache()
    raw, nodes, _, err = solver.run_raw_analysis(params)
    assert err == 0
    solver.clear_solver_cache()
    return raw


def _beam_params(n_spans=1, **overrides):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam",
        "num_spans": n_spans,
        "L_list": [10.0] * 10,
        "Is_list": [0.8] * 10,
        "sw_list": [5.0] * n_spans + [0.0] * (10 - n_spans),
        "E": 30e6,
        "E_span_list": [30e6] * 10,
        "mesh_size": 0.5,
        "step_size": 0.5,
        "phi_mode": "Manual",
        "phi": 1.0,
        "vehicle": {"loads": [10.0], "spacing": [0.0]},
        "vehicle_direction": "Forward",
        "udl_q": 2.5,
        "udl_gap": 2.0,
        "udl_mode": "Moving",
    })
    params.update(overrides)
    return params


def _combine(raw, mode="Unfactored", **overrides):
    params = {
        "KFI": 1.0, "gamma_g": 1.0, "gamma_j": 1.0,
        "gamma_veh": 1.0, "gamma_vehB": 1.0,
        "phi_mode": "Manual", "phi": 1.0,
        "combine_surcharge_vehicle": False,
        "gamma_udl": 0.56, "sls_udl": 0.40,
    }
    params.update(overrides)
    return solver.combine_results(raw, params, mode)


def _expected_traffic(res, raw, eid, key, step_keys=("Vehicle Steps A",)):
    """max/min over coupled steps and the vehicle-absent full adverse UDL."""
    f_v = res["f_vehA_map"].get(eid, res["f_vehA"])
    f_u = res["f_udl"]
    base, side = key.rsplit("_", 1)
    cands = [res["Traffic UDL"][eid][key]]
    for sk in step_keys:
        for s in raw.get(sk, []):
            r = s["res"][eid]
            cands.append(r[base] * f_v + r[f"{base}_udl_{side}"] * f_u)
    reduce = np.maximum.reduce if side == "max" else np.minimum.reduce
    return reduce(cands)


def test_coupled_total_matches_step_formula_all_components():
    raw = _run(_beam_params(n_spans=2))
    res = _combine(raw)

    for eid in ("S1", "S2"):
        for key in ("M_max", "M_min", "V_max", "V_min", "N_max", "N_min",
                    "def_y_max", "def_y_min"):
            expected = (res["Selfweight"][eid][key]
                        + _expected_traffic(res, raw, eid, key))
            np.testing.assert_allclose(
                res["Total Envelope"][eid][key], expected, atol=1e-9,
                err_msg=f"{eid}/{key}")


def test_coupled_total_respects_uls_factor_ratio():
    # The governing step depends on the vehicle/UDL factor ratio, so the
    # coupling must happen at combination time with the factored fields.
    raw = _run(_beam_params(n_spans=2))
    res = _combine(raw, "Design (ULS)", KFI=1.1, gamma_veh=1.4,
                   gamma_udl=0.56, phi=1.2)

    eid = "S1"
    expected = (res["Selfweight"][eid]["M_max"]
                + _expected_traffic(res, raw, eid, "M_max"))
    np.testing.assert_allclose(
        res["Total Envelope"][eid]["M_max"], expected, atol=1e-9)


def test_window_covering_short_deck_drops_udl_from_governing_case():
    # The user-reported case: vehicle + 10 m clear distances cover the
    # whole deck, so no step carries any UDL and the vehicle-governed
    # peak must NOT pick up the UDL term (the former bound added it).
    with_udl = _run(_beam_params(udl_gap=10.0, udl_q=5.0))
    without = _run(_beam_params(udl_gap=10.0, udl_q=0.0))

    res_with = _combine(with_udl, "Design (ULS)", KFI=1.1, gamma_veh=1.4)
    res_without = _combine(without, "Design (ULS)", KFI=1.1, gamma_veh=1.4)

    # Every step's windowed UDL fields vanish on the fully covered deck.
    for s in with_udl["Vehicle Steps A"]:
        assert np.all(s["res"]["S1"]["M_udl_max"] == 0.0)

    peak_with = np.max(res_with["Total Envelope"]["S1"]["M_max"])
    peak_without = np.max(res_without["Total Envelope"]["S1"]["M_max"])
    assert peak_with == pytest.approx(peak_without, rel=1e-9)


def test_footprint_application_reproduces_conservative_superposition():
    raw = _run(_beam_params(udl_gap=10.0, udl_footprint=True))
    res = _combine(raw)

    eid = "S1"
    for key in ("M_max", "M_min", "V_max", "V_min"):
        expected = (res["Selfweight"][eid][key]
                    + res["Vehicle Envelope"][eid][key]
                    + res["Traffic UDL"][eid][key])
        np.testing.assert_allclose(
            res["Total Envelope"][eid][key], expected, atol=1e-9,
            err_msg=f"{eid}/{key}")


def test_static_application_reproduces_conservative_superposition():
    raw = _run(_beam_params(udl_mode="Static"))
    res = _combine(raw)

    eid = "S1"
    expected = (res["Selfweight"][eid]["M_max"]
                + res["Vehicle Envelope"][eid]["M_max"]
                + res["Traffic UDL"][eid]["M_max"])
    np.testing.assert_allclose(
        res["Total Envelope"][eid]["M_max"], expected, atol=1e-9)


def test_vehicle_absent_alternative_governs_for_light_vehicle():
    # Tiny vehicle, heavy UDL, window covering the deck: the coupled steps
    # carry no UDL and a negligible vehicle, so the vehicle-absent full
    # adverse UDL must govern the total.
    raw = _run(_beam_params(
        vehicle={"loads": [0.1], "spacing": [0.0]},
        udl_gap=10.0, udl_q=10.0,
    ))
    res = _combine(raw)

    eid = "S1"
    expected = (res["Selfweight"][eid]["M_max"]
                + res["Traffic UDL"][eid]["M_max"])
    i_peak = int(np.argmax(res["Traffic UDL"][eid]["M_max"]))
    assert res["Total Envelope"][eid]["M_max"][i_peak] == pytest.approx(
        expected[i_peak], rel=1e-9)


def test_total_without_vehicle_is_perm_plus_full_adverse_udl():
    raw = _run(_beam_params(vehicle={"loads": [], "spacing": []}))
    res = _combine(raw)

    eid = "S1"
    expected = (res["Selfweight"][eid]["M_max"]
                + res["Traffic UDL"][eid]["M_max"])
    np.testing.assert_allclose(
        res["Total Envelope"][eid]["M_max"], expected, atol=1e-9)


def test_two_vehicles_couple_with_own_steps_and_cross_superpose():
    raw = _run(_beam_params(
        n_spans=2,
        vehicle={"loads": [10.0], "spacing": [0.0]},
        vehicleB={"loads": [5.0], "spacing": [0.0]},
    ))
    res = _combine(raw)

    eid = "S1"
    f_u = res["f_udl"]
    fA = res["f_vehA_map"].get(eid, res["f_vehA"])
    fB = res["f_vehB_map"].get(eid, res["f_vehB"])

    def coupled(step_key, rev_key, f_v):
        steps = (raw.get(step_key) or []) + (raw.get(rev_key) or [])
        return np.maximum.reduce(
            [s["res"][eid]["M"] * f_v + s["res"][eid]["M_udl_max"] * f_u
             for s in steps])

    cA = coupled("Vehicle Steps A", "Vehicle Steps A_Rev", fA)
    cB = coupled("Vehicle Steps B", "Vehicle Steps B_Rev", fB)
    envA = raw["Vehicle Envelope A"][eid]["M_max"] * fA
    envB = raw["Vehicle Envelope B"][eid]["M_max"] * fB
    ud = res["Traffic UDL"][eid]["M_max"]

    expected = (res["Selfweight"][eid]["M_max"]
                + np.maximum.reduce([ud, cA + envB, envA + cB]))
    np.testing.assert_allclose(
        res["Total Envelope"][eid]["M_max"], expected, atol=1e-9)

    # Tighter than (or equal to) the former full superposition.
    old_bound = (res["Selfweight"][eid]["M_max"]
                 + res["Vehicle Envelope"][eid]["M_max"] + ud)
    assert np.all(res["Total Envelope"][eid]["M_max"] <= old_bound + 1e-9)
