import math

import numpy as np
import pytest

import bricos_data as data
import bricos_solver as solver


def _beam_params(load_q=10.0, *, rise=2.0, vehicle_loads=None, mesh_size=0.5, step_size=0.5):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam",
        "num_spans": 1,
        "L_list": [10.0] * 10,
        "Is_list": [1.0] * 10,
        "sw_list": [load_q] + [0.0] * 9,
        "E": 30e6,
        "E_span_list": [30e6] * 10,
        "mesh_size": mesh_size,
        "step_size": step_size,
        "vehicle": vehicle_loads or {"loads": [], "spacing": []},
        "vehicleB": {"loads": [], "spacing": []},
        "vehicle_direction": "Forward",
        "use_shear_def": False,
        "b_eff": 1.0,
        "phi_mode": "Manual",
        "phi": 1.0,
        "span_geom_0": {
            "type": 1,
            "shape": 0,
            "vals": [1.0, 1.0, 1.0],
            "align_type": 1,
            "incline_mode": 1,
            "incline_val": rise,
        },
    })
    return params


def _run(params):
    raw, nodes, props, err = solver.run_raw_analysis(params)
    assert err == 0
    assert nodes is not None
    return raw, nodes, props


def _assert_all_result_arrays_finite(span, keys):
    for key in keys:
        assert np.all(np.isfinite(span[key]))


def test_inclined_udl_develops_axial_force_and_stays_finite():
    params = _beam_params(load_q=10.0, rise=2.0)

    raw, _, _ = _run(params)
    results = solver.combine_results(raw, params, "Characteristic (No Dynamic Factor)")
    span = results["Dead Load"]["S1"]

    assert np.max(np.abs(span["N"])) > 1e-6
    assert np.max(np.abs(span["V"])) > 1e-6
    assert np.max(np.abs(span["M"])) > 1e-6
    _assert_all_result_arrays_finite(span, ["M", "V", "N", "def_x", "def_y"])


def test_inclined_udl_global_vertical_reaction_equilibrium():
    horizontal_length = 10.0
    rise = 2.0
    q = 10.0
    params = _beam_params(load_q=q, rise=rise)

    raw, _, _ = _run(params)

    expected_total_vertical = q * math.sqrt(horizontal_length**2 + rise**2)
    ry_left = raw["Reactions"][200][1]
    ry_right = raw["Reactions"][201][1]

    assert ry_left + ry_right == pytest.approx(expected_total_vertical, rel=1e-3, abs=1e-3)


def test_inclined_point_vehicle_develops_axial_force_and_stays_finite():
    params = _beam_params(
        load_q=0.0,
        rise=2.0,
        vehicle_loads={"loads": [100.0 / 9.81], "spacing": [0.0]},
        mesh_size=0.5,
        step_size=0.5,
    )

    raw, _, _ = _run(params)
    results = solver.combine_results(raw, params, "Characteristic (No Dynamic Factor)")
    span = results["Vehicle Envelope"]["S1"]

    assert max(np.max(np.abs(span["N_max"])), np.max(np.abs(span["N_min"]))) > 1e-6
    _assert_all_result_arrays_finite(
        span,
        ["M_max", "M_min", "V_max", "V_min", "N_max", "N_min", "def_x_max", "def_x_min", "def_y_max", "def_y_min"],
    )


def test_horizontal_selfweight_has_no_gravity_axial_component():
    params = _beam_params(load_q=10.0, rise=0.0)

    raw, _, _ = _run(params)
    results = solver.combine_results(raw, params, "Characteristic (No Dynamic Factor)")
    span = results["Dead Load"]["S1"]

    assert np.max(np.abs(span["N"])) == pytest.approx(0.0, abs=1e-6)
