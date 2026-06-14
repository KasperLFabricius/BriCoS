import math

import numpy as np

import bricos_data as data
import bricos_solver as solver


def _beam_params(load_q=0.0, vehicle_loads=None, mesh_size=0.5, step_size=0.5):
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
    })
    return params


def _run(params):
    raw, nodes, props, err = solver.run_raw_analysis(params)
    assert err == 0
    assert nodes is not None
    return raw, nodes, props


def test_simply_supported_beam_udl_regression():
    length = 10.0
    q = 10.0
    E = 30e6
    inertia = 1.0 / 12.0

    params = _beam_params(load_q=q)
    raw, _, _ = _run(params)
    results = solver.combine_results(raw, params, "Characteristic (No Dynamic Factor)")
    span = results["Dead Load"]["S1"]

    max_moment = max(abs(np.max(span["M"])), abs(np.min(span["M"])))
    expected_moment = q * length**2 / 8.0
    assert max_moment == pytest_approx(expected_moment, abs=1e-2)

    expected_reaction = q * length / 2.0
    assert raw["Reactions"][200][1] == pytest_approx(expected_reaction, abs=1e-2)
    assert raw["Reactions"][201][1] == pytest_approx(expected_reaction, abs=1e-2)

    expected_deflection = 5.0 * q * length**4 / (384.0 * E * inertia)
    assert abs(np.min(span["def_y"])) == pytest_approx(expected_deflection, rel=1e-3, abs=1e-7)


def test_simply_supported_beam_moving_point_load_regression():
    length = 10.0
    point_load_kn = 100.0
    params = _beam_params(
        load_q=0.0,
        vehicle_loads={"loads": [point_load_kn / 9.81], "spacing": [0.0]},
        mesh_size=0.5,
        step_size=0.5,
    )

    raw, _, _ = _run(params)
    results = solver.combine_results(raw, params, "Characteristic (No Dynamic Factor)")
    span = results["Vehicle Envelope"]["S1"]

    max_moment = max(abs(np.max(span["M_max"])), abs(np.min(span["M_min"])))
    assert max_moment == pytest_approx(point_load_kn * length / 4.0, abs=1.0)

    critical_step = max(
        results["Vehicle Steps A"],
        key=lambda step: max(abs(np.max(step["res"]["S1"]["M"])), abs(np.min(step["res"]["S1"]["M"]))),
    )
    assert critical_step["x"] == pytest_approx(length / 2.0, abs=0.25)


def test_loaded_tapered_member_solver_path_stays_finite():
    params = _beam_params(load_q=10.0, mesh_size=0.5)
    params["span_geom_0"] = {
        "type": 1,
        "shape": 1,
        "vals": [1.0, 0.6, 0.6],
    }

    raw, nodes, _ = _run(params)

    assert nodes is not None
    assert "S1" in raw["Dead Load"]
    span = raw["Dead Load"]["S1"]

    for key in ["M", "V", "N", "def_x", "def_y"]:
        assert np.all(np.isfinite(span[key]))

    assert np.max(np.abs(span["def_y"])) < 1.0


def test_partial_distributed_load_split_uses_local_start_and_end_coordinates():
    sub = solver.split_distributed_trapezoid_for_sub_element(
        [10.0, 20.0, 2.0, 5.0],
        loc_start=5.0,
        loc_end=10.0,
    )

    assert sub == pytest_approx([16.0, 20.0, 0.0, 2.0])


def test_point_load_at_internal_mesh_boundary_belongs_to_one_sub_element():
    x_load = 5.0
    first = solver.point_load_belongs_to_sub_element(x_load, 0.0, 5.0, is_last=False)
    second = solver.point_load_belongs_to_sub_element(x_load, 5.0, 10.0, is_last=True)
    final_end = solver.point_load_belongs_to_sub_element(10.0, 5.0, 10.0, is_last=True)

    assert [first, second].count(True) == 1
    assert second
    assert final_end


def test_vehicle_step_load_at_mesh_boundary_is_stored_once():
    params = _beam_params(
        vehicle_loads={"loads": [100.0 / 9.81], "spacing": [0.0]},
        mesh_size=5.0,
        step_size=1.0,
    )
    raw, _, _ = _run(params)
    midpoint_step = next(step for step in raw["Vehicle Steps A"] if math.isclose(step["x"], 5.0))
    loads = midpoint_step["res"]["S1"]["loads"]

    assert len(loads) == 1
    assert loads[0]["type"] == "point"
    assert loads[0]["params"][1] == pytest_approx(5.0)


def pytest_approx(*args, **kwargs):
    import pytest

    return pytest.approx(*args, **kwargs)