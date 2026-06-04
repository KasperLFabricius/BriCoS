import numpy as np

import bricos_data as data
import bricos_kernels as kernels
import bricos_solver as solver


def _beam_params(*, span_geom=None, vehicle_loads=None, mesh_size=0.5, step_size=0.5):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam",
        "num_spans": 1,
        "L_list": [10.0] * 10,
        "Is_list": [1.0] * 10,
        "sw_list": [0.0] * 10,
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
    if span_geom is not None:
        params["span_geom_0"] = span_geom
    return params


def _run(params):
    raw, nodes, props, err = solver.run_raw_analysis(params)
    assert err == 0
    assert nodes is not None
    return raw, nodes, props


def _vehicle_envelope(params):
    raw, _, _ = _run(params)
    results = solver.combine_results(raw, params, "Characteristic (No Dynamic Factor)")
    assert "S1" in results["Vehicle Envelope"]
    return results["Vehicle Envelope"]["S1"]


def test_vehicle_transverse_point_fef_matches_analytical_for_prismatic_members():
    P = 100.0
    L = 10.0
    a = 4.0
    E = 30e6
    G = 0.0
    eff_vals = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    actual = kernels.jit_vehicle_transverse_point_fef(
        P, a, L, E, G, 0, 1, eff_vals, 1.0, 1.0,
    )
    expected = kernels.jit_fef_point(P, a, L)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_vehicle_transverse_point_fef_matches_analytical_for_tapered_members():
    P = 100.0
    L = 10.0
    a = 4.0
    E = 30e6
    G = 0.0
    eff_vals = np.array([1.0, 0.8, 0.6], dtype=np.float64)
    b_eff = 1.0
    As_avg = 0.75
    actual = kernels.jit_vehicle_transverse_point_fef(
        P, a, L, E, G, 1, 1, eff_vals, b_eff, As_avg,
    )
    expected = kernels.jit_fef_point(P, a, L)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_horizontal_tapered_span_under_moving_point_load_stays_finite():
    params = _beam_params(
        span_geom={"type": 1, "shape": 1, "vals": [1.0, 0.6, 0.6]},
        vehicle_loads={"loads": [100.0 / 9.81], "spacing": [0.0]},
        mesh_size=0.5,
        step_size=0.5,
    )

    raw, _, _ = _run(params)
    assert "S1" in raw["Vehicle Envelope A"]
    results = solver.combine_results(raw, params, "Characteristic (No Dynamic Factor)")
    span = results["Vehicle Envelope"]["S1"]

    keys = [
        "M_max", "M_min", "V_max", "V_min", "N_max", "N_min",
        "def_x_max", "def_x_min", "def_y_max", "def_y_min",
    ]
    for key in keys:
        assert np.all(np.isfinite(span[key]))

    assert max(np.max(np.abs(span["def_y_max"])), np.max(np.abs(span["def_y_min"]))) < 1.0
    assert max(np.max(np.abs(span["M_max"])), np.max(np.abs(span["M_min"]))) > 1e-6
    assert max(np.max(np.abs(span["V_max"])), np.max(np.abs(span["V_min"]))) > 1e-6


def test_constant_fake_taper_moving_load_matches_prismatic_baseline():
    vehicle = {"loads": [100.0 / 9.81], "spacing": [0.0]}
    prismatic = _beam_params(
        span_geom={"type": 1, "shape": 0, "vals": [1.0, 1.0, 1.0]},
        vehicle_loads=vehicle,
        mesh_size=0.5,
        step_size=0.5,
    )
    fake_taper = _beam_params(
        span_geom={"type": 1, "shape": 1, "vals": [1.0, 1.0, 1.0]},
        vehicle_loads=vehicle,
        mesh_size=0.5,
        step_size=0.5,
    )

    span_prismatic = _vehicle_envelope(prismatic)
    span_fake_taper = _vehicle_envelope(fake_taper)

    for key in ["M_max", "M_min", "V_max", "V_min"]:
        np.testing.assert_allclose(span_fake_taper[key], span_prismatic[key], rtol=1e-6, atol=1e-6)
