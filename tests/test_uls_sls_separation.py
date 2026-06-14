import numpy as np
import pytest

import bricos_data as data
import bricos_solver as solver


def _raw_with_static_and_vehicle():
    """Synthetic raw results with one element carrying static and vehicle
    effects, mirroring the structure combine_results consumes."""
    x = np.array([0.0, 1.0])
    ones = np.array([1.0, 1.0])
    base = {
        "x": x,
        "M": ones.copy(), "V": ones.copy(), "N": ones.copy(),
        "def_x": ones.copy(), "def_y": ones.copy(),
        "loads": [], "L": 1.0, "cx": 1.0, "cy": 0.0,
        "ni": (0.0, 0.0), "nj": (1.0, 0.0), "ni_id": 200, "nj_id": 201,
    }
    env_keys = (
        "M_max", "M_min", "V_max", "V_min", "N_max", "N_min",
        "def_x_max", "def_x_min", "def_y_max", "def_y_min",
    )
    veh_env = {"S1": {**{k: ones.copy() for k in env_keys}, "base": dict(base)}}
    return {
        "Dead Load": {"S1": dict(base)},
        "Soil": {"S1": dict(base)},
        "Surcharge": {},
        "Vehicle Envelope A": veh_env,
        "Vehicle Envelope B": {},
        "Traffic UDL": {"S1": {**{k: ones.copy() for k in env_keys}, "base": dict(base)}},
        "udl_line_load": 5.0,
        "phi_calc": 1.20, "phi_log": [],
        "Phi Members": {},
    }


def _params(**overrides):
    params = {
        "KFI": 1.1,
        "gamma_g": 1.25, "gamma_j": 1.0,
        "gamma_veh": 1.4, "gamma_vehB": 1.05, "gamma_udl": 0.56,
        "sls_g": 1.0, "sls_j": 1.0,
        "sls_veh": 1.0, "sls_vehB": 0.75, "sls_udl": 0.40,
        "phi_mode": "Calculate",
        "phi_sls_mode": "Same",
        "combine_surcharge_vehicle": False,
    }
    params.update(overrides)
    return params


def test_sls_mode_uses_dedicated_factor_set():
    raw = _raw_with_static_and_vehicle()
    params = _params(sls_g=0.9, sls_j=0.8, sls_veh=0.7, sls_udl=0.40)

    res = solver.combine_results(raw, params, "Characteristic (SLS)")

    # Permanent components: SLS factors, no KFI.
    np.testing.assert_allclose(res["Dead Load"]["S1"]["M_max"], 0.9)
    np.testing.assert_allclose(res["Soil"]["S1"]["M_max"], 0.8)
    # Vehicle A: sls_veh x phi (phi treatment 'Same' -> full phi).
    np.testing.assert_allclose(res["Vehicle Envelope"]["S1"]["M_max"], 0.7 * 1.20)
    # Traffic UDL: sls_udl, never phi.
    np.testing.assert_allclose(res["Traffic UDL"]["S1"]["M_max"], 0.40)


def test_sls_vehicle_b_factor_075():
    raw = _raw_with_static_and_vehicle()
    raw["Vehicle Envelope B"] = {
        "S1": {**{k: np.array([1.0, 1.0]) for k in (
            "M_max", "M_min", "V_max", "V_min", "N_max", "N_min",
            "def_x_max", "def_x_min", "def_y_max", "def_y_min")},
            "base": dict(raw["Vehicle Envelope A"]["S1"]["base"])},
    }
    params = _params(sls_veh=1.0, sls_vehB=0.75)

    res = solver.combine_results(raw, params, "Characteristic (SLS)")

    # A at 1.00 x phi plus B at 0.75 x phi (Fig. B3.2).
    np.testing.assert_allclose(
        res["Vehicle Envelope"]["S1"]["M_max"], (1.0 + 0.75) * 1.20)


def test_uls_mode_unaffected_by_sls_factors():
    raw = _raw_with_static_and_vehicle()
    params = _params(sls_g=0.0, sls_veh=0.0, sls_udl=0.0)

    res = solver.combine_results(raw, params, "Design (ULS)")

    np.testing.assert_allclose(res["Dead Load"]["S1"]["M_max"], 1.1 * 1.25)
    np.testing.assert_allclose(res["Vehicle Envelope"]["S1"]["M_max"], 1.1 * 1.4 * 1.20)
    np.testing.assert_allclose(res["Traffic UDL"]["S1"]["M_max"], 1.1 * 0.56)


def test_unfactored_mode_all_ones_no_phi():
    raw = _raw_with_static_and_vehicle()
    params = _params()

    for mode_name in ("Unfactored", "Characteristic (No Dynamic Factor)"):
        res = solver.combine_results(raw, params, mode_name)
        np.testing.assert_allclose(res["Dead Load"]["S1"]["M_max"], 1.0)
        np.testing.assert_allclose(res["Soil"]["S1"]["M_max"], 1.0)
        # No phi: vehicle factor exactly 1.0 despite phi_calc = 1.20.
        np.testing.assert_allclose(res["Vehicle Envelope"]["S1"]["M_max"], 1.0)
        np.testing.assert_allclose(res["Traffic UDL"]["S1"]["M_max"], 1.0)


def test_sanitize_migrates_udl_sls_factor():
    params = data.get_def()
    params["udl_sls_factor"] = 0.30
    sanitized = data.sanitize_input_data(params)

    assert sanitized["sls_udl"] == pytest.approx(0.30)
    assert "udl_sls_factor" not in sanitized


def test_defaults_follow_vejledning_b32():
    params = data.get_def()
    assert params["sls_g"] == 1.0
    assert params["sls_j"] == 1.0
    assert params["sls_veh"] == 1.0
    assert params["sls_vehB"] == 0.75
    assert params["sls_udl"] == 0.40
    assert params["analyze_uls"] is True
    assert params["analyze_sls"] is True


def test_cache_key_excludes_sls_set_and_toggles():
    params = data.get_def()
    filtered = solver.solver_cache_params(params)

    for key in ("sls_g", "sls_j", "sls_veh", "sls_vehB", "sls_udl",
                "analyze_uls", "analyze_sls"):
        assert key not in filtered
