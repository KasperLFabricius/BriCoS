import ast
import inspect
import textwrap

import numpy as np

import bricos_data as data
import bricos_report
import bricos_solver as solver


def test_app_version_is_centralized_from_data_module():
    assert data.APP_VERSION == "0.40"
    assert bricos_report.data_mod.APP_VERSION == data.APP_VERSION


def test_get_clear_initializes_empty_vehicle_and_surcharge_state():
    params = data.get_clear("A", "Beam")

    assert params["vehicle"] == {"loads": [], "spacing": []}
    assert params["vehicleB"] == {"loads": [], "spacing": []}
    assert params["combine_surcharge_vehicle"] is False
    assert params["use_shear_def"] is False


def test_sanitize_input_data_preserves_geometry_defaults_and_material_lengths():
    params = data.get_clear("A", "Beam")
    params["span_geom_0"] = {"type": 1, "shape": 0, "vals": []}
    params["Is_list"][0] = 1.25
    params["fck_span_list"] = []
    params["E_span_list"] = []
    params["fck_wall_list"] = []
    params["E_wall_list"] = []

    sanitized = data.sanitize_input_data(params)

    assert sanitized["span_geom_0"]["vals"] == [1.25, 1.25, 1.25]
    assert len(sanitized["fck_span_list"]) == 10
    assert len(sanitized["E_span_list"]) == 10
    assert len(sanitized["fck_wall_list"]) == 11
    assert len(sanitized["E_wall_list"]) == 11


def test_combine_results_scales_all_deformation_keys_for_static_components():
    x = np.array([0.0, 1.0])
    base = {
        "S1": {
            "x": x,
            "M": np.array([1.0, 2.0]),
            "V": np.array([3.0, 4.0]),
            "N": np.array([5.0, 6.0]),
            "def_x": np.array([0.1, 0.2]),
            "def_y": np.array([-0.3, -0.4]),
            "loads": [],
            "L": 1.0,
            "cx": 1.0,
            "cy": 0.0,
            "ni": (0.0, 0.0),
            "nj": (1.0, 0.0),
            "ni_id": 200,
            "nj_id": 201,
        }
    }
    raw = {
        "Selfweight": base,
        "Soil": {},
        "Surcharge": {},
        "Vehicle Envelope A": {},
        "Vehicle Envelope B": {},
        "Vehicle Steps A": [],
        "Vehicle Steps A_Rev": [],
        "Vehicle Steps B": [],
        "Vehicle Steps B_Rev": [],
        "phi_calc": 1.0,
        "phi_log": [],
        "Reactions": {},
    }
    params = {
        "KFI": 2.0,
        "gamma_g": 3.0,
        "gamma_j": 1.0,
        "gamma_veh": 1.0,
        "gamma_vehB": 1.0,
        "phi_mode": "Manual",
        "phi": 1.0,
        "combine_surcharge_vehicle": False,
    }

    results = solver.combine_results(raw, params, "Design (ULS)")
    span = results["Selfweight"]["S1"]

    np.testing.assert_allclose(span["def_x"], base["S1"]["def_x"] * 6.0)
    np.testing.assert_allclose(span["def_y"], base["S1"]["def_y"] * 6.0)
    np.testing.assert_allclose(span["def_x_max"], base["S1"]["def_x"] * 6.0)
    np.testing.assert_allclose(span["def_x_min"], base["S1"]["def_x"] * 6.0)
    np.testing.assert_allclose(span["def_y_max"], base["S1"]["def_y"] * 6.0)
    np.testing.assert_allclose(span["def_y_min"], base["S1"]["def_y"] * 6.0)


def test_combine_results_has_no_duplicate_literal_dict_keys():
    tree = ast.parse(textwrap.dedent(inspect.getsource(solver.combine_results)))
    duplicates = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            seen = set()
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if key.value in seen:
                        duplicates.append(key.value)
                    seen.add(key.value)

    assert duplicates == []
