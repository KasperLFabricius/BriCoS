import ast
import inspect
import json
import os
import textwrap

import numpy as np
import pandas as pd

import bricos_data as data
import bricos_report
import bricos_solver as solver


def test_load_data_from_df_reports_skipped_rows():
    st = data.st
    st.session_state.clear()
    df = pd.DataFrame([
        {"System": "sysA", "Parameter": "num_spans", "Value": "2"},
        {"System": "sysA", "Parameter": "L_list", "Value": "{not valid json"},
        {"System": "Global", "Parameter": "result_mode", "Value": json.dumps("Design (ULS)")},
    ])

    loaded, skipped = data.load_data_from_df(df)

    assert loaded is True
    assert skipped == ["sysA/L_list"]
    assert st.session_state["sysA"]["num_spans"] == 2
    assert st.session_state["result_mode"] == "Design (ULS)"


def test_load_data_from_df_rejects_missing_columns():
    loaded, skipped = data.load_data_from_df(pd.DataFrame([{"Foo": 1}]))

    assert loaded is False
    assert skipped == []


def test_get_writable_path_frozen_uses_appdata(monkeypatch, tmp_path):
    monkeypatch.setattr(data.sys, "frozen", True, raising=False)
    monkeypatch.setattr(data.sys, "executable", str(tmp_path / "app" / "BriCoS.exe"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    path = data.get_writable_path("latest_session.csv")

    expected_dir = os.path.join(str(tmp_path), "Roaming", "BriCoS")
    assert path == os.path.join(expected_dir, "latest_session.csv")
    assert os.path.isdir(expected_dir)

    legacy = data.get_legacy_writable_path("latest_session.csv")
    assert legacy == os.path.join(str(tmp_path), "app", "latest_session.csv")


def test_get_writable_path_dev_uses_module_directory():
    module_dir = os.path.dirname(os.path.abspath(data.__file__))

    assert data.get_writable_path("x.csv") == os.path.join(module_dir, "x.csv")
    assert data.get_legacy_writable_path("x.csv") is None


def test_ai_manifest_is_utf8_json():
    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(data.__file__)), ".bricos_ai_manifest.json"
    )
    with open(manifest_path, "rb") as f:
        obj = json.loads(f.read().decode("utf-8"))

    assert obj.get("project") == "BriCoS"


def test_app_version_is_centralized_from_data_module():
    assert isinstance(data.APP_VERSION, str)
    assert data.APP_VERSION
    assert bricos_report.data_mod.APP_VERSION == data.APP_VERSION

    parts = data.APP_VERSION.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts)


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


def test_solver_cache_params_strips_cosmetic_and_combination_keys():
    params = data.get_def()
    params["name"] = "Renamed System"
    params["scale_manual"] = 4.2
    params["_vehicle_text_errors"] = {"vehicle": ["bad input"]}

    filtered = solver.solver_cache_params(params)

    for key in solver.NON_SOLVER_PARAM_KEYS:
        assert key not in filtered

    # Solver-relevant keys must survive untouched.
    assert filtered["mesh_size"] == params["mesh_size"]
    assert filtered["step_size"] == params["step_size"]
    assert filtered["L_list"] == params["L_list"]
    assert filtered["vehicle"] == params["vehicle"]
    assert filtered["phi_mode"] == params["phi_mode"]
    assert filtered["mode"] == params["mode"]


def test_combine_results_missing_envelope_b_element_contributes_zero():
    x = np.array([0.0, 1.0])
    env_a = np.array([10.0, 20.0])
    base = {
        "x": x,
        "M": np.zeros(2), "V": np.zeros(2), "N": np.zeros(2),
        "def_x": np.zeros(2), "def_y": np.zeros(2),
        "loads": [], "L": 1.0, "cx": 1.0, "cy": 0.0,
        "ni": (0.0, 0.0), "nj": (1.0, 0.0), "ni_id": 200, "nj_id": 201,
    }
    env_keys = (
        "M_max", "M_min", "V_max", "V_min", "N_max", "N_min",
        "def_x_max", "def_x_min", "def_y_max", "def_y_min",
    )
    raw = {
        "Selfweight": {}, "Soil": {}, "Surcharge": {},
        "Vehicle Envelope A": {"S1": {**{k: env_a for k in env_keys}, "base": base}},
        # S1 is missing from envelope B entirely; it must contribute zero,
        # not fall back to envelope A (which double-counted vehicle A).
        "Vehicle Envelope B": {},
        "phi_calc": 1.0, "phi_log": [],
    }
    params = {
        "KFI": 1.0, "gamma_g": 1.0, "gamma_j": 1.0,
        "gamma_veh": 1.0, "gamma_vehB": 10.0,
        "phi_mode": "Manual", "phi": 1.0,
        "combine_surcharge_vehicle": False,
    }

    results = solver.combine_results(raw, params, "Design (ULS)")
    veh = results["Vehicle Envelope"]["S1"]

    for key in env_keys:
        np.testing.assert_allclose(veh[key], env_a)


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
