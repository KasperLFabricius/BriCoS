"""combine_results memo (v0.65): combining is pure in (raw identity,
factor params, result mode) and is now memoized in session state - the
UI reruns, the report chapters and the Excel-export click all reuse one
combination instead of recomputing it."""
import ast
import inspect

import numpy as np
import pytest

import bricos_data as data
import bricos_solver as solver


def _beam_params(**overrides):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam",
        "num_spans": 2,
        "L_list": [10.0] * 10,
        "Is_list": [0.8] * 10,
        "sw_list": [5.0, 5.0] + [0.0] * 8,
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
        "KFI": 1.1,
        "gamma_g": 1.0,
        "gamma_veh": 1.4,
    })
    params.update(overrides)
    return params


@pytest.fixture()
def raw_and_params():
    solver.clear_solver_cache()
    params = _beam_params()
    raw, nodes, _, err = solver.run_raw_analysis(params)
    assert err == 0
    yield raw, params
    solver.clear_solver_cache()


def test_repeat_combination_returns_the_memoized_object(raw_and_params):
    raw, params = raw_and_params
    first = solver.combine_results(raw, params, "Design (ULS)")
    again = solver.combine_results(raw, params, "Design (ULS)")
    assert again is first

    other_mode = solver.combine_results(raw, params, "Unfactored")
    assert other_mode is not first
    assert solver.combine_results(raw, params, "Unfactored") is other_mode


def test_factor_change_busts_the_memo_and_scales_correctly(raw_and_params):
    raw, params = raw_and_params
    base = solver.combine_results(raw, params, "Design (ULS)")

    changed = dict(params)
    changed["gamma_g"] = 1.35
    out = solver.combine_results(raw, changed, "Design (ULS)")
    assert out is not base
    np.testing.assert_allclose(
        out["Selfweight"]["S1"]["M_max"],
        np.asarray(base["Selfweight"]["S1"]["M_max"]) * 1.35, atol=1e-9)

    # Reverting the factor hits the original entry again.
    assert solver.combine_results(raw, params, "Design (ULS)") is base


def test_every_combine_param_key_busts_the_memo(raw_and_params):
    raw, params = raw_and_params
    probe = {
        "KFI": 0.9, "gamma_g": 1.17, "gamma_j": 1.23, "gamma_veh": 1.31,
        "gamma_vehB": 1.07, "gamma_udl": 0.77, "phi_mode": "Calculate",
        "phi": 1.41, "phi_sls_mode": "Manual", "phi_sls": 1.09,
        "sls_g": 0.93, "sls_j": 0.91, "sls_veh": 0.87, "sls_vehB": 0.83,
        "sls_udl": 0.61, "combine_surcharge_vehicle": True,
    }
    assert set(probe) == set(solver.COMBINE_PARAM_KEYS)

    base = solver.combine_results(raw, params, "Design (ULS)")
    for key, value in probe.items():
        changed = dict(params)
        changed[key] = value
        assert solver.combine_results(raw, changed, "Design (ULS)") is not base, key


def test_fresh_analysis_gets_a_new_token_and_memo_entry():
    solver.clear_solver_cache()
    params = _beam_params()
    raw1, _, _, _ = solver.run_raw_analysis(params)
    out1 = solver.combine_results(raw1, params, "Unfactored")

    # The solver cache returns the same raw (same token) for identical
    # params; a geometry change produces a fresh raw with a new token.
    raw_same, _, _, _ = solver.run_raw_analysis(params)
    assert raw_same["combine_token"] == raw1["combine_token"]
    assert solver.combine_results(raw_same, params, "Unfactored") is out1

    params2 = _beam_params(sw_list=[7.0, 7.0] + [0.0] * 8)
    raw2, _, _, _ = solver.run_raw_analysis(params2)
    assert raw2["combine_token"] != raw1["combine_token"]
    assert solver.combine_results(raw2, params2, "Unfactored") is not out1
    solver.clear_solver_cache()


def test_clear_solver_cache_drops_the_memo(raw_and_params):
    raw, params = raw_and_params
    first = solver.combine_results(raw, params, "Design (ULS)")
    solver.clear_solver_cache()
    assert solver.combine_results(raw, params, "Design (ULS)") is not first


def test_legacy_raw_without_token_bypasses_the_memo():
    raw = solver.get_safe_error_result()
    assert "combine_token" not in raw
    params = _beam_params()
    a = solver.combine_results(raw, params, "Unfactored")
    b = solver.combine_results(raw, params, "Unfactored")
    assert a is not b


def test_memo_size_stays_bounded(raw_and_params):
    raw, params = raw_and_params
    for i in range(3 * solver._COMBINE_MEMO_MAX_ENTRIES):
        changed = dict(params)
        changed["gamma_g"] = 1.0 + i * 1e-3
        solver.combine_results(raw, changed, "Design (ULS)")
    memo = solver.st.session_state[solver._COMBINE_MEMO_KEY]
    assert len(memo) <= solver._COMBINE_MEMO_MAX_ENTRIES


def test_combine_param_keys_cover_every_params_read_in_the_implementation():
    # Source-inspection guard: a future params read added to the
    # implementation must be added to COMBINE_PARAM_KEYS, or the memo
    # would serve stale results when that parameter changes.
    src = inspect.getsource(solver._combine_results_impl)
    tree = ast.parse(src)
    reads = set()
    for node in ast.walk(tree):
        # params.get('key', ...)
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "params"
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            reads.add(node.args[0].value)
        # params['key']
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "params"
                and isinstance(node.slice, ast.Constant)):
            reads.add(node.slice.value)
    assert reads == set(solver.COMBINE_PARAM_KEYS)
