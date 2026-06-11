"""Batched UDL segment recovery must reproduce the legacy per-segment path.

The batched kernel (jit_udl_segment_recovery_batch) replaces one full
Python detailed recovery per deck sub-element; USE_BATCH_STEP_RECOVERY=False
forces the legacy path for both the vehicle steps and the UDL segments.
"""
import numpy as np
import pytest

import bricos_data as data
import bricos_solver as solver


def _run_both_paths(params, monkeypatch):
    assert solver.USE_BATCH_STEP_RECOVERY is True
    solver.clear_solver_cache()
    raw_fast, _, _, err = solver.run_raw_analysis(params)
    assert err == 0

    monkeypatch.setattr(solver, "USE_BATCH_STEP_RECOVERY", False)
    solver.clear_solver_cache()
    raw_legacy, _, _, err = solver.run_raw_analysis(params)
    assert err == 0
    solver.clear_solver_cache()
    return raw_fast, raw_legacy


ENV_KEYS = ("M_max", "M_min", "V_max", "V_min", "N_max", "N_min",
            "def_x_max", "def_x_min", "def_y_max", "def_y_min")


def _assert_udl_env_equal(raw_fast, raw_legacy):
    env_f = raw_fast["Traffic UDL"]
    env_l = raw_legacy["Traffic UDL"]
    assert set(env_f.keys()) == set(env_l.keys())
    assert env_l  # the UDL envelope must actually exist
    for pid in env_l:
        for key in ENV_KEYS:
            np.testing.assert_allclose(
                env_f[pid][key], env_l[pid][key], rtol=1e-7, atol=1e-9,
                err_msg=f"{pid}/{key} mismatch")


def _assert_step_udl_fields_equal(steps_fast, steps_legacy):
    assert len(steps_fast) == len(steps_legacy)
    assert len(steps_fast) > 0
    udl_keys = ("M_udl_max", "M_udl_min", "V_udl_max", "V_udl_min",
                "N_udl_max", "N_udl_min")
    seen_any = False
    for step_f, step_l in zip(steps_fast, steps_legacy):
        for eid, res_l in step_l["res"].items():
            res_f = step_f["res"][eid]
            for key in udl_keys:
                if key not in res_l:
                    assert key not in res_f
                    continue
                seen_any = True
                np.testing.assert_allclose(
                    res_f[key], res_l[key], rtol=1e-7, atol=1e-9,
                    err_msg=f"{eid}/{key} mismatch")
            assert res_f.get("udl_loaded_ranges") == res_l.get("udl_loaded_ranges")
    assert seen_any  # the steps must actually carry UDL coupling fields


def _beam_params(**overrides):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam",
        "num_spans": 2,
        "L_list": [10.0, 8.0] + [0.0] * 8,
        "Is_list": [0.8] * 10,
        "sw_list": [5.0, 5.0] + [0.0] * 8,
        "E": 30e6,
        "E_span_list": [30e6] * 10,
        "mesh_size": 1.0,
        "step_size": 0.5,
        "vehicle": {"loads": [10.0, 12.0], "spacing": [0.0, 2.0]},
        "vehicle_direction": "Forward",
        "phi_mode": "Manual",
        "phi": 1.0,
        "udl_q": 7.5,
        "udl_gap": 2.0,
        "udl_mode": "Moving",
    })
    params.update(overrides)
    return params


def _frame_params_inclined_tapered(**overrides):
    params = data.get_clear("A", "Frame")
    params.update({
        "mode": "Frame",
        "num_spans": 2,
        "L_list": [10.0, 10.0] + [0.0] * 8,
        "Is_list": [0.8] * 10,
        "sw_list": [5.0, 5.0] + [0.0] * 8,
        "h_list": [6.0, 6.0, 6.0] + [0.0] * 8,
        "Iw_list": [0.7] * 11,
        "E": 30e6,
        "E_span_list": [30e6] * 10,
        "E_wall_list": [30e6] * 11,
        "mesh_size": 1.0,
        "step_size": 1.0,
        "vehicle": {"loads": [15.0, 15.0], "spacing": [0.0, 2.0]},
        "vehicle_direction": "Both",
        "phi_mode": "Manual",
        "phi": 1.0,
        "udl_q": 5.0,
        "udl_gap": 0.0,
        "udl_mode": "Moving",
        # Inclined span: the gravity UDL develops both transverse and
        # axial local components (cx/cy decomposition in the kernel).
        "span_geom_0": {
            "type": 1, "shape": 0, "vals": [0.8, 0.8, 0.8],
            "align_type": 1, "incline_mode": 0, "incline_val": 5.0,
        },
        # Tapered span: non-prismatic k_local enters the S-matrix
        # recovery; the trapezoid FEF itself is EI-independent.
        "span_geom_1": {
            "type": 1, "shape": 1, "vals": [0.6, 0.7, 0.9],
            "align_type": 0, "incline_mode": 0, "incline_val": 0.0,
        },
    })
    params.update(overrides)
    return params


def test_batched_udl_matches_legacy_two_span_beam(monkeypatch):
    raw_fast, raw_legacy = _run_both_paths(_beam_params(), monkeypatch)

    _assert_udl_env_equal(raw_fast, raw_legacy)
    _assert_step_udl_fields_equal(
        raw_fast["Vehicle Steps A"], raw_legacy["Vehicle Steps A"])


def test_batched_udl_matches_legacy_inclined_tapered_frame(monkeypatch):
    raw_fast, raw_legacy = _run_both_paths(
        _frame_params_inclined_tapered(), monkeypatch)

    _assert_udl_env_equal(raw_fast, raw_legacy)
    _assert_step_udl_fields_equal(
        raw_fast["Vehicle Steps A"], raw_legacy["Vehicle Steps A"])
    _assert_step_udl_fields_equal(
        raw_fast["Vehicle Steps A_Rev"], raw_legacy["Vehicle Steps A_Rev"])


def test_batched_udl_matches_legacy_without_vehicle(monkeypatch):
    # UDL alone (static envelope only, no steps): the envelope must still
    # come out identical through the batched path.
    params = _beam_params(vehicle={"loads": [], "spacing": []})
    raw_fast, raw_legacy = _run_both_paths(params, monkeypatch)

    _assert_udl_env_equal(raw_fast, raw_legacy)


def test_batched_udl_with_shear_deformation(monkeypatch):
    # Timoshenko stiffness changes k_local (and so the S-matrix recovery);
    # the batched path must follow it.
    raw_fast, raw_legacy = _run_both_paths(
        _beam_params(use_shear_def=True, nu=0.2), monkeypatch)

    _assert_udl_env_equal(raw_fast, raw_legacy)
