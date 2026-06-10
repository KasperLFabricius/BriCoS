import numpy as np
import pytest

import bricos_data as data
import bricos_solver as solver


def _clear_solver_cache():
    fn = getattr(solver, "_run_raw_analysis_cached", None)
    if fn is not None and hasattr(fn, "clear"):
        fn.clear()


def _run_both_paths(params, monkeypatch):
    """Run the analysis with the batched kernel and the legacy Python path."""
    assert solver.USE_BATCH_STEP_RECOVERY is True
    _clear_solver_cache()
    raw_fast, _, _, err = solver.run_raw_analysis(params)
    assert err == 0

    monkeypatch.setattr(solver, "USE_BATCH_STEP_RECOVERY", False)
    _clear_solver_cache()
    raw_legacy, _, _, err = solver.run_raw_analysis(params)
    assert err == 0
    _clear_solver_cache()
    return raw_fast, raw_legacy


def _assert_steps_equal(steps_fast, steps_legacy):
    assert len(steps_fast) == len(steps_legacy)
    assert len(steps_fast) > 0
    for step_f, step_l in zip(steps_fast, steps_legacy):
        assert step_f["x"] == pytest.approx(step_l["x"])
        assert set(step_f["res"].keys()) == set(step_l["res"].keys())
        for eid, res_l in step_l["res"].items():
            res_f = step_f["res"][eid]
            np.testing.assert_allclose(res_f["x"], res_l["x"], atol=1e-12)
            for key in ("M", "V", "N", "def_x", "def_y"):
                np.testing.assert_allclose(
                    res_f[key], res_l[key], rtol=1e-7, atol=1e-7,
                    err_msg=f"{eid}/{key} mismatch",
                )
            assert res_f["L"] == pytest.approx(res_l["L"])
            assert res_f["cx"] == pytest.approx(res_l["cx"])
            assert res_f["cy"] == pytest.approx(res_l["cy"])
            assert res_f["ni_id"] == res_l["ni_id"]
            assert res_f["nj_id"] == res_l["nj_id"]

            loads_f, loads_l = res_f["loads"], res_l["loads"]
            assert len(loads_f) == len(loads_l)
            for lf, ll in zip(loads_f, loads_l):
                assert lf["type"] == ll["type"]
                assert lf["is_gravity"] == ll["is_gravity"]
                assert lf["params"][0] == pytest.approx(ll["params"][0])
                assert lf["params"][1] == pytest.approx(ll["params"][1])


def _beam_params():
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
        "vehicle": {"loads": [10.0, 12.0, 8.0], "spacing": [0.0, 1.5, 3.0]},
        "vehicle_direction": "Forward",
        "phi_mode": "Manual",
        "phi": 1.0,
    })
    return params


def _frame_params_with_inclined_span():
    params = data.get_clear("A", "Frame")
    params.update({
        "mode": "Frame",
        "num_spans": 1,
        "L_list": [12.0] + [0.0] * 9,
        "Is_list": [0.8] * 10,
        "sw_list": [5.0] + [0.0] * 9,
        "h_list": [6.0, 6.0] + [0.0] * 9,
        "Iw_list": [0.7] * 11,
        "E": 30e6,
        "E_span_list": [30e6] * 10,
        "E_wall_list": [30e6] * 11,
        "mesh_size": 1.0,
        "step_size": 0.5,
        "vehicle": {"loads": [15.0, 15.0], "spacing": [0.0, 2.0]},
        "vehicle_direction": "Both",
        "phi_mode": "Manual",
        "phi": 1.0,
        # Inclined span: gravity axles develop both transverse and axial
        # components, exercising the cx/cy decomposition in the kernel.
        "span_geom_0": {
            "type": 1, "shape": 0, "vals": [0.8, 0.8, 0.8],
            "align_type": 1, "incline_mode": 0, "incline_val": 4.0,
        },
    })
    return params


def test_batched_step_recovery_matches_legacy_two_span_beam(monkeypatch):
    raw_fast, raw_legacy = _run_both_paths(_beam_params(), monkeypatch)

    _assert_steps_equal(raw_fast["Vehicle Steps A"], raw_legacy["Vehicle Steps A"])


def test_batched_step_recovery_matches_legacy_inclined_frame_both_directions(monkeypatch):
    raw_fast, raw_legacy = _run_both_paths(_frame_params_with_inclined_span(), monkeypatch)

    _assert_steps_equal(raw_fast["Vehicle Steps A"], raw_legacy["Vehicle Steps A"])
    _assert_steps_equal(raw_fast["Vehicle Steps A_Rev"], raw_legacy["Vehicle Steps A_Rev"])


def test_batched_step_recovery_envelopes_unchanged(monkeypatch):
    # The envelope kernel path is independent of the step-recovery path;
    # both runs must produce identical envelopes.
    raw_fast, raw_legacy = _run_both_paths(_beam_params(), monkeypatch)

    env_f = raw_fast["Vehicle Envelope A"]
    env_l = raw_legacy["Vehicle Envelope A"]
    assert set(env_f.keys()) == set(env_l.keys())
    for eid in env_l:
        for key in ("M_max", "M_min", "V_max", "V_min", "N_max", "N_min"):
            np.testing.assert_allclose(env_f[eid][key], env_l[eid][key], rtol=1e-9, atol=1e-9)
