import numpy as np
import pytest

import bricos_data as data
import bricos_results_ui as results_ui
import bricos_solver as solver


def _clear():
    solver.clear_solver_cache()


def _beam_params(**overrides):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam",
        "num_spans": 1,
        "L_list": [10.0] * 10,
        "Is_list": [0.8] * 10,
        "sw_list": [0.0] * 10,
        "E": 30e6,
        "E_span_list": [30e6] * 10,
        "mesh_size": 0.5,
        "step_size": 1.0,
        "phi_mode": "Manual",
        "phi": 1.0,
        "vehicle": {"loads": [10.0], "spacing": [0.0]},
        "vehicle_direction": "Forward",
        "udl_q": 5.0,
    })
    params.update(overrides)
    return params


def _run(params):
    _clear()
    raw, nodes, _, err = solver.run_raw_analysis(params)
    assert err == 0
    _clear()
    return raw


UDL_FIELD_KEYS = [f"{b}_udl_{s}" for b in ("M", "V", "N", "def_x", "def_y")
                  for s in ("max", "min")]


def _mid_step(raw):
    steps = raw["Vehicle Steps A"]
    return min(steps, key=lambda s: abs(s["x"] - 5.0))


def test_step_udl_fields_present_and_bounded_by_static_envelope():
    raw = _run(_beam_params(udl_mode="Moving", udl_gap=2.0))
    static_env = raw["Traffic UDL"]["S1"]

    for step in raw["Vehicle Steps A"]:
        res = step["res"]["S1"]
        for key in UDL_FIELD_KEYS:
            assert key in res
        # The windowed fields can never exceed the full-deck adverse fields.
        assert np.all(res["M_udl_max"] <= static_env["M_max"] + 1e-9)
        assert np.all(res["M_udl_min"] >= static_env["M_min"] - 1e-9)
        assert "udl_loaded_ranges" in res


def test_huge_gap_excludes_all_udl():
    raw = _run(_beam_params(udl_mode="Moving", udl_gap=100.0))
    res = _mid_step(raw)["res"]["S1"]

    np.testing.assert_allclose(res["M_udl_max"], 0.0, atol=1e-12)
    np.testing.assert_allclose(res["M_udl_min"], 0.0, atol=1e-12)
    assert res["udl_loaded_ranges"] == []


def test_footprint_and_static_modes_equal_full_adverse_every_step():
    static_env_arrays = _run(_beam_params())["Traffic UDL"]["S1"]

    for overrides in (dict(udl_mode="Moving", udl_footprint=True),
                      dict(udl_mode="Static")):
        raw = _run(_beam_params(**overrides))
        for step in raw["Vehicle Steps A"]:
            res = step["res"]["S1"]
            np.testing.assert_allclose(res["M_udl_max"], static_env_arrays["M_max"])
            np.testing.assert_allclose(res["M_udl_min"], static_env_arrays["M_min"])
            # Full deck loaded.
            assert res["udl_loaded_ranges"] == [(0.0, 10.0)]


def test_smaller_gap_gives_at_least_as_much_udl():
    res_g0 = _mid_step(_run(_beam_params(udl_mode="Moving", udl_gap=0.0)))["res"]["S1"]
    res_g2 = _mid_step(_run(_beam_params(udl_mode="Moving", udl_gap=2.0)))["res"]["S1"]

    assert np.all(res_g0["M_udl_max"] >= res_g2["M_udl_max"] - 1e-9)
    assert np.all(res_g0["M_udl_min"] <= res_g2["M_udl_min"] + 1e-9)


def test_window_ranges_exclude_vehicle_window():
    raw = _run(_beam_params(udl_mode="Moving", udl_gap=2.0))
    step = _mid_step(raw)
    axle_x = step["x"]
    ranges = step["res"]["S1"]["udl_loaded_ranges"]

    assert len(ranges) >= 1
    for (a, b) in ranges:
        assert 0.0 - 1e-9 <= a < b <= 10.0 + 1e-9
        # The window is snapped inward to segment boundaries, so the loaded
        # ranges may approach the window but never cross the axle position.
        assert b <= axle_x - 2.0 + 0.5 + 1e-9 or a >= axle_x + 2.0 - 0.5 - 1e-9


def test_entry_exit_window_includes_offdeck_axles():
    # Two-axle vehicle (4 m apart) entering/leaving a 10 m deck with gap 0:
    # axles still off the deck must bound the UDL-free window, so no UDL is
    # applied under the vehicle footprint near the abutments.
    raw = _run(_beam_params(udl_mode="Moving", udl_gap=0.0,
                            vehicle={"loads": [10.0, 10.0],
                                     "spacing": [0.0, 4.0]}))
    steps = {round(float(s["x"]), 6): s for s in raw["Vehicle Steps A"]}

    # Entering: front axle at 2.0 m, rear axle at -2.0 m (off deck). The
    # deck between the abutment and the front axle is under the vehicle.
    entry = steps[2.0]["res"]["S1"]
    assert entry["udl_loaded_ranges"] == [(2.0, 10.0)]

    # Leaving: front axle at 12.0 m (off deck), rear axle at 8.0 m.
    exiting = steps[12.0]["res"]["S1"]
    assert exiting["udl_loaded_ranges"] == [(0.0, 8.0)]


def test_no_udl_keys_when_inactive():
    raw = _run(_beam_params(udl_q=0.0))
    res = raw["Vehicle Steps A"][0]["res"]["S1"]
    for key in UDL_FIELD_KEYS:
        assert key not in res
    assert "udl_loaded_ranges" not in res


def test_cache_key_includes_footprint_and_mode():
    params = data.get_def()
    filtered = solver.solver_cache_params(params)
    assert filtered["udl_footprint"] == params["udl_footprint"]
    assert filtered["udl_mode"] == params["udl_mode"]


# --- step viewer helpers ---

def _synthetic_steps():
    def mk(mval, udl_max, udl_min):
        arr = np.array(mval, dtype=float)
        return {
            "M": arr, "V": arr * 0.5, "N": arr * 0.0,
            "def_x": arr * 0.01, "def_y": arr * -0.01,
            "M_udl_max": np.full_like(arr, udl_max),
            "M_udl_min": np.full_like(arr, udl_min),
        }
    return [
        {"x": 0.0, "res": {"S1": mk([0.0, -10.0], 2.0, -1.0)}},
        {"x": 1.0, "res": {"S1": mk([0.0, -30.0], 2.0, -1.0)}},
        {"x": 2.0, "res": {"S1": mk([5.0, -20.0], 2.0, -8.0)}},
    ]


def test_find_critical_step_vehicle_only():
    steps = _synthetic_steps()
    idx, val = results_ui.find_critical_step(
        steps, "S1", "M", "min", {}, 1.0, 1.0, results_ui.STEP_EFFECTS_VEHICLE)
    assert idx == 1 and val == pytest.approx(-30.0)

    idx, val = results_ui.find_critical_step(
        steps, "S1", "M", "max", {}, 1.0, 1.0, results_ui.STEP_EFFECTS_VEHICLE)
    assert idx == 2 and val == pytest.approx(5.0)


def test_find_critical_step_combined_changes_governing_step():
    steps = _synthetic_steps()
    # With the UDL included, step 2's large adverse-min UDL (-8) makes it
    # govern sagging: -20 - 8 = -28 vs step 1's -30 - 1 = -31 -> still step 1;
    # with factor 4 on the UDL: step 2 = -52 vs step 1 = -34 -> step 2.
    idx, _ = results_ui.find_critical_step(
        steps, "S1", "M", "min", {}, 1.0, 1.0, results_ui.STEP_EFFECTS_COMBINED)
    assert idx == 1
    idx, val = results_ui.find_critical_step(
        steps, "S1", "M", "min", {}, 1.0, 4.0, results_ui.STEP_EFFECTS_COMBINED)
    assert idx == 2 and val == pytest.approx(-52.0)


def test_find_critical_step_udl_only_and_vehicle_factor():
    steps = _synthetic_steps()
    idx, val = results_ui.find_critical_step(
        steps, "S1", "M", "min", {}, 1.0, 2.0, results_ui.STEP_EFFECTS_UDL)
    assert idx == 2 and val == pytest.approx(-16.0)

    # Per-member vehicle factor map applies to the vehicle part.
    idx, val = results_ui.find_critical_step(
        steps, "S1", "M", "min", {"S1": 2.0}, 1.0, 1.0, results_ui.STEP_EFFECTS_VEHICLE)
    assert idx == 1 and val == pytest.approx(-60.0)


def test_resolve_component_key_deformation_per_member():
    assert results_ui.resolve_component_key("S1", "def") == "def_y"
    assert results_ui.resolve_component_key("W2", "def") == "def_x"
    assert results_ui.resolve_component_key("S1", "M") == "M"


def test_step_display_loads_omits_axles_in_udl_only_view():
    loads = [
        {"type": "point", "params": [100.0, 2.5]},
        {"type": "distributed_trapezoid", "is_gravity": True,
         "params": [5.0, 5.0, 0.0, 10.0]},
    ]

    # UDL-only view: the axle arrows are not part of the displayed effects.
    shown = results_ui.step_display_loads(
        loads, 2.0, results_ui.STEP_EFFECTS_UDL)
    assert [l["type"] for l in shown] == ["distributed_trapezoid"]

    # Vehicle and combined views keep the axles, factored.
    for mode in (results_ui.STEP_EFFECTS_VEHICLE,
                 results_ui.STEP_EFFECTS_COMBINED):
        shown = results_ui.step_display_loads(loads, 2.0, mode)
        assert [l["type"] for l in shown] == ["point", "distributed_trapezoid"]
        assert shown[0]["params"][0] == pytest.approx(200.0)

    # The raw (cached) loads must never be mutated by the factoring.
    assert loads[0]["params"][0] == 100.0
    assert results_ui.step_display_loads(None, 1.0, results_ui.STEP_EFFECTS_VEHICLE) == []
