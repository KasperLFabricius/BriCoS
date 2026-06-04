import io

import bricos_data as data
from bricos_report import BricosReportGenerator


def _params(vehicle=None, vehicle_b=None, direction="Forward", combine_surcharge_vehicle=False):
    params = data.get_clear("A", "Beam")
    params.update({
        "name": "System",
        "vehicle": vehicle if vehicle is not None else {"loads": [], "spacing": []},
        "vehicleB": vehicle_b if vehicle_b is not None else {"loads": [], "spacing": []},
        "vehicle_direction": direction,
        "combine_surcharge_vehicle": combine_surcharge_vehicle,
    })
    return params


def _generator(params_a, raw_a=None, params_b=None, raw_b=None, nodes_b=None):
    state = {
        "sysA": params_a,
        "sysB": params_b or _params(),
        "model_props_A": {"Spans": {}, "Walls": {}},
        "model_props_B": {"Spans": {}, "Walls": {}},
    }
    return BricosReportGenerator(
        io.BytesIO(),
        {},
        state,
        raw_a or {},
        raw_b or {},
        {200: (0.0, 0.0), 201: (10.0, 0.0)},
        nodes_b,
    )


def test_surcharge_interaction_wording_uses_actual_combination_flag():
    exclusive = _params(combine_surcharge_vehicle=False)
    simultaneous = _params(combine_surcharge_vehicle=True)

    assert BricosReportGenerator._surcharge_interaction_text(exclusive) == (
        "Exclusive: envelope of vehicle traffic or surcharge"
    )
    assert BricosReportGenerator._surcharge_interaction_text(simultaneous) == (
        "Simultaneous: vehicle traffic + surcharge"
    )


def test_characteristic_formula_includes_phi_and_does_not_label_no_phi_as_sls():
    formula = BricosReportGenerator._characteristic_formula_text(_params())

    assert "Phi" in formula
    assert "No Phi" not in formula


def test_characteristic_formula_exclusive_mode_reports_vehicle_surcharge_envelope():
    formula = BricosReportGenerator._characteristic_formula_text(
        _params(combine_surcharge_vehicle=False)
    )

    assert "Phi" in formula
    assert "Vehicle traffic" in formula
    assert "Envelope" in formula
    assert "Vehicle traffic + 1.0 · Surcharge" not in formula
    assert "No Phi" not in formula


def test_characteristic_formula_simultaneous_mode_reports_vehicle_plus_surcharge():
    formula = BricosReportGenerator._characteristic_formula_text(
        _params(combine_surcharge_vehicle=True)
    )

    assert "Phi" in formula
    assert "Vehicle" in formula
    assert "1.0 · Phi · Vehicle traffic + 1.0 · Surcharge" in formula
    assert "Envelope" not in formula
    assert "No Phi" not in formula


def test_has_any_vehicle_false_when_all_vehicle_lists_are_empty():
    gen = _generator(_params())

    try:
        assert not gen._has_any_vehicle()
    finally:
        gen.executor.shutdown(wait=True)


def test_direction_aware_vehicle_step_sources_for_both_directions():
    params = _params(
        vehicle={"loads": [10.0], "spacing": [0.0]},
        direction="Both",
    )
    raw = {
        "Vehicle Steps A": [{"x": 1.0, "res": {}}],
        "Vehicle Steps A_Rev": [{"x": 9.0, "res": {}}],
    }
    gen = _generator(params, raw)

    try:
        sources = list(gen._iter_vehicle_step_sources())
    finally:
        gen.executor.shutdown(wait=True)

    assert [(src[4], src[6]) for src in sources] == [
        ("Vehicle Steps A", "Forward"),
        ("Vehicle Steps A_Rev", "Reverse"),
    ]


def test_reverse_direction_reports_only_reverse_vehicle_steps():
    params = _params(
        vehicle={"loads": [10.0], "spacing": [0.0]},
        direction="Reverse",
    )
    raw = {
        "Vehicle Steps A": [{"x": 1.0, "res": {}}],
        "Vehicle Steps A_Rev": [{"x": 9.0, "res": {}}],
    }
    gen = _generator(params, raw)

    try:
        sources = list(gen._iter_vehicle_step_sources())
    finally:
        gen.executor.shutdown(wait=True)

    assert [(src[4], src[6]) for src in sources] == [("Vehicle Steps A_Rev", "Reverse")]