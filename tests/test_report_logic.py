import io

import bricos_data as data
from bricos_report import BricosReportGenerator


def _params(vehicle=None, vehicle_b=None, direction="Forward", combine_surcharge_vehicle=False, surcharge=None, **overrides):
    params = data.get_clear("A", "Beam")
    params.update({
        "name": "System",
        "vehicle": vehicle if vehicle is not None else {"loads": [], "spacing": []},
        "vehicleB": vehicle_b if vehicle_b is not None else {"loads": [], "spacing": []},
        "vehicle_direction": direction,
        "combine_surcharge_vehicle": combine_surcharge_vehicle,
        "surcharge": [] if surcharge is None else surcharge,
    })
    params.update(overrides)
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


def test_characteristic_formula_reports_permanent_only_without_vehicle_or_surcharge():
    formula = BricosReportGenerator._characteristic_formula_text(_params())

    assert formula == "1.0 · SW"
    assert "Other variable actions" not in formula


def test_characteristic_formula_reports_surcharge_only_without_vehicle():
    formula = BricosReportGenerator._characteristic_formula_text(
        _params(surcharge=[{"q": 10.0}], sls_veh=1.0)
    )

    assert formula == "1.0 · SW + 1.0 · Surcharge"
    assert "VehA" not in formula
    assert "Other variable actions" not in formula


def test_characteristic_formula_reports_vehicle_only_without_surcharge():
    formula = BricosReportGenerator._characteristic_formula_text(
        _params(vehicle={"loads": [10.0], "spacing": [0.0]}, sls_veh=1.0)
    )

    assert formula == "1.0 · SW + 1.0 · Phi · VehA"
    assert "Surcharge" not in formula
    assert "Other variable actions" not in formula


def test_characteristic_formula_uses_sls_factor_set():
    formula = BricosReportGenerator._characteristic_formula_text(
        _params(
            vehicle={"loads": [10.0], "spacing": [0.0]},
            vehicleB={"loads": [5.0], "spacing": [0.0]},
            sls_veh=1.0, sls_vehB=0.75, sls_g=0.9,
        )
    )

    assert formula == "0.9 · SW + 1.0 · Phi · VehA + 0.75 · Phi · VehB"


def test_characteristic_formula_exclusive_mode_reports_vehicle_surcharge_envelope():
    formula = BricosReportGenerator._characteristic_formula_text(
        _params(
            vehicle={"loads": [10.0], "spacing": [0.0]},
            surcharge=[{"q": 10.0}],
            combine_surcharge_vehicle=False,
            sls_veh=1.0,
        )
    )

    assert formula == "1.0 · SW + Envelope(1.0 · Phi · VehA, 1.0 · Surcharge)"
    assert "Other variable actions" not in formula


def test_characteristic_formula_simultaneous_mode_reports_vehicle_plus_surcharge():
    formula = BricosReportGenerator._characteristic_formula_text(
        _params(
            vehicle={"loads": [10.0], "spacing": [0.0]},
            surcharge=[{"q": 10.0}],
            combine_surcharge_vehicle=True,
            sls_veh=1.0,
        )
    )

    assert formula == "1.0 · SW + 1.0 · Phi · VehA + 1.0 · Surcharge"
    assert "Envelope" not in formula
    assert "Other variable actions" not in formula


def _eq_case(ax=0.0, ay=0.0, rx=0.0, ry=0.0, has_loads=None):
    case = {
        "applied_x": ax, "applied_y": ay,
        "reactions_x": rx, "reactions_y": ry,
        "residual_x": ax + rx, "residual_y": ay + ry,
    }
    if has_loads is not None:
        case["has_loads"] = has_loads
    return case


def test_equilibrium_rows_distinguish_cancelling_loads_from_no_loads():
    # Mirrored earth pressure on both walls cancels in the global sums; the
    # row must still PASS instead of claiming "(no loads)".
    rows = BricosReportGenerator._equilibrium_rows({
        "Selfweight": _eq_case(ay=-200.0, ry=200.0, has_loads=True),
        "Soil": _eq_case(has_loads=True),
        "Surcharge": _eq_case(has_loads=False),
    })

    by_case = {r[0]: r for r in rows[1:]}
    assert by_case["Selfweight"][-1] == "PASS"
    assert by_case["Soil"][-1] == "PASS"
    assert by_case["Surcharge"][-1] == "(no loads)"


def test_equilibrium_rows_flag_residual_failures():
    rows = BricosReportGenerator._equilibrium_rows({
        "Soil": _eq_case(ax=60.0, rx=-50.0, has_loads=True),
    })

    assert rows[1][-1] == "CHECK FAILED"


def test_equilibrium_rows_legacy_results_fall_back_to_magnitudes():
    # Raw results generated before the has_loads flag existed: all-zero
    # sums still read as "(no loads)", non-zero sums get the check.
    rows = BricosReportGenerator._equilibrium_rows({
        "Selfweight": _eq_case(ay=-100.0, ry=100.0),
        "Soil": _eq_case(),
    })

    by_case = {r[0]: r for r in rows[1:]}
    assert by_case["Selfweight"][-1] == "PASS"
    assert by_case["Soil"][-1] == "(no loads)"


def test_std_table_wraps_only_overflowing_string_cells():
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph

    gen = _generator(_params())
    long_text = ("not applied (intensity includes the dynamic increment, "
                 "DK NA A.2.3.2)")
    try:
        t = gen._make_std_table(
            [["Header", "Dynamic factor"], ["a", long_text]],
            [2 * cm, 3 * cm], font_size=8)
    finally:
        gen.executor.shutdown(wait=True)

    cells = t._cellvalues
    assert isinstance(cells[1][1], Paragraph)  # overflowing body cell wraps
    assert isinstance(cells[0][0], str)        # short cells stay plain
    assert isinstance(cells[1][0], str)


def test_udl_application_text_modes():
    static = _params(udl_mode="Static")
    assert BricosReportGenerator._udl_application_text(static, True) == (
        "static full deck at every step"
    )

    moving = _params(udl_mode="Moving", udl_gap=5.0, udl_footprint=False)
    txt = BricosReportGenerator._udl_application_text(moving, True)
    assert "5.00 m" in txt  # custom distances must be reported verbatim
    assert "conservative" in txt

    footprint = _params(udl_mode="Moving", udl_footprint=True)
    assert "within the vehicle window" in (
        BricosReportGenerator._udl_application_text(footprint, True)
    )

    no_vehicle = BricosReportGenerator._udl_application_text(moving, False)
    assert "no vehicle load model" in no_vehicle
    assert "clear distance" not in no_vehicle


def test_critical_step_titles_state_lead_axle_chainage():
    import numpy as np

    params = _params(vehicle={"loads": [10.0], "spacing": [0.0]},
                     num_spans=1)
    arr = np.array([0.0, -25.0])
    steps = [
        {"x": -4.0, "res": {"S1": {"M": arr, "V": arr * 0.5}}},
        {"x": 6.0, "res": {"S1": {"M": -arr, "V": arr}}},
    ]
    gen = _generator(params, {"Vehicle Steps A": steps})

    try:
        groups = gen._identify_critical_steps(
            params, {"Vehicle Steps A": steps}, "System A",
            {200: (0.0, 0.0)}, "Vehicle Steps A", "Vehicle A", "Forward")
    finally:
        gen.executor.shutdown(wait=True)

    titles = [p["title"] for g in groups for p in g["plots"]]
    assert titles
    # Negative chainage (vehicle entering/leaving the deck) is labeled as
    # the lead-axle position instead of a bare "X=-4.00m".
    assert any("lead axle at x = -4.00 m" in t for t in titles)
    assert all("lead axle at x =" in t for t in titles)


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