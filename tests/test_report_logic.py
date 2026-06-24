import io
import sys
import types

import pytest

import bricos_data as data
import bricos_report
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

    assert formula == "1.0 · SW + 1.0 · Φ · VehA"
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

    assert formula == "0.9 · SW + 1.0 · Φ · VehA + 0.75 · Φ · VehB"


def test_characteristic_formula_exclusive_mode_reports_vehicle_surcharge_envelope():
    formula = BricosReportGenerator._characteristic_formula_text(
        _params(
            vehicle={"loads": [10.0], "spacing": [0.0]},
            surcharge=[{"q": 10.0}],
            combine_surcharge_vehicle=False,
            sls_veh=1.0,
        )
    )

    assert formula == "1.0 · SW + Envelope(1.0 · Φ · VehA, 1.0 · Surcharge)"
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

    assert formula == "1.0 · SW + 1.0 · Φ · VehA + 1.0 · Surcharge"
    assert "Envelope" not in formula
    assert "Other variable actions" not in formula


def _eq_case(ax=0.0, ay=0.0, rx=0.0, ry=0.0, has_loads=None, parts=None):
    case = {
        "applied_x": ax, "applied_y": ay,
        "reactions_x": rx, "reactions_y": ry,
        "residual_x": ax + rx, "residual_y": ay + ry,
    }
    if has_loads is not None:
        case["has_loads"] = has_loads
    if parts is not None:
        # (x_pos, x_neg, y_pos, y_neg) per the v0.60 solver keys.
        case["applied_x_pos"], case["applied_x_neg"] = parts[0], parts[1]
        case["applied_y_pos"], case["applied_y_neg"] = parts[2], parts[3]
    return case


def test_equilibrium_rows_drop_cases_without_loads():
    # Cases without any load definition are no longer listed at all;
    # cancelling loads (zero sums, has_loads=True) still get a PASS row.
    rows = BricosReportGenerator._equilibrium_rows({
        "Dead Load": _eq_case(ay=-200.0, ry=200.0, has_loads=True),
        "Soil": _eq_case(has_loads=True),
        "Surcharge": _eq_case(has_loads=False),
    })

    by_case = {r[0]: r for r in rows[1:]}
    assert by_case["Dead Load"][-1] == "PASS"
    assert by_case["Soil"][-1] == "PASS"
    assert "Surcharge" not in by_case


def test_equilibrium_rows_show_cancelling_parts_when_net_is_zero():
    # Mirrored earth pressure on both walls cancels the net x-sum; the
    # applied cell must surface the opposing parts so the soil application
    # stays visible in the report.
    rows = BricosReportGenerator._equilibrium_rows({
        "Soil": _eq_case(has_loads=True, parts=(60.0, -60.0, 0.0, 0.0)),
    })

    applied = rows[1][1]
    assert "0.00 / 0.00" in applied
    assert "+60.00" in applied and "-60.00" in applied
    assert "cancel" in applied

    # A non-cancelling case must NOT carry the breakdown noise.
    rows = BricosReportGenerator._equilibrium_rows({
        "Soil": _eq_case(ax=60.0, rx=-60.0, has_loads=True,
                         parts=(60.0, 0.0, 0.0, 0.0)),
    })
    assert "cancel" not in rows[1][1]


def test_equilibrium_rows_flag_residual_failures():
    rows = BricosReportGenerator._equilibrium_rows({
        "Soil": _eq_case(ax=60.0, rx=-50.0, has_loads=True),
    })

    assert rows[1][-1] == "CHECK FAILED"


def test_equilibrium_rows_legacy_results_fall_back_to_magnitudes():
    # Raw results generated before the has_loads flag existed: all-zero
    # sums are treated as unloaded (dropped), non-zero sums get the check
    # and tolerate the missing pos/neg keys.
    rows = BricosReportGenerator._equilibrium_rows({
        "Dead Load": _eq_case(ay=-100.0, ry=100.0),
        "Soil": _eq_case(),
    })

    by_case = {r[0]: r for r in rows[1:]}
    assert by_case["Dead Load"][-1] == "PASS"
    assert "Soil" not in by_case


def test_std_table_renders_all_string_cells_as_aligned_paragraphs():
    # v0.60: every string cell becomes a Paragraph (consistent alignment,
    # wrapping, and non-WinAnsi glyphs like Φ in any cell). First column
    # left, all other columns centered, header and body alike.
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph

    gen = _generator(_params())
    try:
        t = gen._make_std_table(
            [["Header", "Dynamic factor"], ["a", "Φ applied"]],
            [2 * cm, 3 * cm], font_size=8)
    finally:
        gen.executor.shutdown(wait=True)

    cells = t._cellvalues
    assert all(isinstance(c, Paragraph) for row in cells for c in row)
    assert cells[0][0].style.alignment == TA_LEFT
    assert cells[1][0].style.alignment == TA_LEFT
    assert cells[0][1].style.alignment == TA_CENTER
    assert cells[1][1].style.alignment == TA_CENTER
    assert cells[0][0].style.fontName == "Helvetica-Bold"  # header row
    assert cells[1][1].style.fontName == "Helvetica"


def test_std_table_tolerates_plain_text_alongside_markup():
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph

    gen = _generator(_params())
    try:
        t = gen._make_std_table(
            [["Header", "Value"],
             ["Forwards & Backwards", "q<sub>top</sub>"]],
            [3 * cm, 3 * cm], font_size=8)
    finally:
        gen.executor.shutdown(wait=True)

    cells = t._cellvalues
    # Bare '&' must not break the cell (escaped if the parse rejects it);
    # inline markup like subscripts parses as markup.
    assert isinstance(cells[1][0], Paragraph)
    assert "Forwards" in cells[1][0].text
    assert isinstance(cells[1][1], Paragraph)


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


def _minimal_raw():
    return {
        "Dead Load": {}, "Soil": {}, "Surcharge": {},
        "Vehicle Envelope A": {}, "Vehicle Envelope B": {},
        "phi_calc": 1.0, "phi_log": [],
    }


def test_combined_results_are_memoized_per_system_and_mode():
    gen = _generator(_params(), raw_a=_minimal_raw())

    try:
        first = gen._combined("A", "Unfactored")
        again = gen._combined("A", "Unfactored")
        other_mode = gen._combined("A", "Design (ULS)")
    finally:
        gen.executor.shutdown(wait=True)

    assert first is again            # component chapters reuse one combination
    assert other_mode is not first   # but modes stay distinct


def test_persistent_image_export_starts_and_stops_sync_server(monkeypatch):
    calls = []
    stub = types.SimpleNamespace(
        start_sync_server=lambda **kw: calls.append(("start", kw)),
        stop_sync_server=lambda **kw: calls.append(("stop", kw)),
    )
    monkeypatch.setitem(sys.modules, "kaleido", stub)
    monkeypatch.setattr(bricos_report, "_chrome_available", lambda: True)

    with bricos_report.persistent_image_export():
        assert calls == [("start", {"silence_warnings": True})]
    assert calls[-1] == ("stop", {"silence_warnings": True})


def test_persistent_image_export_tolerates_kaleido_without_sync_server(monkeypatch):
    # kaleido 0.2.x has no sync server; the context must degrade to a no-op.
    monkeypatch.setitem(sys.modules, "kaleido", types.SimpleNamespace())

    entered = False
    with bricos_report.persistent_image_export():
        entered = True
    assert entered


def test_persistent_image_export_skips_server_without_chrome(monkeypatch):
    # start_sync_server returns before the browser launches in its thread;
    # with no Chrome that thread dies and every export hangs forever on the
    # unserviced queue. The server must not be started at all then.
    calls = []
    stub = types.SimpleNamespace(
        start_sync_server=lambda **kw: calls.append("start"),
        stop_sync_server=lambda **kw: calls.append("stop"),
    )
    monkeypatch.setitem(sys.modules, "kaleido", stub)
    monkeypatch.setattr(bricos_report, "_chrome_available", lambda: False)

    with bricos_report.persistent_image_export():
        pass
    assert calls == []


def test_chrome_available_uses_browser_lookup_and_env_override(monkeypatch):
    from choreographer.browsers.chromium import Chromium

    monkeypatch.delenv("BROWSER_PATH", raising=False)
    monkeypatch.setattr(Chromium, "find_browser",
                        classmethod(lambda cls, **kw: None))
    assert bricos_report._chrome_available() is False

    monkeypatch.setattr(Chromium, "find_browser",
                        classmethod(lambda cls, **kw: r"C:\chrome.exe"))
    assert bricos_report._chrome_available() is True

    # Explicit user override wins without consulting the lookup.
    monkeypatch.setattr(Chromium, "find_browser",
                        classmethod(lambda cls, **kw: None))
    monkeypatch.setenv("BROWSER_PATH", r"C:\custom\chrome.exe")
    assert bricos_report._chrome_available() is True


def test_uls_formula_shows_no_kfi_soil_without_raw_value():
    params = _params(
        KFI=1.1,
        gamma_g=1.0,
        gamma_j=1.0 / 1.1,
        soil=[{"wall_idx": 0, "face": "L", "h": 6.0, "q_top": 0.0, "q_bot": 20.0}],
    )
    params["soil"] = [{"wall_idx": 0, "face": "L", "h": 6.0, "q_top": 0.0, "q_bot": 20.0}]
    gen = _generator(params)

    try:
        eq = gen._build_uls_equation_text()
    finally:
        gen.executor.shutdown(wait=True)

    assert "1.0·Soil (KFI negated)" in eq
    assert "0.90" not in eq  # the raw 1/KFI value must never leak


def test_uls_formula_keeps_plain_soil_factor():
    params = _params(KFI=1.1, gamma_j=1.0)
    params["soil"] = [{"wall_idx": 0, "face": "L", "h": 6.0, "q_top": 0.0, "q_bot": 20.0}]
    gen = _generator(params)

    try:
        eq = gen._build_uls_equation_text()
    finally:
        gen.executor.shutdown(wait=True)

    assert "1.1·1.0·Soil" in eq
    assert "KFI negated" not in eq


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
            params, "System A", {200: (0.0, 0.0)},
            [("Forward", steps)], "Vehicle A")
    finally:
        gen.executor.shutdown(wait=True)

    titles = [p["title"] for g in groups for p in g["plots"]]
    assert titles
    # Negative chainage (vehicle entering/leaving the deck) is labeled as
    # the lead-axle position instead of a bare "X=-4.00m".
    assert any("lead axle at x = -4.00 m" in t for t in titles)
    assert all("lead axle at x =" in t for t in titles)
    # Single analyzed direction: titles stay free of direction noise.
    assert all("Forward" not in t for t in titles)


def test_critical_steps_combine_labels_sharing_one_governing_step():
    import numpy as np

    # Min M and Max M govern at the SAME step: one plot with the combined
    # label instead of silently dropping the second extreme (the plot for
    # e.g. "Max M" used to vanish without a trace).
    params = _params(vehicle={"loads": [10.0], "spacing": [0.0]},
                     num_spans=1)
    steps = [
        {"x": 2.0, "res": {"S1": {"M": np.array([30.0, -30.0]),
                                  "V": np.array([5.0, -5.0])}}},
        {"x": 4.0, "res": {"S1": {"M": np.array([10.0, -10.0]),
                                  "V": np.array([2.0, -2.0])}}},
    ]
    gen = _generator(params, {"Vehicle Steps A": steps})

    try:
        groups = gen._identify_critical_steps(
            params, "System A", {200: (0.0, 0.0)},
            [("Forward", steps)], "Vehicle A")
    finally:
        gen.executor.shutdown(wait=True)

    titles = [p["title"] for g in groups for p in g["plots"]]
    assert len(titles) == 2
    assert any("Min M & Max M" in t for t in titles)
    assert any("Min V & Max V" in t for t in titles)


def test_critical_steps_attribute_extremes_to_governing_direction():
    import numpy as np

    params = _params(vehicle={"loads": [10.0], "spacing": [0.0]},
                     direction="Both", num_spans=1)
    fwd = [{"x": 2.0, "res": {"S1": {"M": np.array([40.0, 0.0]),
                                     "V": np.array([8.0, -1.0])}}}]
    rev = [{"x": 7.0, "res": {"S1": {"M": np.array([5.0, -40.0]),
                                     "V": np.array([1.0, -8.0])}}}]
    gen = _generator(params, {"Vehicle Steps A": fwd, "Vehicle Steps A_Rev": rev})

    try:
        groups = gen._identify_critical_steps(
            params, "System A", {200: (0.0, 0.0)},
            [("Forward", fwd), ("Reverse", rev)], "Vehicle A")
    finally:
        gen.executor.shutdown(wait=True)

    titles = [p["title"] for g in groups for p in g["plots"]]
    # Max M and Max V govern forward, Min M and Min V govern reverse.
    assert any("Max M" in t and "Forward" in t for t in titles)
    assert any("Min M" in t and "Reverse" in t for t in titles)
    assert any("Max V" in t and "Forward" in t for t in titles)
    assert any("Min V" in t and "Reverse" in t for t in titles)


def test_has_any_vehicle_false_when_all_vehicle_lists_are_empty():
    gen = _generator(_params())

    try:
        assert not gen._has_any_vehicle()
    finally:
        gen.executor.shutdown(wait=True)


def test_vehicle_step_sources_merge_directions_per_vehicle():
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

    assert len(sources) == 1
    _, _, sys_label, _, veh_label, dir_steps = sources[0]
    assert (sys_label, veh_label) == ("System A", "Vehicle A")
    assert [d for d, _ in dir_steps] == ["Forward", "Reverse"]
    assert dir_steps[0][1] is raw["Vehicle Steps A"]
    assert dir_steps[1][1] is raw["Vehicle Steps A_Rev"]


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

    assert len(sources) == 1
    dir_steps = sources[0][5]
    assert [d for d, _ in dir_steps] == ["Reverse"]
    assert dir_steps[0][1] is raw["Vehicle Steps A_Rev"]


def test_unfactored_vehicle_table_reports_governing_direction_per_extreme():
    import numpy as np

    params = _params(vehicle={"loads": [10.0], "spacing": [0.0]},
                     direction="Both", num_spans=1)
    fwd = [{"x": 2.0, "res": {"S1": {"M": np.array([40.0, 0.0]),
                                     "V": np.array([8.0, -1.0])}}}]
    rev = [{"x": 7.0, "res": {"S1": {"M": np.array([5.0, -40.0]),
                                     "V": np.array([1.0, -8.0])}}}]
    gen = _generator(params, {"Vehicle Steps A": fwd, "Vehicle Steps A_Rev": rev})

    try:
        gen._add_unfactored_vehicle_table()
    finally:
        gen.executor.shutdown(wait=True)

    # One merged row per element: extremes annotated with their direction.
    from reportlab.platypus import Paragraph, KeepTogether
    tables = [e for e in gen.elements if isinstance(e, KeepTogether)]
    assert tables
    cells = tables[-1]._content[0]._cellvalues
    body = cells[1]
    texts = [c.text if isinstance(c, Paragraph) else c for c in body]
    assert texts[0] == "S1"
    assert texts[1] == "40.0 (Fwd)"   # M_max governed forward
    assert texts[2] == "-40.0 (Rev)"  # M_min governed reverse
    assert texts[3] == "8.0 (Fwd)"
    assert texts[4] == "-8.0 (Rev)"
    assert len(cells) == 2  # header + ONE row for S1 (directions merged)


def test_system_has_component_checks_each_load_case():
    p = _params()
    p["sw_list"] = [0.0] * 10
    assert not BricosReportGenerator._system_has_component(p, "Dead Load")
    p["sw_list"][0] = 5.0
    assert BricosReportGenerator._system_has_component(p, "Dead Load")

    assert not BricosReportGenerator._system_has_component(p, "Soil")
    p["soil"] = [{"wall_idx": 0, "face": "L", "h": 6.0, "q_top": 0.0, "q_bot": 20.0}]
    assert BricosReportGenerator._system_has_component(p, "Soil")

    assert not BricosReportGenerator._system_has_component(p, "Surcharge")
    p["surcharge"] = [{"wall_idx": 0, "q": 5.0}]
    assert BricosReportGenerator._system_has_component(p, "Surcharge")

    p["udl_q"] = 0.0
    assert not BricosReportGenerator._system_has_component(p, "Traffic UDL")
    p["udl_q"] = 4.0
    assert BricosReportGenerator._system_has_component(p, "Traffic UDL")


# --- Dynamic factor (Phi) visibility vs. analyzed limit states (v0.72) ---

def _flatten_report_text(gen):
    """All rendered text in gen.elements: Paragraph text plus table cells."""
    from reportlab.platypus import Paragraph, KeepTogether, Table
    out = []

    def walk(e):
        if isinstance(e, Paragraph):
            out.append(e.text)
        elif isinstance(e, KeepTogether):
            for c in e._content:
                walk(c)
        elif isinstance(e, Table):
            for row in e._cellvalues:
                for c in row:
                    walk(c)
        elif isinstance(e, str):
            out.append(e)

    for e in gen.elements:
        walk(e)
    return " || ".join(out)


def _phi_params(**overrides):
    p = _params(vehicle={"loads": [10.0], "spacing": [0.0]}, num_spans=1)
    p.update({"phi_mode": "Manual", "phi": 1.20})
    p.update(overrides)
    return p


def test_dynamic_factor_section_shows_sls_when_sls_analyzed():
    gen = _generator(_phi_params())
    gen._add_dynamic_factor_section(
        _phi_params(phi_sls_mode="Reduced"),
        {"phi_calc": 1.20, "Phi Members": {}},
        analyze_sls=True,
    )
    txt = _flatten_report_text(gen)
    assert "Φ (ULS)" in txt
    assert "Φ (SLS)" in txt
    assert "SLS reduction enabled" in txt


def test_dynamic_factor_section_hides_sls_when_sls_not_analyzed():
    gen = _generator(_phi_params())
    gen._add_dynamic_factor_section(
        _phi_params(phi_sls_mode="Reduced"),
        {"phi_calc": 1.20, "Phi Members": {}},
        analyze_sls=False,
    )
    txt = _flatten_report_text(gen)
    assert "Φ (ULS)" in txt           # base/ULS column stays
    assert "Φ (SLS)" not in txt       # SLS column dropped
    assert "SLS reduction" not in txt  # SLS treatment note dropped


def test_input_summary_omits_phi_when_no_limit_state_analyzed():
    # Unfactored-only: Phi is never applied, so the settings line drops the
    # dynamic factor, no Dynamic Factor section is added, and the factor table
    # documents the actual combination (all loads at factor 1.0, no Phi)
    # instead of the stored ULS/SLS factors and "Phi applied".
    p = _phi_params(analyze_uls=False, analyze_sls=False, gamma_veh=1.4)
    gen = _generator(p)
    gen._add_system_input_summary("System A", p, {"phi_calc": 1.20, "Phi Members": {}},
                                  {"Spans": {}, "Walls": {}}, "sysA")
    txt = _flatten_report_text(gen)
    assert "none (unfactored combination only)" in txt
    assert "Φ:" not in txt
    assert "Dynamic Factor" not in txt
    # Factor table reflects the unfactored combination, not the stored factors.
    assert "Combination factor" in txt
    assert "Φ applied" not in txt
    assert "1.4" not in txt  # stored vehicle ULS factor must not appear


def test_input_summary_keeps_phi_when_a_limit_state_analyzed():
    p = _phi_params(analyze_uls=True, analyze_sls=False)
    gen = _generator(p)
    gen._add_system_input_summary("System A", p, {"phi_calc": 1.20, "Phi Members": {}},
                                  {"Spans": {}, "Walls": {}}, "sysA")
    txt = _flatten_report_text(gen)
    assert "Φ:" in txt
    assert "Dynamic Factor" in txt
    assert "Φ applied" in txt  # the vehicle row keeps the dynamic factor note


def test_dynamic_factor_section_sls_only_drops_uls_column():
    # SLS-only analysis: the Φ (ULS) column is dropped, Φ (SLS) remains.
    gen = _generator(_phi_params())
    gen._add_dynamic_factor_section(
        _phi_params(phi_sls_mode="Reduced"),
        {"phi_calc": 1.20, "Phi Members": {}},
        analyze_uls=False, analyze_sls=True,
    )
    txt = _flatten_report_text(gen)
    assert "Φ (SLS)" in txt
    assert "Φ (ULS)" not in txt
    assert "SLS reduction enabled" in txt


def test_input_summary_phi_line_lists_each_analyzed_limit_state():
    # Both limit states, SLS reduced: the settings line reports a value per
    # analyzed limit state (SLS reflecting the reduction of the base phi).
    p = _phi_params(analyze_uls=True, analyze_sls=True, phi=1.20, phi_sls_mode="Reduced")
    gen = _generator(p)
    gen._add_system_input_summary("System A", p, {"phi_calc": 1.20, "Phi Members": {}},
                                  {"Spans": {}, "Walls": {}}, "sysA")
    txt = _flatten_report_text(gen)
    assert "ULS 1.200" in txt
    assert "SLS 1.100" in txt   # 1 + (1.20 - 1)/2


def test_input_summary_phi_line_sls_only():
    p = _phi_params(analyze_uls=False, analyze_sls=True, phi=1.20, phi_sls_mode="Same")
    gen = _generator(p)
    gen._add_system_input_summary("System A", p, {"phi_calc": 1.20, "Phi Members": {}},
                                  {"Spans": {}, "Walls": {}}, "sysA")
    txt = _flatten_report_text(gen)
    assert "SLS 1.200" in txt
    assert "ULS 1.200" not in txt


# --- Single-system reporting: the report fills its primary slot with the first
# system that solved, so a report can be produced for either system alone. ---

_SLOT_NODES = {200: (0.0, 0.0), 201: (10.0, 0.0)}


def _slot_state():
    return {
        "sysA": _params(name="Sys A model"),
        "sysB": _params(name="Sys B model"),
        "model_props_A": {"Spans": {}, "Walls": {}},
        "model_props_B": {"Spans": {}, "Walls": {}},
    }


def test_report_keeps_system_a_primary_when_both_valid():
    state = _slot_state()
    gen = BricosReportGenerator(io.BytesIO(), {}, state,
                                {"Dead Load": {}}, {"Dead Load": {}},
                                _SLOT_NODES, _SLOT_NODES)
    assert gen.valid_B is True
    assert gen.label_A == "System A" and gen.key_A == "sysA"
    assert gen.label_B == "System B" and gen.key_B == "sysB"
    assert gen.params_A is state["sysA"]
    assert gen.params_B is state["sysB"]


def test_report_uses_system_a_alone_when_system_b_invalid():
    state = _slot_state()
    gen = BricosReportGenerator(io.BytesIO(), {}, state,
                                {"Dead Load": {}}, None, _SLOT_NODES, None)
    assert gen.valid_B is False
    assert gen.label_A == "System A" and gen.key_A == "sysA"
    assert gen.params_A is state["sysA"]
    assert gen.nodes_A is _SLOT_NODES


def test_report_promotes_system_b_to_primary_when_system_a_invalid():
    state = _slot_state()
    raw_b = {"Dead Load": {}}
    gen = BricosReportGenerator(io.BytesIO(), {}, state,
                                None, raw_b, None, _SLOT_NODES)
    # Only one system is valid, so the second slot is not rendered ...
    assert gen.valid_B is False
    # ... but the single valid system keeps its true identity in slot A.
    assert gen.label_A == "System B" and gen.key_A == "sysB"
    assert gen.params_A is state["sysB"]
    assert gen.raw_A is raw_b
    assert gen.nodes_A is _SLOT_NODES


def test_report_requires_at_least_one_valid_system():
    state = _slot_state()
    with pytest.raises(ValueError):
        BricosReportGenerator(io.BytesIO(), {}, state, None, None, None, None)