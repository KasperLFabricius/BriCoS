import numpy as np
import pytest

import bricos_data as data
import bricos_results_ui as results_ui
import bricos_solver as solver
from bricos_report import BricosReportGenerator


def _frame_params_with_zero_height_walls(load_q=10.0):
    params = data.get_clear("A", "Frame")
    params.update({
        "mode": "Frame",
        "num_spans": 1,
        "L_list": [10.0] * 10,
        "Is_list": [1.0] * 10,
        "sw_list": [load_q] + [0.0] * 9,
        # Zero-height walls: the solver places the restraints directly at
        # the span top nodes (200+i) instead of wall base nodes (100+i).
        "h_list": [0.0] * 11,
        "Iw_list": [0.0] * 11,
        "E": 30e6,
        "E_span_list": [30e6] * 10,
        "E_wall_list": [30e6] * 11,
        "mesh_size": 0.5,
        "phi_mode": "Manual",
        "phi": 1.0,
    })
    return params


def test_solver_reports_restrained_nodes_for_zero_height_frame_walls():
    params = _frame_params_with_zero_height_walls()

    raw, nodes, _, err = solver.run_raw_analysis(params)

    assert err == 0
    assert raw["Restrained Nodes"] == [200, 201]
    assert all(nid in nodes for nid in raw["Restrained Nodes"])


def test_reaction_envelope_includes_zero_height_wall_supports():
    q = 10.0
    length = 10.0
    params = _frame_params_with_zero_height_walls(load_q=q)

    raw, nodes, _, err = solver.run_raw_analysis(params)
    assert err == 0
    combined = solver.combine_results(raw, params, "Characteristic (No Dynamic Factor)")

    reacts = results_ui.get_reaction_envelope(
        combined["Total Envelope"], nodes, params["mode"], combined["Restrained Nodes"]
    )

    # The legacy Frame-mode heuristic (support node y < -0.01) returned an
    # empty dict for this model because the restraints sit at y = 0.
    assert set(reacts.keys()) == {200, 201}
    total_ry = reacts[200]["Ry_max"] + reacts[201]["Ry_max"]
    assert total_ry == pytest.approx(q * length, rel=1e-6)


def test_reaction_envelope_keeps_legacy_heuristic_without_restrained_nodes():
    # Results produced before 'Restrained Nodes' existed must keep working:
    # Beam/Superstructure heuristic treats node ids >= 200 as supports.
    res = {
        "S1": {
            "x": np.array([0.0, 1.0]),
            "M_max": np.array([0.0, 0.0]), "M_min": np.array([0.0, 0.0]),
            "V_max": np.array([5.0, -5.0]), "V_min": np.array([5.0, -5.0]),
            "N_max": np.array([0.0, 0.0]), "N_min": np.array([0.0, 0.0]),
            "cx": 1.0, "cy": 0.0,
            "ni_id": 200, "nj_id": 201,
        }
    }
    nodes = {200: (0.0, 0.0), 201: (10.0, 0.0)}

    reacts = results_ui.get_reaction_envelope(res, nodes, "Beam")

    assert set(reacts.keys()) == {200, 201}
    assert reacts[200]["Ry_max"] == pytest.approx(5.0)
    assert reacts[201]["Ry_max"] == pytest.approx(5.0)


def test_report_valid_support_nodes_prefers_solver_restrained_nodes():
    raw = {"Restrained Nodes": [100, 201]}
    params = {"mode": "Frame", "num_spans": 1}

    assert BricosReportGenerator._valid_support_nodes(params, raw) == {100, 201}


def test_report_valid_support_nodes_falls_back_to_legacy_heuristic():
    assert BricosReportGenerator._valid_support_nodes(
        {"mode": "Frame", "num_spans": 1}, {}
    ) == {100, 101}
    assert BricosReportGenerator._valid_support_nodes(
        {"mode": "Superstructure", "num_spans": 2}, None
    ) == {200, 201, 202}
