"""Distributed element spring supports (elastic foundation): tributary
lumping, closed-form settlement, equilibrium, reaction aggregation keys,
validation and persistence plumbing."""
import numpy as np
import pytest

import bricos_data as data
import bricos_results_ui as results_ui
import bricos_solver as solver


# --- tributary lumping (pure helper) ----------------------------------------

def test_tributary_lumping_sums_to_k_per_m_times_length():
    # 3 equal sub-elements of 2 m on parent S1.
    nodes = {200: (0.0, 0.0), 1000: (2.0, 0.0), 1001: (4.0, 0.0), 201: (6.0, 0.0)}
    elems = [
        {'parent': 'S1', 'nodes': (200, 1000)},
        {'parent': 'S1', 'nodes': (1000, 1001)},
        {'parent': 'S1', 'nodes': (1001, 201)},
    ]
    cfg = [{'el': 'S1', 'kx': 100.0, 'ky': 200.0, 'km': 50.0}]

    springs, node_map = solver.accumulate_foundation_springs(nodes, elems, cfg)

    assert node_map == {'S1': {'nodes': [200, 201, 1000, 1001], 'end_trib': 1.0}}
    # End nodes carry one tributary half (1 m), interior nodes two (2 m).
    assert springs[200] == pytest.approx([100.0, 200.0, 50.0])
    assert springs[1000] == pytest.approx([200.0, 400.0, 100.0])
    # Sum per component = k_per_m * L exactly.
    for c, k in enumerate((100.0, 200.0, 50.0)):
        assert sum(v[c] for v in springs.values()) == pytest.approx(k * 6.0)


def test_zero_and_unknown_definitions_are_inert():
    nodes = {200: (0.0, 0.0), 201: (6.0, 0.0)}
    elems = [{'parent': 'S1', 'nodes': (200, 201)}]
    springs, node_map = solver.accumulate_foundation_springs(
        nodes, elems,
        [{'el': 'S1', 'kx': 0.0, 'ky': 0.0, 'km': 0.0},  # all-zero: inert
         {'el': 'S9', 'ky': 1e4},                         # not in mesh
         'garbage', None])                                # malformed
    assert springs == {} and node_map == {}


# --- physics: beam fully carried by its foundation ---------------------------

def _beam_on_foundation(q=20.0, ky=1e4, L=10.0):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam", "num_spans": 1,
        "L_list": [L] * 10, "Is_list": [0.5] * 10,
        "sw_list": [q] + [0.0] * 9,
        "E": 33e6, "E_span_list": [33e6] * 10,
        "mesh_size": 0.5, "phi_mode": "Manual", "phi": 1.0,
        # Zero-stiffness point supports: the foundation must carry all load.
        "supports": [{'type': 'Custom Spring', 'k': [0.0, 0.0, 0.0]},
                     {'type': 'Custom Spring', 'k': [0.0, 0.0, 0.0]}],
        # kx stabilises the horizontal rigid-body mode; km left 0.
        "elastic_foundations": [{'el': 'S1', 'kx': ky, 'ky': ky, 'km': 0.0}],
    })
    return params


def test_uniform_settlement_matches_q_over_k():
    q, ky, L = 20.0, 1e4, 10.0
    params = _beam_on_foundation(q, ky, L)
    raw, nodes, _, err = solver.run_raw_analysis(params)
    assert err == 0

    dl = raw['Dead Load']
    v_expected = -q / ky  # -2.0e-3 m uniform settlement (exact solution)
    for eid, dat in dl.items():
        if 'def_y' not in dat:
            continue
        assert np.allclose(dat['def_y'], v_expected, rtol=5e-3), eid
        # A simply-supported span would carry qL^2/8 = 250 kNm; on the full
        # bed only the end fixed-end-moment imbalance remains (< 1 kNm).
        assert np.max(np.abs(dat['M'])) < 1.0, eid

    eq = raw['Equilibrium']['Dead Load']
    assert eq['applied_y'] == pytest.approx(-q * L, rel=1e-9)
    assert eq['reactions_y'] == pytest.approx(q * L, rel=1e-6)
    assert abs(eq['residual_y']) < 1e-3


def test_reaction_keys_and_foundation_aggregation():
    q, ky, L = 20.0, 1e4, 10.0
    params = _beam_on_foundation(q, ky, L)
    raw, nodes, _, err = solver.run_raw_analysis(params)
    assert err == 0

    # 20 sub-elements -> 21 bedded nodes including both span-end nodes.
    assert raw['Point Support Nodes'] == [200, 201]
    fmap = raw['Foundation Node Map']
    assert set(fmap.keys()) == {'S1'}
    assert len(fmap['S1']['nodes']) == 21
    assert fmap['S1']['end_trib'] == pytest.approx(0.25)
    assert set(raw['Restrained Nodes']) == set(fmap['S1']['nodes'])

    combined = solver.combine_results(raw, params, "Unfactored")
    assert combined['Foundation Node Map'] == fmap
    assert combined['Point Support Nodes'] == [200, 201]

    # Point rows (end-force sums at the span-end nodes) carry the local
    # bedding tributary share: k*end_trib*|v| = 1e4*0.25*2e-3 = 5 kN each.
    reacts = results_ui.get_reaction_envelope(
        combined['Total Envelope'], nodes, params['mode'],
        combined['Point Support Nodes'])
    assert set(reacts.keys()) == {200, 201}
    point_ry = sum(r['Ry_max'] for r in reacts.values())
    assert point_ry == pytest.approx(10.0, rel=1e-2)

    # The foundation row excludes those end shares, so all rows together
    # recover the applied load exactly.
    fnd_r = solver.foundation_reaction_resultants(
        combined['Total Envelope'], params['elastic_foundations'],
        combined['Foundation Node Map'], combined['Point Support Nodes'])
    assert set(fnd_r.keys()) == {'S1'}
    assert fnd_r['S1']['Rx_max'] == pytest.approx(0.0, abs=1e-6)
    assert point_ry + fnd_r['S1']['Ry_max'] == pytest.approx(q * L, rel=5e-3)


def test_frame_wall_foundation_solves_and_maps_nodes():
    params = data.get_clear("A", "Frame")
    params.update({
        "mode": "Frame", "num_spans": 1,
        "L_list": [10.0] * 10, "Is_list": [0.5] * 10,
        "h_list": [6.0] * 11, "Iw_list": [0.5] * 11,
        "sw_list": [20.0] + [0.0] * 9,
        "E": 33e6, "E_span_list": [33e6] * 10, "E_wall_list": [33e6] * 11,
        "mesh_size": 0.5, "phi_mode": "Manual", "phi": 1.0,
        "elastic_foundations": [{'el': 'W1', 'kx': 5e3, 'ky': 0.0, 'km': 0.0}],
    })
    raw, nodes, _, err = solver.run_raw_analysis(params)
    assert err == 0
    fmap = raw['Foundation Node Map']
    # 12 sub-elements on the 6 m wall -> 13 nodes including base 100, top 200.
    assert set(fmap.keys()) == {'W1'}
    assert len(fmap['W1']['nodes']) == 13
    assert 100 in fmap['W1']['nodes'] and 200 in fmap['W1']['nodes']
    # Wall bases stay point supports.
    assert raw['Point Support Nodes'] == [100, 101]
    assert abs(raw['Equilibrium']['Dead Load']['residual_y']) < 1e-3


# --- validation & persistence plumbing ---------------------------------------

def _base_beam_params():
    params = data.get_clear("A", "Beam")
    params.update({"mode": "Beam", "num_spans": 1, "L_list": [10.0] * 10,
                   "Is_list": [0.5] * 10})
    return params


def test_validation_rejects_bad_definitions():
    params = _base_beam_params()
    params['elastic_foundations'] = [{'el': 'W1', 'ky': 1e4}]
    assert any('Frame mode' in e for e in data.validate_analysis_inputs(params, "A"))

    params['elastic_foundations'] = [{'el': 'S5', 'ky': 1e4}]
    assert any('not present' in e for e in data.validate_analysis_inputs(params, "A"))

    params['elastic_foundations'] = [{'el': 'S1', 'ky': -5.0}]
    assert any('negative' in e for e in data.validate_analysis_inputs(params, "A"))

    params['elastic_foundations'] = [{'el': 'S1', 'ky': 1e4},
                                     {'el': 'S1', 'kx': 1.0}]
    assert any('more than once' in e for e in data.validate_analysis_inputs(params, "A"))


def test_validation_accepts_valid_definition():
    params = _base_beam_params()
    params['elastic_foundations'] = [{'el': 'S1', 'kx': 0.0, 'ky': 1e4, 'km': 0.0}]
    errors = data.validate_analysis_inputs(params, "A")
    assert not any('spring support' in e for e in errors)


def test_report_reaction_table_appends_foundation_row():
    import io
    from bricos_report import BricosReportGenerator

    q, ky, L = 20.0, 1e4, 10.0
    params = _beam_on_foundation(q, ky, L)
    raw, nodes, _, err = solver.run_raw_analysis(params)
    assert err == 0
    combined = solver.combine_results(raw, params, "Unfactored")

    state = {"sysA": params, "sysB": data.get_clear("B", "Beam"),
             "model_props_A": {"Spans": {}, "Walls": {}},
             "model_props_B": {"Spans": {}, "Walls": {}}}
    gen = BricosReportGenerator(io.BytesIO(), {}, state, raw, {}, nodes, None)
    # Point supports only in the per-node listing...
    assert gen._valid_support_nodes(params, raw) == {200, 201}

    gen._add_reaction_table(combined['Total Envelope'], params, {}, None)
    tables = []
    for el in gen.elements:
        for inner in getattr(el, '_content', [el]):  # unwrap KeepTogether
            if hasattr(inner, '_cellvalues'):
                tables.append(inner)
    assert tables

    def cell(v):  # cells are Paragraphs
        return v.getPlainText() if hasattr(v, 'getPlainText') else str(v)

    rows = [[cell(v) for v in r] for r in tables[-1]._cellvalues]
    labels = [r[0] for r in rows]
    # ...plus one integrated resultant row for the bedded member.
    assert "S1 fnd." in labels
    fnd_row = rows[labels.index("S1 fnd.")]
    # Ry max ~ interior foundation share (applied 200 kN minus the two
    # 5 kN end-node shares carried by the support rows); Mz shown as --.
    assert float(fnd_row[3]) == pytest.approx(190.0, rel=2e-2)
    assert fnd_row[5] == "--" and fnd_row[6] == "--"


def test_cache_key_and_sanitize_keep_foundations():
    params = _base_beam_params()
    params['elastic_foundations'] = [{'el': 'S1', 'ky': 1e4}]
    assert solver.solver_cache_params(params).get('elastic_foundations') == \
        [{'el': 'S1', 'ky': 1e4}]
    # Older sessions without the key are seeded with an empty list.
    stale = {k: v for k, v in params.items() if k != 'elastic_foundations'}
    assert data.sanitize_input_data(stale)['elastic_foundations'] == []
