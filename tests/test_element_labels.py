"""v0.82: element-name labels (S1, S2, W1, ...) on the diagrams.

Covers the per-element coordinate match, the chip-drawing helpers, and the
create_plotly_fig wiring: off by default; per-system colours; a single neutral
chip only for elements that actually overlap between A and B (decided per
element); labels gated on the system whose result is actually plotted; font
scaling with diagram height; and the structure-only mode used by the report.
"""
import re

import numpy as np
import plotly.graph_objects as go

import bricos_viz as viz


GEOM = {
    'S1': {'ni': (0.0, 0.0), 'nj': (10.0, 0.0), 'L': 10.0},
    'W1': {'ni': (0.0, 0.0), 'nj': (0.0, -6.0), 'L': 6.0},
    'W2': {'ni': (10.0, 0.0), 'nj': (10.0, -6.0), 'L': 6.0},
}
GEOM_SHIFTED = {
    eid: {'ni': (d['ni'][0] + 5.0, d['ni'][1]),
          'nj': (d['nj'][0] + 5.0, d['nj'][1]), 'L': d['L']}
    for eid, d in GEOM.items()
}
# Same W1 as GEOM, but a longer S1 so W2 is offset: only W1 overlaps.
GEOM_PARTIAL = {
    'S1': {'ni': (0.0, 0.0), 'nj': (12.0, 0.0), 'L': 12.0},
    'W1': {'ni': (0.0, 0.0), 'nj': (0.0, -6.0), 'L': 6.0},
    'W2': {'ni': (12.0, 0.0), 'nj': (12.0, -6.0), 'L': 6.0},
}


def _chips(fig):
    return [a for a in fig.layout.annotations if a.text and re.match(r'^[SW]\d+$', a.text)]


def _bg_counts(fig):
    counts = {}
    for a in _chips(fig):
        counts[a.bgcolor] = counts.get(a.bgcolor, 0) + 1
    return counts


# --- per-element coordinate match -------------------------------------------

def test_element_coords_match_same_and_different():
    a = {'ni': (0.0, 0.0), 'nj': (10.0, 0.0)}
    assert viz._element_coords_match(a, dict(a)) is True
    assert viz._element_coords_match(a, {'ni': (1.0, 0.0), 'nj': (10.0, 0.0)}) is False


def test_element_coords_match_missing_node_is_false():
    assert viz._element_coords_match({'ni': (0.0, 0.0)}, {'ni': (0.0, 0.0), 'nj': (1.0, 0.0)}) is False


# --- chip style + single-system drawing -------------------------------------

def test_chip_is_white_text_on_colour_fill():
    """Distinct from value labels (white box, coloured text)."""
    fig = go.Figure()
    viz._draw_element_labels(fig, GEOM, 'blue', 12.0)
    chips = _chips(fig)
    assert sorted(a.text for a in chips) == ['S1', 'W1', 'W2']
    assert all(a.font.color == 'white' for a in chips)
    assert all(a.bgcolor == 'blue' for a in chips)


# --- create_plotly_fig wiring ----------------------------------------------

def test_labels_off_by_default():
    fig = viz.create_plotly_fig({}, GEOM, {}, 'M', show_A=True, show_B=False, geom_A=GEOM)
    assert _chips(fig) == []


def test_labels_per_system_when_geometry_fully_differs():
    fig = viz.create_plotly_fig(
        {}, GEOM, GEOM_SHIFTED, 'M', show_A=True, show_B=True,
        geom_A=GEOM, geom_B=GEOM_SHIFTED, show_element_names=True)
    assert _bg_counts(fig) == {'blue': 3, 'red': 3}


def test_single_neutral_chip_when_geometry_identical():
    fig = viz.create_plotly_fig(
        {}, GEOM, {k: dict(v) for k, v in GEOM.items()}, 'M', show_A=True, show_B=True,
        geom_A=GEOM, geom_B={k: dict(v) for k, v in GEOM.items()}, show_element_names=True)
    assert _bg_counts(fig) == {'#333333': 3}


def test_partial_overlap_merges_only_the_coinciding_element():
    # W1 coincides -> one neutral chip; S1 and W2 differ -> blue + red each.
    fig = viz.create_plotly_fig(
        {}, GEOM, GEOM_PARTIAL, 'M', show_A=True, show_B=True,
        geom_A=GEOM, geom_B=GEOM_PARTIAL, show_element_names=True)
    assert _bg_counts(fig) == {'#333333': 1, 'blue': 2, 'red': 2}
    neutral = [a.text for a in _chips(fig) if a.bgcolor == '#333333']
    assert neutral == ['W1']


def test_labels_gated_on_actually_plotted_data():
    # show_B is on and geom_B is present, but System B has no result data for
    # this case (sysB_data empty) so it is not plotted -> no red chips, even
    # though the geometries are identical (no spurious neutral merge either).
    fig = viz.create_plotly_fig(
        {}, GEOM, {}, 'M', show_A=True, show_B=True,
        geom_A=GEOM, geom_B=GEOM, show_element_names=True)
    assert _bg_counts(fig) == {'blue': 3}


def test_element_label_font_scales_with_diagram_height():
    small = viz.create_plotly_fig({}, GEOM, {}, 'M', target_height=1.0, show_A=True,
                                  show_B=False, geom_A=GEOM, show_element_names=True)
    big = viz.create_plotly_fig({}, GEOM, {}, 'M', target_height=8.0, show_A=True,
                                show_B=False, geom_A=GEOM, show_element_names=True)
    assert _chips(big)[0].font.size > _chips(small)[0].font.size


# --- structure-only mode (report element-layout diagram) --------------------

def test_structure_only_suppresses_result_diagram():
    res = {'S1': {'ni': (0.0, 0.0), 'nj': (10.0, 0.0), 'L': 10.0, 'cx': 1.0, 'cy': 0.0,
                  'x': np.array([0.0, 5.0, 10.0]), 'M': np.array([0.0, 10.0, 0.0])}}
    full = viz.create_plotly_fig({}, res, {}, 'M', show_A=True, show_B=False, geom_A=res)
    only = viz.create_plotly_fig({}, res, {}, 'M', show_A=True, show_B=False,
                                 geom_A=res, structure_only=True)
    assert len(only.data) < len(full.data)
