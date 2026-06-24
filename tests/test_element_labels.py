"""v0.82: element-name labels (S1, S2, W1, ...) on the diagrams.

Covers the geometry-match dedup, the label-drawing helper, the create_plotly_fig
wiring (off by default, per-system colours, single neutral label when A and B
share geometry, font scaling with diagram height) and the structure-only mode
used by the report's element-layout diagram.
"""
import numpy as np
import plotly.graph_objects as go

import bricos_viz as viz


GEOM = {
    'S1': {'ni': (0.0, 0.0), 'nj': (10.0, 0.0), 'L': 10.0},
    'S2': {'ni': (10.0, 0.0), 'nj': (20.0, 0.0), 'L': 10.0},
    'W1': {'ni': (0.0, 0.0), 'nj': (0.0, -6.0), 'L': 6.0},
}
GEOM_SHIFTED = {
    eid: {'ni': (d['ni'][0] + 5.0, d['ni'][1]),
          'nj': (d['nj'][0] + 5.0, d['nj'][1]), 'L': d['L']}
    for eid, d in GEOM.items()
}


# --- geometry-match dedup ---------------------------------------------------

def test_geoms_match_identical():
    assert viz._geoms_match(GEOM, {k: dict(v) for k, v in GEOM.items()}) is True


def test_geoms_match_different_coords():
    assert viz._geoms_match(GEOM, GEOM_SHIFTED) is False


def test_geoms_match_different_element_sets():
    g2 = {k: v for k, v in GEOM.items() if k != 'W1'}
    assert viz._geoms_match(GEOM, g2) is False


def test_geoms_match_empty_is_false():
    assert viz._geoms_match(GEOM, {}) is False
    assert viz._geoms_match({}, GEOM) is False


# --- label-drawing helper ---------------------------------------------------

def test_draw_element_labels_one_per_element_in_colour():
    fig = go.Figure()
    viz._draw_element_labels(fig, GEOM, 'blue', 12.0)
    texts = sorted(a.text for a in fig.layout.annotations)
    assert texts == ['S1', 'S2', 'W1']
    assert all(a.font.color == 'blue' for a in fig.layout.annotations)


# --- create_plotly_fig wiring ----------------------------------------------

def _label_anns(fig):
    return [a for a in fig.layout.annotations if a.text in ('S1', 'S2', 'W1')]


def test_labels_off_by_default():
    fig = viz.create_plotly_fig({}, GEOM, {}, 'M', show_A=True, show_B=False, geom_A=GEOM)
    assert _label_anns(fig) == []


def test_labels_per_system_when_geometry_differs():
    fig = viz.create_plotly_fig(
        {}, GEOM, GEOM_SHIFTED, 'M', show_A=True, show_B=True,
        geom_A=GEOM, geom_B=GEOM_SHIFTED, show_element_names=True)
    colors = [a.font.color for a in _label_anns(fig)]
    assert colors.count('blue') == 3
    assert colors.count('red') == 3


def test_single_neutral_label_when_geometry_identical():
    fig = viz.create_plotly_fig(
        {}, GEOM, {k: dict(v) for k, v in GEOM.items()}, 'M', show_A=True, show_B=True,
        geom_A=GEOM, geom_B={k: dict(v) for k, v in GEOM.items()}, show_element_names=True)
    anns = _label_anns(fig)
    assert len(anns) == 3
    assert all(a.font.color == '#333333' for a in anns)


def test_element_label_font_scales_with_diagram_height():
    small = viz.create_plotly_fig({}, GEOM, {}, 'M', target_height=1.0, show_A=True,
                                  show_B=False, geom_A=GEOM, show_element_names=True)
    big = viz.create_plotly_fig({}, GEOM, {}, 'M', target_height=8.0, show_A=True,
                                show_B=False, geom_A=GEOM, show_element_names=True)
    assert _label_anns(big)[0].font.size > _label_anns(small)[0].font.size


# --- structure-only mode (report element-layout diagram) --------------------

def test_structure_only_suppresses_result_diagram():
    res = {'S1': {'ni': (0.0, 0.0), 'nj': (10.0, 0.0), 'L': 10.0, 'cx': 1.0, 'cy': 0.0,
                  'x': np.array([0.0, 5.0, 10.0]), 'M': np.array([0.0, 10.0, 0.0])}}
    full = viz.create_plotly_fig({}, res, {}, 'M', show_A=True, show_B=False, geom_A=res)
    only = viz.create_plotly_fig({}, res, {}, 'M', show_A=True, show_B=False,
                                 geom_A=res, structure_only=True)
    assert len(only.data) < len(full.data)
