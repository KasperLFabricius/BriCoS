"""v0.76: section-depth (thickness) overlay.

Pure-helper coverage for the toggle that draws each member's true section
depth to scale, centred on the member axis: the inertia gate
(model_uses_inertia_sections), the per-element profile lookup
(_section_profile_for) and the band polygon (section_band_polygon).
"""
import numpy as np
import pytest

import bricos_data as data
import bricos_viz as viz


# --- inertia gate -----------------------------------------------------------

def _clear_height_params():
    p = data.get_clear("A", "Frame")
    p["num_spans"] = 2
    return p


def test_height_only_model_is_not_inertia():
    assert data.model_uses_inertia_sections(_clear_height_params()) is False
    assert data.model_uses_inertia_sections(data.get_def()) is False


def test_inertia_span_or_wall_flags_model():
    p = _clear_height_params()
    p["span_geom_0"] = {"type": 0, "shape": 0, "vals": [0.1, 0.1, 0.1]}
    assert data.model_uses_inertia_sections(p) is True

    p2 = _clear_height_params()
    p2["wall_geom_1"] = {"type": 0, "shape": 0, "vals": [0.1, 0.1, 0.1]}
    assert data.model_uses_inertia_sections(p2) is True


def test_inertia_section_beyond_active_spans_is_ignored():
    p = _clear_height_params()  # num_spans = 2 -> spans 0,1 active
    p["span_geom_5"] = {"type": 0, "shape": 0, "vals": [0.1, 0.1, 0.1]}
    assert data.model_uses_inertia_sections(p) is False


def test_superstructure_ignores_inertia_walls():
    p = data.get_clear("A", "Superstructure")
    p["num_spans"] = 1
    p["wall_geom_0"] = {"type": 0, "shape": 0, "vals": [0.1, 0.1, 0.1]}
    assert data.model_uses_inertia_sections(p) is False


# --- per-element profile lookup --------------------------------------------

def test_section_profile_prefers_stored_geom():
    params = {"span_geom_0": {"type": 1, "shape": 2, "vals": [0.5, 1.0, 0.5]},
              "Is_list": [9.9] * 10}
    vtype, shape, vals = viz._section_profile_for(params, "S1")
    assert (vtype, shape) == (1, 2)
    assert vals == [0.5, 1.0, 0.5]


def test_section_profile_falls_back_to_simple_list():
    params = {"Is_list": [0.8, 0.7] + [0.0] * 8, "Iw_list": [0.6] * 11}
    assert viz._section_profile_for(params, "S2") == (1, 0, [0.7, 0.7, 0.7])
    assert viz._section_profile_for(params, "W1") == (1, 0, [0.6, 0.6, 0.6])


# --- band polygon -----------------------------------------------------------

def test_constant_deck_band_is_centred_and_to_scale():
    xs, ys = viz.section_band_polygon((0.0, 0.0), (10.0, 0.0), 10.0,
                                      val_type=1, shape=0, vals=[1.0, 1.0, 1.0], b_eff=1.0)
    xs, ys = np.asarray(xs), np.asarray(ys)
    # Depth 1.0 m, centred on the axis (y = 0): band spans +-0.5 m vertically.
    assert ys.max() == pytest.approx(0.5)
    assert ys.min() == pytest.approx(-0.5)
    assert xs.min() >= -1e-9 and xs.max() <= 10.0 + 1e-9
    # Closed polygon (first point repeated at the end).
    assert xs[0] == pytest.approx(xs[-1]) and ys[0] == pytest.approx(ys[-1])


def test_linear_taper_band_follows_end_depths():
    xs, ys = viz.section_band_polygon((0.0, 0.0), (10.0, 0.0), 10.0,
                                      val_type=1, shape=1, vals=[1.5, 0.0, 0.5], b_eff=1.0)
    ys = np.asarray(ys)
    # +-h/2: start depth 1.5 -> 0.75, end depth 0.5 -> 0.25.
    assert ys.max() == pytest.approx(0.75)
    assert ys.min() == pytest.approx(-0.75)
    # The thin end only reaches +-0.25.
    xs = np.asarray(xs)
    end_top = ys[np.argmax(xs)]
    assert abs(end_top) == pytest.approx(0.25, abs=1e-6)


def test_three_point_band_uses_mid_depth():
    xs, ys = viz.section_band_polygon((0.0, 0.0), (10.0, 0.0), 10.0,
                                      val_type=1, shape=2, vals=[0.5, 1.0, 0.5], b_eff=1.0)
    ys = np.asarray(ys)
    # Mid depth 1.0 -> +-0.5 is the maximum half-depth along the member.
    assert ys.max() == pytest.approx(0.5)
    assert ys.min() == pytest.approx(-0.5)


def test_vertical_wall_band_is_horizontal():
    xs, ys = viz.section_band_polygon((0.0, 0.0), (0.0, 5.0), 5.0,
                                      val_type=1, shape=0, vals=[0.4, 0.4, 0.4], b_eff=1.0)
    xs, ys = np.asarray(xs), np.asarray(ys)
    # Vertical member -> depth offsets horizontally (+-0.2 m), height spans 0..5.
    assert xs.max() == pytest.approx(0.2)
    assert xs.min() == pytest.approx(-0.2)
    assert ys.min() == pytest.approx(0.0) and ys.max() == pytest.approx(5.0)


def test_inertia_value_maps_to_rectangular_depth():
    # I = b_eff * h^3 / 12 with b_eff=1, h=1 -> I = 1/12, so the band depth = 1.0.
    xs, ys = viz.section_band_polygon((0.0, 0.0), (4.0, 0.0), 4.0,
                                      val_type=0, shape=0, vals=[1.0 / 12.0] * 3, b_eff=1.0)
    ys = np.asarray(ys)
    assert ys.max() == pytest.approx(0.5, abs=1e-6)
    assert ys.min() == pytest.approx(-0.5, abs=1e-6)


def test_degenerate_member_yields_empty_polygon():
    assert viz.section_band_polygon((0.0, 0.0), (0.0, 0.0), 0.0, 1, 0, [1.0, 1.0, 1.0], 1.0) == ([], [])
