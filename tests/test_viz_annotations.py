import numpy as np
import pytest

import bricos_viz as viz


def _ann(x, y, text="123.4", perp=(0.0, 1.0)):
    return {"x": x, "y": y, "text": text, "color": "blue",
            "perp_x": perp[0], "perp_y": perp[1]}


def test_solve_annotations_legacy_footprint_without_extent():
    anns = viz.solve_annotations([_ann(0.0, 0.0)])
    assert anns[0]["w"] == pytest.approx(len("123.4") * 0.15)
    assert anns[0]["h"] == pytest.approx(0.40)


def test_solve_annotations_footprint_scales_with_extent_and_font():
    base = viz.solve_annotations([_ann(0.0, 0.0)], extent_x=15.0, font_scale=1.0)
    double = viz.solve_annotations([_ann(0.0, 0.0)], extent_x=30.0, font_scale=1.0)
    scaled = viz.solve_annotations([_ann(0.0, 0.0)], extent_x=15.0, font_scale=1.5)

    assert double[0]["w"] == pytest.approx(2.0 * base[0]["w"])
    assert double[0]["h"] == pytest.approx(2.0 * base[0]["h"])
    assert scaled[0]["w"] == pytest.approx(1.5 * base[0]["w"])

    # The extent-aware footprint matches the legacy calibration (~15 m
    # structure at font scale 1) within ~10%, so existing plots keep their
    # look while longer decks get proportionally larger footprints.
    assert base[0]["w"] == pytest.approx(len("123.4") * 0.15, rel=0.1)
    assert base[0]["h"] == pytest.approx(0.40, rel=0.1)


def test_solve_annotations_separates_stacked_labels_on_long_decks():
    # Two labels at the same spot on a 36 m deck. With the legacy fixed
    # footprint the solver considered them ~0.75 m wide and left them
    # visually stacked; the extent-aware footprint must push them at least
    # a label-height apart.
    anns = viz.solve_annotations(
        [_ann(18.0, 2.0), _ann(18.0, 2.0)], extent_x=36.0, font_scale=1.5)

    dist_y = abs(anns[0]["y"] - anns[1]["y"])
    dist_x = abs(anns[0]["x"] - anns[1]["x"])
    box_h = anns[0]["h"]
    assert max(dist_x / max(anns[0]["w"], 1e-9), dist_y / box_h) >= 1.0


def test_solve_annotations_empty_input():
    assert viz.solve_annotations([]) == []


def test_structure_extent_x_spans_given_sources_only():
    # Sizing must follow the systems actually shown: a hidden comparison
    # system with a longer deck must not inflate the extent (report
    # critical-step plots pass both geometries with one show flag).
    sys_a = {"S1": {"ni": (0.0, 0.0), "nj": (10.0, 0.0)}}
    sys_b = {"S1": {"ni": (0.0, 0.0), "nj": (36.0, 0.0)},
             "meta": {"no_geometry": True}}

    assert viz.structure_extent_x([sys_a]) == pytest.approx(10.0)
    assert viz.structure_extent_x([sys_a, sys_b]) == pytest.approx(36.0)
    assert viz.structure_extent_x([sys_a, None, {}]) == pytest.approx(10.0)
    assert viz.structure_extent_x([]) is None
    # Zero horizontal extent (single wall) falls back to None -> legacy
    # footprint constants.
    wall = {"W1": {"ni": (5.0, 0.0), "nj": (5.0, -8.0)}}
    assert viz.structure_extent_x([wall]) is None
