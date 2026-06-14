import numpy as np

import bricos_data as data
import bricos_kernels as kernels
import bricos_solver as solver


def _empty_vehicle():
    return {"loads": [], "spacing": []}


def _frame_params(span_geom, *, use_shear_def=False):
    params = data.get_clear("A", "Frame")
    params.update({
        "mode": "Frame",
        "num_spans": 1,
        "L_list": [10.0] * 10,
        "h_list": [8.0] * 11,
        "Iw_list": [0.5] * 11,
        "Is_list": [0.5] * 10,
        "sw_list": [20.0] + [0.0] * 9,
        "E": 33e6,
        "E_span_list": [33e6] * 10,
        "E_wall_list": [33e6] * 11,
        "mesh_size": 2.0,
        "step_size": 1.0,
        "vehicle": _empty_vehicle(),
        "vehicleB": _empty_vehicle(),
        "soil": [],
        "surcharge": [],
        "use_shear_def": use_shear_def,
        "b_eff": 3.0,
        "phi_mode": "Manual",
        "phi": 1.0,
        "span_geom_0": span_geom,
    })
    for i in range(2):
        params[f"wall_geom_{i}"] = {
            "type": 1,
            "shape": 0,
            "vals": [0.5, 0.5, 0.5],
        }
    return params


def _span_geom(vals, shape):
    return {
        "type": 1,
        "shape": shape,
        "vals": vals,
        "align_type": 1,
        "incline_mode": 0,
        "incline_val": 5.0,
    }


def _run(params):
    raw, nodes, props, err = solver.run_raw_analysis(params)
    assert err == 0
    assert nodes is not None
    assert props is not None
    return raw


def _selfweight_span(raw):
    assert "S1" in raw["Dead Load"]
    return raw["Dead Load"]["S1"]


def _assert_thicker_midspan_frame_is_not_more_flexible(*, use_shear_def):
    constant = _run(_frame_params(
        _span_geom([0.5, 0.5, 0.5], 0),
        use_shear_def=use_shear_def,
    ))
    thicker_mid = _run(_frame_params(
        _span_geom([0.5, 0.6, 0.5], 2),
        use_shear_def=use_shear_def,
    ))

    span_a = _selfweight_span(constant)
    span_b = _selfweight_span(thicker_mid)

    assert abs(np.min(span_b["def_y"])) <= abs(np.min(span_a["def_y"])) * 1.02
    # Engineering sign convention (v0.58): midspan sagging is the positive
    # moment peak, hogging at the frame corners is negative. A thicker
    # midspan attracts moment, so the sagging peak grows and the corner
    # hogging magnitude reduces.
    assert np.max(span_b["M"]) > np.max(span_a["M"])
    assert abs(np.min(span_b["M"])) < abs(np.min(span_a["M"]))


def test_one_span_frame_midspan_thickening_is_stiffer_with_shear_disabled():
    _assert_thicker_midspan_frame_is_not_more_flexible(use_shear_def=False)


def test_one_span_frame_midspan_thickening_is_stiffer_with_shear_enabled():
    _assert_thicker_midspan_frame_is_not_more_flexible(use_shear_def=True)


def _beam_params(span_geom):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam",
        "num_spans": 1,
        "L_list": [10.0] * 10,
        "Is_list": [0.5] * 10,
        "sw_list": [20.0] + [0.0] * 9,
        "E": 33e6,
        "E_span_list": [33e6] * 10,
        "mesh_size": 1.0,
        "step_size": 1.0,
        "vehicle": _empty_vehicle(),
        "vehicleB": _empty_vehicle(),
        "use_shear_def": False,
        "b_eff": 3.0,
        "phi_mode": "Manual",
        "phi": 1.0,
        "span_geom_0": span_geom,
    })
    return params


def test_simple_beam_midspan_thickening_is_not_more_flexible():
    constant = _selfweight_span(_run(_beam_params(_span_geom([0.5, 0.5, 0.5], 0))))
    thicker_mid = _selfweight_span(_run(_beam_params(_span_geom([0.5, 0.6, 0.5], 2))))

    assert abs(np.min(thicker_mid["def_y"])) <= abs(np.min(constant["def_y"])) * 1.02


def test_three_point_section_profile_uses_start_mid_end_convention():
    vals = np.array([0.5, 0.6, 0.5], dtype=np.float64)
    L = 10.0

    actual = [kernels.get_section_value_at_x(x * L, L, vals, 2) for x in [0.0, 0.25, 0.5, 0.75, 1.0]]

    np.testing.assert_allclose(actual, [0.5, 0.55, 0.6, 0.55, 0.5], rtol=0.0, atol=1e-12)


def test_non_prismatic_local_stiffness_matrix_is_well_formed():
    k_loc, *_ = kernels.jit_non_prismatic_matrices(
        0.0, 0.0, 10.0, 0.0,
        33e6, 0.0,
        2, 1, np.array([0.5, 0.6, 0.5], dtype=np.float64),
        3.0 * (0.5 + 4.0 * 0.6 + 0.5) / 6.0,
        3.0 * 0.5 * 5.0 / 6.0,
        3.0,
    )

    assert np.all(np.isfinite(k_loc))
    np.testing.assert_allclose(k_loc, k_loc.T, rtol=1e-10, atol=1e-8)

    bending = k_loc[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])]
    eigvals = np.linalg.eigvalsh(bending)
    assert eigvals[-1] > 0.0
    assert np.count_nonzero(eigvals > 1e-6) >= 2
