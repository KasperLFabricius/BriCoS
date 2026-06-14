"""Validation campaign (v0.62): closed-form and independent-reference
checks across static systems, support conditions, section variation,
inclination, shear deformation, moving loads and material scaling.

Conventions under test (engineering convention, v0.58): M sagging
positive, V classical, N tension positive, deflections in global axes
(downward negative). Simple-input section values are HEIGHTS (type 1):
I = b_eff * H^3 / 12, A = b_eff * H.
"""
import math

import numpy as np
import pytest

import bricos_data as data
import bricos_solver as solver

E_MOD = 30e6  # kN/m^2
H_SEC = 1.0   # m -> I = 1/12 m^4 with b_eff = 1.0
I_SEC = H_SEC ** 3 / 12.0
L = 10.0
Q = 10.0  # kN/m

FIXED = {"type": "Fixed", "k": [1e14, 1e14, 1e14]}
PINNED = {"type": "Pinned", "k": [1e14, 1e14, 0.0]}
ROLLER = {"type": "Roller X", "k": [0.0, 1e14, 0.0]}


def _beam(n_spans=1, q=Q, H=H_SEC, supports=None, **overrides):
    params = data.get_clear("A", "Beam")
    params.update({
        "mode": "Beam",
        "num_spans": n_spans,
        "L_list": [L] * 10,
        "Is_list": [H] * 10,
        "sw_list": [q] * n_spans + [0.0] * (10 - n_spans),
        "E": E_MOD,
        "E_span_list": [E_MOD] * 10,
        "mesh_size": 0.5,
        "step_size": 0.5,
        "vehicle": {"loads": [], "spacing": []},
        "vehicleB": {"loads": [], "spacing": []},
        "vehicle_direction": "Forward",
        "use_shear_def": False,
        "b_eff": 1.0,
        "phi_mode": "Manual",
        "phi": 1.0,
    })
    if supports is not None:
        params["supports"] = [dict(s) for s in supports]
    params.update(overrides)
    return params


def _run(params):
    solver.clear_solver_cache()
    raw, nodes, props, err = solver.run_raw_analysis(params)
    assert err == 0
    solver.clear_solver_cache()
    return raw, nodes, props


def _static_case(params, case="Dead Load"):
    raw, _, _ = _run(params)
    res = solver.combine_results(raw, params, "Unfactored")
    return raw, res[case]


# ==========================================
# 1. PRISMATIC CLOSED FORMS (POINTWISE)
# ==========================================

def test_simply_supported_udl_pointwise_fields():
    raw, sw = _static_case(_beam())
    s1 = sw["S1"]
    x = np.asarray(s1["x"])

    np.testing.assert_allclose(s1["M"], Q * x * (L - x) / 2.0, atol=1e-6)
    np.testing.assert_allclose(s1["V"], Q * (L / 2.0 - x), atol=1e-6)
    v_exact = -Q * x * (L**3 - 2.0 * L * x**2 + x**3) / (24.0 * E_MOD * I_SEC)
    np.testing.assert_allclose(s1["def_y"], v_exact, atol=1e-9)
    np.testing.assert_allclose(s1["N"], 0.0, atol=1e-6)

    assert raw["Reactions"][200][1] == pytest.approx(Q * L / 2.0, rel=1e-9)
    assert raw["Reactions"][201][1] == pytest.approx(Q * L / 2.0, rel=1e-9)


def test_fixed_fixed_udl_pointwise_fields():
    _, sw = _static_case(_beam(supports=[FIXED, FIXED]))
    s1 = sw["S1"]
    x = np.asarray(s1["x"])

    m_exact = Q * (6.0 * L * x - L**2 - 6.0 * x**2) / 12.0
    np.testing.assert_allclose(s1["M"], m_exact, atol=1e-4)
    assert s1["M"][0] == pytest.approx(-Q * L**2 / 12.0, rel=1e-6)
    mid = np.isclose(x, L / 2.0)
    assert np.max(s1["M"][mid]) == pytest.approx(Q * L**2 / 24.0, rel=1e-6)
    assert np.min(s1["def_y"]) == pytest.approx(
        -Q * L**4 / (384.0 * E_MOD * I_SEC), rel=1e-6)


def test_propped_cantilever_udl_closed_form():
    raw, sw = _static_case(_beam(supports=[FIXED, ROLLER]))
    s1 = sw["S1"]
    x = np.asarray(s1["x"])

    # R_prop = 3wL/8; M_fixed = -wL^2/8; max sagging 9wL^2/128 at 5L/8.
    assert raw["Reactions"][201][1] == pytest.approx(3.0 * Q * L / 8.0, rel=1e-6)
    assert s1["M"][0] == pytest.approx(-Q * L**2 / 8.0, rel=1e-6)
    m_exact = -Q * L**2 / 8.0 + 5.0 * Q * L * x / 8.0 - Q * x**2 / 2.0
    np.testing.assert_allclose(s1["M"], m_exact, atol=1e-4)
    at_58 = np.isclose(x, 5.0 * L / 8.0)
    if at_58.any():
        assert np.max(s1["M"][at_58]) == pytest.approx(
            9.0 * Q * L**2 / 128.0, rel=1e-6)


def test_three_span_interior_support_moments():
    # Three equal spans, all loaded: M_B = M_C = -wL^2/10.
    _, sw = _static_case(_beam(n_spans=3))
    m_b = sw["S1"]["M"][-1]
    m_c = sw["S3"]["M"][0]
    assert m_b == pytest.approx(-Q * L**2 / 10.0, rel=1e-6)
    assert m_c == pytest.approx(-Q * L**2 / 10.0, rel=1e-6)
    # Continuity across the support.
    assert sw["S2"]["M"][0] == pytest.approx(m_b, rel=1e-9)


def test_spring_supports_add_rigid_settlement_to_deflection():
    k = 1.0e4  # kN/m
    soft = [{"type": "Custom Spring", "k": [1e14, k, 0.0]},
            {"type": "Custom Spring", "k": [0.0, k, 0.0]}]
    _, sw = _static_case(_beam(supports=soft))
    s1 = sw["S1"]
    x = np.asarray(s1["x"])

    settlement = (Q * L / 2.0) / k
    v_beam = Q * L**4 * 5.0 / 384.0 / (E_MOD * I_SEC)
    mid = np.isclose(x, L / 2.0)
    assert np.min(s1["def_y"][mid]) == pytest.approx(
        -(v_beam + settlement), rel=1e-6)
    # Statically determinate: moments unchanged by the springs.
    np.testing.assert_allclose(s1["M"], Q * x * (L - x) / 2.0, atol=1e-4)


# ==========================================
# 2. TIMOSHENKO SHEAR DEFORMATION
# ==========================================

def test_timoshenko_adds_exact_shear_deflection_term():
    nu = 0.2
    p_eb = _beam(use_shear_def=False)
    p_tim = _beam(use_shear_def=True, nu=nu)

    _, sw_eb = _static_case(p_eb)
    _, sw_tim = _static_case(p_tim)

    x = np.asarray(sw_eb["S1"]["x"])
    mid = np.isclose(x, L / 2.0)
    d_eb = np.min(sw_eb["S1"]["def_y"][mid])
    d_tim = np.min(sw_tim["S1"]["def_y"][mid])

    g_mod = E_MOD / (2.0 * (1.0 + nu))
    a_s = (5.0 / 6.0) * 1.0 * H_SEC
    shear_term = Q * L**2 / (8.0 * g_mod * a_s)
    assert (d_eb - d_tim) == pytest.approx(shear_term, rel=1e-4)
    # Determinate system: bending unchanged.
    np.testing.assert_allclose(sw_tim["S1"]["M"], sw_eb["S1"]["M"], atol=1e-6)


# ==========================================
# 3. MOVING LOADS vs INFLUENCE-LINE REFERENCE
# ==========================================

def test_single_axle_envelope_matches_influence_lines():
    p_kn = 100.0
    params = _beam(q=0.0, vehicle={"loads": [p_kn / 9.81], "spacing": [0.0]})
    raw, _, _ = _run(params)
    res = solver.combine_results(raw, params, "Unfactored")
    env = res["Vehicle Envelope"]["S1"]
    x = np.asarray(env["x"])

    # The influence-line maximum M(x) = P x (L-x)/L requires the load AT
    # the section; with discrete 0.5 m stepping that is only reachable at
    # sections on the step grid. Between them the envelope must stay
    # BELOW the continuum influence line (the discrete optimum has the
    # load at the nearest step position).
    inner = (x > 1e-9) & (x < L - 1e-9)
    on_grid = inner & np.isclose(np.round(x * 2.0), x * 2.0, atol=1e-9)
    np.testing.assert_allclose(
        env["M_max"][on_grid],
        p_kn * x[on_grid] * (L - x[on_grid]) / L, atol=1e-6)
    assert np.all(env["M_max"] <= p_kn * x * (L - x) / L + 1e-6)

    # Shear: both one-sided values at a loaded section are exact - the
    # unloaded side from the left sub-element's end entry, the loaded side
    # (since the v0.63 boundary-jump fix) from the right sub-element's
    # start entry.
    for xv in np.unique(x[on_grid]):
        at = np.isclose(x, xv)
        assert np.max(env["V_max"][at]) == pytest.approx(
            p_kn * (L - xv) / L, rel=1e-9)
        assert np.min(env["V_min"][at]) == pytest.approx(
            -p_kn * xv / L, rel=1e-9)

    # Peak deflection: PL^3/48EI at midspan with the load there.
    assert np.min(env["def_y_min"]) == pytest.approx(
        -p_kn * L**3 / (48.0 * E_MOD * I_SEC), rel=1e-6)


def test_single_axle_shear_envelope_carries_loaded_side_value():
    # v0.63 fix of the v0.62 validation finding: a point load placed
    # exactly on a result section now contributes its shear jump AT the
    # section (loaded-side value on the owning sub-element's start entry),
    # so the V envelope no longer misses it by one step.
    p_kn = 100.0
    params = _beam(q=0.0, vehicle={"loads": [p_kn / 9.81], "spacing": [0.0]})
    raw, _, _ = _run(params)
    res = solver.combine_results(raw, params, "Unfactored")
    env = res["Vehicle Envelope"]["S1"]
    x = np.asarray(env["x"])

    mid = np.isclose(x, L / 2.0)
    assert np.min(env["V_min"][mid]) == pytest.approx(-p_kn / 2.0, rel=1e-9)


def test_step_arrays_carry_both_sides_of_the_jump_at_shared_sections():
    # With the axle on an interior mesh boundary, the two result entries
    # at that section carry V(x-) and V(x+): the full jump is visible AT
    # the load for the step viewer, report plots and tabular export.
    p_kn = 100.0
    params = _beam(q=0.0, vehicle={"loads": [p_kn / 9.81], "spacing": [0.0]})
    raw, _, _ = _run(params)
    step = next(s for s in raw["Vehicle Steps A"] if abs(s["x"] - L / 2.0) < 1e-9)
    r = step["res"]["S1"]
    x = np.asarray(r["x"])
    at = np.isclose(x, L / 2.0)

    assert np.max(r["V"][at]) == pytest.approx(p_kn / 2.0, rel=1e-9)   # V(x-)
    assert np.min(r["V"][at]) == pytest.approx(-p_kn / 2.0, rel=1e-9)  # V(x+)
    # M is continuous: both entries agree.
    assert np.ptp(r["M"][at]) == pytest.approx(0.0, abs=1e-6)


def test_axle_over_support_keeps_member_end_shear_semantics():
    # Member ENDS keep the historical convention: an axle directly over a
    # support registers in the adjoining member-start shear (the reaction
    # envelopes are built from member end forces), and the rest of the
    # structure stays unloaded.
    p_kn = 100.0
    params = _beam(n_spans=2, q=0.0,
                   vehicle={"loads": [p_kn / 9.81], "spacing": [0.0]})
    raw, _, _ = _run(params)

    step = next(s for s in raw["Vehicle Steps A"] if abs(s["x"] - L) < 1e-9)
    s1, s2 = step["res"]["S1"], step["res"]["S2"]
    # Load goes straight into the interior support: no bending anywhere,
    # S2's start entry carries the reaction-bearing end shear.
    assert np.max(np.abs(s1["M"])) == pytest.approx(0.0, abs=1e-6)
    assert np.max(np.abs(s2["M"])) == pytest.approx(0.0, abs=1e-6)
    assert s2["V"][0] == pytest.approx(p_kn, rel=1e-9)
    assert np.max(np.abs(np.asarray(s2["V"])[1:])) == pytest.approx(0.0, abs=1e-6)
    assert np.max(np.abs(s1["V"])) == pytest.approx(0.0, abs=1e-6)


def test_multi_axle_envelope_matches_brute_force_statics():
    # Independent reference: per recorded step, place the axles by simple
    # statics on the simply supported span and envelope the moments.
    # Axle j sits at step_x - cumdist[j]: the FIRST axle in the list is
    # the lead axle.
    axles_kn = np.array([80.0, 120.0])
    spacing = np.array([0.0, 2.0])
    params = _beam(q=0.0, vehicle={
        "loads": list(axles_kn / 9.81), "spacing": list(spacing)})
    raw, _, _ = _run(params)
    res = solver.combine_results(raw, params, "Unfactored")
    env = res["Vehicle Envelope"]["S1"]
    x = np.asarray(env["x"])

    cum = np.cumsum(spacing)

    def moment_curve(lead_x):
        m = np.zeros_like(x)
        for d, p in zip(cum, axles_kn):
            pos = lead_x - d
            if pos < -1e-9 or pos > L + 1e-9:
                continue
            r_left = p * (L - pos) / L
            m += np.where(x <= pos, r_left * x, p * (L - x) * pos / L)
        return m

    m_ref = np.zeros_like(x)
    for step in raw["Vehicle Steps A"]:
        m_ref = np.maximum(m_ref, moment_curve(step["x"]))
    np.testing.assert_allclose(env["M_max"], m_ref, atol=1e-6)

    # The discrete peak must bracket the continuum optimum within the
    # resolution of the 0.5 m stepping.
    best = max(float(np.max(moment_curve(lead)))
               for lead in np.arange(0.0, L + cum[-1] + 0.01, 0.01))
    assert np.max(env["M_max"]) <= best + 1e-6
    assert np.max(env["M_max"]) >= 0.99 * best


def test_finer_stepping_tightens_envelope_and_keeps_peak():
    p_kn = 100.0
    veh = {"loads": [p_kn / 9.81], "spacing": [0.0]}
    p_coarse = _beam(q=0.0, vehicle=veh, step_size=0.5)
    p_fine = _beam(q=0.0, vehicle=veh, step_size=0.1)

    raw_c, _, _ = _run(p_coarse)
    raw_f, _, _ = _run(p_fine)
    env_c = solver.combine_results(raw_c, p_coarse, "Unfactored")["Vehicle Envelope"]["S1"]
    env_f = solver.combine_results(raw_f, p_fine, "Unfactored")["Vehicle Envelope"]["S1"]

    # Finer stepping can only widen (or keep) the envelope, never shrink it.
    assert np.all(env_f["M_max"] >= np.asarray(env_c["M_max"]) - 1e-9)
    assert np.all(env_f["V_min"] <= np.asarray(env_c["V_min"]) + 1e-9)
    # The peak (load at midspan, on both grids) is unchanged: PL/4.
    assert np.max(env_c["M_max"]) == pytest.approx(p_kn * L / 4.0, rel=1e-9)
    assert np.max(env_f["M_max"]) == pytest.approx(p_kn * L / 4.0, rel=1e-9)


def test_tapered_span_moving_load_still_matches_statics():
    # Statically determinate: the moving-load moment envelope is pure
    # statics regardless of the taper, exercising the non-prismatic batch
    # step recovery against an independent reference.
    p_kn = 100.0
    params = _beam(q=0.0, vehicle={"loads": [p_kn / 9.81], "spacing": [0.0]},
                   span_geom_0={"type": 1, "shape": 1, "vals": [1.0, 0.6, 0.6]})
    raw, _, _ = _run(params)
    env = solver.combine_results(raw, params, "Unfactored")["Vehicle Envelope"]["S1"]
    x = np.asarray(env["x"])

    inner = (x > 1e-9) & (x < L - 1e-9)
    on_grid = inner & np.isclose(np.round(x * 2.0), x * 2.0, atol=1e-9)
    np.testing.assert_allclose(
        env["M_max"][on_grid],
        p_kn * x[on_grid] * (L - x[on_grid]) / L, atol=1e-6)


def test_unequal_spans_interior_support_moment_closed_form():
    # Two spans L1=10, L2=6, same EI, UDL on both:
    # M_B = -w (L1^3 + L2^3) / (8 (L1 + L2)).
    l2 = 6.0
    params = _beam(n_spans=2)
    params["L_list"] = [L, l2] + [0.0] * 8
    _, sw = _static_case(params)

    m_b = -Q * (L**3 + l2**3) / (8.0 * (L + l2))
    assert sw["S1"]["M"][-1] == pytest.approx(m_b, rel=1e-6)
    assert sw["S2"]["M"][0] == pytest.approx(m_b, rel=1e-6)


def test_reverse_direction_mirrors_envelope_on_symmetric_system():
    veh = {"loads": [80.0 / 9.81, 120.0 / 9.81], "spacing": [0.0, 2.0]}
    p_fwd = _beam(q=0.0, vehicle=veh, vehicle_direction="Forward")
    p_rev = _beam(q=0.0, vehicle=veh, vehicle_direction="Reverse")

    raw_f, _, _ = _run(p_fwd)
    raw_r, _, _ = _run(p_rev)
    env_f = solver.combine_results(raw_f, p_fwd, "Unfactored")["Vehicle Envelope"]["S1"]
    env_r = solver.combine_results(raw_r, p_rev, "Unfactored")["Vehicle Envelope"]["S1"]

    assert np.max(env_r["M_max"]) == pytest.approx(np.max(env_f["M_max"]), rel=1e-9)
    x = np.asarray(env_f["x"])
    xu = np.unique(x)
    f_interp = np.interp(xu, x, env_f["M_max"])
    r_interp = np.interp(xu, x, env_r["M_max"])
    np.testing.assert_allclose(r_interp, f_interp[::-1], atol=1e-6)


# ==========================================
# 4. SECTION VARIATION (TAPER / 3-POINT)
# ==========================================

def test_taper_with_equal_ends_matches_prismatic():
    p_const = _beam()
    p_lin = _beam(span_geom_0={"type": 1, "shape": 1, "vals": [H_SEC, H_SEC, H_SEC]})
    p_3pt = _beam(span_geom_0={"type": 1, "shape": 2, "vals": [H_SEC, H_SEC, H_SEC]})

    _, sw_c = _static_case(p_const)
    _, sw_l = _static_case(p_lin)
    _, sw_3 = _static_case(p_3pt)
    for key in ("M", "V", "def_y"):
        np.testing.assert_allclose(sw_l["S1"][key], sw_c["S1"][key], atol=1e-7)
        np.testing.assert_allclose(sw_3["S1"][key], sw_c["S1"][key], atol=1e-7)


def test_symmetric_taper_keeps_determinate_moments_and_bounds_deflection():
    h_end, h_mid = 0.8, 0.5
    p_taper = _beam(span_geom_0={"type": 1, "shape": 2, "vals": [h_end, h_mid, h_end]})
    _, sw_t = _static_case(p_taper)
    s1 = sw_t["S1"]
    x = np.asarray(s1["x"])

    # Statically determinate: M(x) independent of the stiffness profile.
    np.testing.assert_allclose(s1["M"], Q * x * (L - x) / 2.0, rtol=1e-6, atol=1e-6)
    # Symmetry of the response.
    xu = np.unique(x)
    d_interp = np.interp(xu, x, s1["def_y"])
    np.testing.assert_allclose(d_interp, d_interp[::-1], atol=1e-9)

    # Deflection bracketed by the uniform sections.
    _, sw_stiff = _static_case(_beam(H=h_end))
    _, sw_soft = _static_case(_beam(H=h_mid))
    d_t = abs(np.min(s1["def_y"]))
    assert abs(np.min(sw_stiff["S1"]["def_y"])) < d_t < abs(np.min(sw_soft["S1"]["def_y"]))


def test_tapered_deflection_matches_virtual_work_integral():
    # Linear taper H 1.0 -> 0.6, SS UDL. Independent reference: unit-load
    # virtual work delta(x0) = int M(x) m(x) / (E I(x)) dx on a dense grid.
    h0, h1 = 1.0, 0.6
    params = _beam(mesh_size=0.25,
                   span_geom_0={"type": 1, "shape": 1, "vals": [h0, h1, h1]})
    _, sw = _static_case(params)
    s1 = sw["S1"]
    x = np.asarray(s1["x"])
    x0 = L / 2.0

    xs = np.linspace(0.0, L, 20001)
    h_x = h0 + (h1 - h0) * xs / L
    i_x = h_x**3 / 12.0
    m_x = Q * xs * (L - xs) / 2.0
    m_unit = np.where(xs <= x0, xs * (L - x0) / L, x0 * (L - xs) / L)
    delta = np.trapezoid(m_x * m_unit / (E_MOD * i_x), xs)

    mid = np.isclose(x, x0)
    assert np.min(s1["def_y"][mid]) == pytest.approx(-delta, rel=5e-3)
    # Moments stay the determinate closed form regardless of taper.
    np.testing.assert_allclose(s1["M"], Q * x * (L - x) / 2.0, rtol=1e-6, atol=1e-6)


# ==========================================
# 5. INCLINED SPANS
# ==========================================

def test_inclined_ss_span_bending_and_reactions_closed_form():
    rise = 2.0
    params = _beam(span_geom_0={
        "type": 1, "shape": 0, "vals": [H_SEC] * 3,
        "align_type": 1, "incline_mode": 1, "incline_val": rise,
    })
    raw, sw = _static_case(params)
    s1 = sw["S1"]

    l_m = math.sqrt(L**2 + rise**2)
    cos_t = L / l_m
    # Gravity q per member length; transverse component bends the member:
    # M_max = (q cos(theta)) * L_m^2 / 8 at mid-member.
    assert np.max(s1["M"]) == pytest.approx(Q * cos_t * l_m**2 / 8.0, rel=1e-6)
    # Total vertical reaction = full weight.
    total_ry = raw["Reactions"][200][1] + raw["Reactions"][201][1]
    assert total_ry == pytest.approx(Q * l_m, rel=1e-9)
    # Axial force present and bounded by the tangential resultant.
    assert np.max(np.abs(s1["N"])) > 1e-3
    assert np.max(np.abs(s1["N"])) <= Q * l_m + 1e-6


# ==========================================
# 6. FRAMES: LIMIT AND SYMMETRY CHECKS
# ==========================================

def _portal(h_wall=5.0, wall_H=0.5, deck_H=H_SEC, q=Q, **overrides):
    params = data.get_clear("A", "Frame")
    params.update({
        "mode": "Frame",
        "num_spans": 1,
        "L_list": [L] * 10,
        "Is_list": [deck_H] * 10,
        "sw_list": [q] + [0.0] * 9,
        "h_list": [h_wall, h_wall] + [0.0] * 9,
        "Iw_list": [wall_H] * 11,
        "E": E_MOD,
        "E_span_list": [E_MOD] * 10,
        "E_wall_list": [E_MOD] * 11,
        "mesh_size": 0.5,
        "step_size": 0.5,
        "vehicle": {"loads": [], "spacing": []},
        "vehicleB": {"loads": [], "spacing": []},
        "vehicle_direction": "Forward",
        "use_shear_def": False,
        "b_eff": 1.0,
        "phi_mode": "Manual",
        "phi": 1.0,
        "soil": [],
        "surcharge": [],
    })
    params.update(overrides)
    return params


def test_portal_symmetry_under_deck_udl():
    _, sw = _static_case(_portal())
    s1 = sw["S1"]
    x = np.asarray(s1["x"])

    # Symmetric portal: corner moments equal, deck moment symmetric,
    # column axial = wL/2 compression each.
    assert s1["M"][0] == pytest.approx(s1["M"][-1], rel=1e-6)
    xu = np.unique(x)
    m_interp = np.interp(xu, x, s1["M"])
    np.testing.assert_allclose(m_interp, m_interp[::-1], atol=1e-6)
    for wid in ("W1", "W2"):
        n_wall = sw[wid]["N"]
        assert np.min(n_wall) == pytest.approx(-Q * L / 2.0, rel=1e-6)
    # Static deck moment relation: M_mid - M_end = wL^2/8 exactly.
    mid = np.isclose(x, L / 2.0)
    sagging_rel = np.max(s1["M"][mid]) - s1["M"][0]
    assert sagging_rel == pytest.approx(Q * L**2 / 8.0, rel=1e-9)


def test_portal_stiff_columns_approach_fixed_fixed_deck():
    _, sw = _static_case(_portal(wall_H=5.0))  # I_wall = 1000 x deck
    s1 = sw["S1"]
    assert s1["M"][0] == pytest.approx(-Q * L**2 / 12.0, rel=2e-2)
    x = np.asarray(s1["x"])
    mid = np.isclose(x, L / 2.0)
    assert np.max(s1["M"][mid]) == pytest.approx(Q * L**2 / 24.0, rel=5e-2)


def test_portal_flexible_columns_approach_simply_supported_deck():
    _, sw = _static_case(_portal(wall_H=0.05))  # I_wall = 1/8000 x deck
    s1 = sw["S1"]
    x = np.asarray(s1["x"])
    mid = np.isclose(x, L / 2.0)
    assert abs(s1["M"][0]) < 0.02 * Q * L**2 / 8.0
    assert np.max(s1["M"][mid]) == pytest.approx(Q * L**2 / 8.0, rel=2e-2)


def test_soil_and_surcharge_wall_against_elastic_prop_closed_form():
    # Soil (triangular, max at base) and surcharge (uniform) on wall W1 of
    # a portal whose deck is an axial strut (negligible bending): the far
    # wall is an ELASTIC horizontal prop with exactly the loaded wall's
    # own tip stiffness 3EI/h^3, so the prop carries HALF the rigid-prop
    # force. Cantilever tip deflections give the rigid-prop forces
    # R_rigid = q_b h/10 (triangle, max at base) and 3W/8 (uniform);
    # halving them:
    #   soil:      R_top = W/5 / 2 = W/10,  M_base = -(Wh/3 - R h) = -7Wh/30
    #   surcharge: R_top = 3W/16,           M_base = -(Wh/2 - R h) = -5Wh/16
    h, q_b, q_s = 8.0, 20.0, 5.0
    p_soil = _portal(h_wall=h, deck_H=0.05, q=0.0,
                     soil=[{"wall_idx": 0, "face": "L", "h": h,
                            "q_top": 0.0, "q_bot": q_b}])
    _, res_soil = _static_case(p_soil, case="Soil")
    w_tri = q_b * h / 2.0
    w1 = res_soil["W1"]
    assert w1["M"][0] == pytest.approx(-7.0 * w_tri * h / 30.0, rel=1e-2)
    assert np.max(np.abs(w1["V"])) == pytest.approx(0.9 * w_tri, rel=1e-2)
    assert abs(w1["M"][-1]) < 0.02 * abs(w1["M"][0])  # deck bending ~ none

    p_sur = _portal(h_wall=h, deck_H=0.05, q=0.0,
                    surcharge=[{"wall_idx": 0, "face": "R", "q": q_s, "h": h}])
    _, res_sur = _static_case(p_sur, case="Surcharge")
    w_uni = q_s * h
    w1s = res_sur["W1"]
    assert abs(w1s["M"][0]) == pytest.approx(5.0 * w_uni * h / 16.0, rel=1e-2)
    assert np.max(np.abs(w1s["V"])) == pytest.approx(13.0 * w_uni / 16.0, rel=1e-2)


# ==========================================
# 7. MATERIAL / SECTION SCALING
# ==========================================

def test_deflection_scales_inversely_with_E_and_b_eff():
    _, sw_base = _static_case(_beam())
    base = np.asarray(sw_base["S1"]["def_y"])

    p_2e = _beam(E=2 * E_MOD, E_span_list=[2 * E_MOD] * 10)
    _, sw_2e = _static_case(p_2e)
    np.testing.assert_allclose(sw_2e["S1"]["def_y"], base / 2.0, atol=1e-12)
    np.testing.assert_allclose(sw_2e["S1"]["M"], sw_base["S1"]["M"], atol=1e-6)

    p_2b = _beam(b_eff=2.0)
    _, sw_2b = _static_case(p_2b)
    np.testing.assert_allclose(sw_2b["S1"]["def_y"], base / 2.0, atol=1e-12)
    np.testing.assert_allclose(sw_2b["S1"]["M"], sw_base["S1"]["M"], atol=1e-6)


# ==========================================
# 8. PARTIAL FACTORS END TO END
# ==========================================

def test_uls_partial_factors_scale_components_linearly():
    veh = {"loads": [100.0 / 9.81], "spacing": [0.0]}
    params = _beam(vehicle=veh, KFI=1.1, gamma_g=1.35, gamma_veh=1.4,
                   phi=1.25, gamma_udl=0.56)
    raw, _, _ = _run(params)
    uls = solver.combine_results(raw, params, "Design (ULS)")
    unf = solver.combine_results(raw, params, "Unfactored")

    np.testing.assert_allclose(
        uls["Dead Load"]["S1"]["M_max"],
        np.asarray(unf["Dead Load"]["S1"]["M_max"]) * 1.1 * 1.35, atol=1e-9)
    np.testing.assert_allclose(
        uls["Vehicle Envelope"]["S1"]["M_max"],
        np.asarray(unf["Vehicle Envelope"]["S1"]["M_max"]) * 1.1 * 1.4 * 1.25,
        atol=1e-9)


def test_manual_phi_multiplies_vehicle_term_only():
    veh = {"loads": [100.0 / 9.81], "spacing": [0.0]}
    p1 = _beam(vehicle=veh, phi=1.0)
    p2 = _beam(vehicle=veh, phi=1.3)
    raw1, _, _ = _run(p1)
    raw2, _, _ = _run(p2)
    r1 = solver.combine_results(raw1, p1, "Design (ULS)")
    r2 = solver.combine_results(raw2, p2, "Design (ULS)")

    np.testing.assert_allclose(
        r2["Vehicle Envelope"]["S1"]["M_max"],
        np.asarray(r1["Vehicle Envelope"]["S1"]["M_max"]) * 1.3, atol=1e-9)
    np.testing.assert_allclose(
        r2["Dead Load"]["S1"]["M_max"], r1["Dead Load"]["S1"]["M_max"],
        atol=1e-9)
