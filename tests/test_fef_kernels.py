import numpy as np
import pytest

import bricos_kernels as kernels


def _dense_simpson_trapezoid_fef(q_s, q_e, h_s, h_e, L, n=200001):
    """High-resolution Simpson reference for the Hermite-consistent FEF."""
    xs = np.linspace(h_s, h_e, n)
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    qx = q_s + (q_e - q_s) * (xs - h_s) / (h_e - h_s)
    xi = xs / L
    n_v1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
    n_m1 = xs * (1.0 - xi) ** 2
    n_v2 = 3.0 * xi**2 - 2.0 * xi**3
    n_m2 = xs * (xi**2 - xi)
    h = (h_e - h_s) / (n - 1)
    return [float(np.sum(w * qx * shape) * h / 3.0) for shape in (n_v1, n_m1, n_v2, n_m2)]


def test_fef_trapezoid_full_udl_matches_exact_fixed_end_values():
    L = 10.0
    q = 10.0

    f = kernels.jit_fef_trapezoid(q, q, 0.0, L, L)

    # Exact fixed-end values for a full-span UDL. Simpson integration is
    # exact here because the integrand is cubic.
    assert f[1] == pytest.approx(q * L / 2.0, rel=1e-9)
    assert f[4] == pytest.approx(q * L / 2.0, rel=1e-9)
    assert f[2] == pytest.approx(q * L**2 / 12.0, rel=1e-9)
    assert f[5] == pytest.approx(-q * L**2 / 12.0, rel=1e-9)


def test_fef_trapezoid_partial_triangular_load_matches_dense_reference():
    L = 10.0
    q_s, q_e, h_s, h_e = 0.0, 30.0, 2.0, 7.0

    f = kernels.jit_fef_trapezoid(q_s, q_e, h_s, h_e, L)
    ref = _dense_simpson_trapezoid_fef(q_s, q_e, h_s, h_e, L)

    # Composite Simpson with 21 points carries ~2e-6 relative error on the
    # quartic integrand (the previous trapezoid rule was ~3e-3).
    assert f[1] == pytest.approx(ref[0], rel=1e-5)
    assert f[2] == pytest.approx(ref[1], rel=1e-5)
    assert f[4] == pytest.approx(ref[2], rel=1e-5)
    assert f[5] == pytest.approx(ref[3], rel=1e-5)


def test_fef_trapezoid_vertical_equilibrium():
    L = 10.0
    q_s, q_e, h_s, h_e = 5.0, 25.0, 1.0, 8.5

    f = kernels.jit_fef_trapezoid(q_s, q_e, h_s, h_e, L)
    total_load = (q_s + q_e) / 2.0 * (h_e - h_s)

    assert f[1] + f[4] == pytest.approx(total_load, rel=1e-9)


def test_fef_axial_trapezoid_full_uniform_load_matches_exact():
    L = 10.0
    q = 7.0

    f = kernels.jit_fef_axial_trapezoid(q, q, 0.0, L, L)

    # Linear shape function x linear load is quadratic: Simpson is exact.
    assert f[0] == pytest.approx(q * L / 2.0, rel=1e-9)
    assert f[3] == pytest.approx(q * L / 2.0, rel=1e-9)


def test_fef_axial_trapezoid_partial_triangular_load_matches_exact():
    L = 10.0
    q_s, q_e, h_s, h_e = 0.0, 12.0, 2.0, 8.0

    f = kernels.jit_fef_axial_trapezoid(q_s, q_e, h_s, h_e, L)

    # Closed-form: F = integral q(x) dx, split by lever arms about each end.
    total = (q_s + q_e) / 2.0 * (h_e - h_s)
    # Centroid position of the partial trapezoid measured from x=0.
    x_c = h_s + (h_e - h_s) / 3.0 * (q_s + 2.0 * q_e) / (q_s + q_e)
    assert f[0] == pytest.approx(total * (1.0 - x_c / L), rel=1e-9)
    assert f[3] == pytest.approx(total * (x_c / L), rel=1e-9)
    assert f[0] + f[3] == pytest.approx(total, rel=1e-9)
