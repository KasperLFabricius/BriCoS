import numpy as np

import bricos_kernels as kernels


def test_constant_section_non_prismatic_stiffness_matches_prismatic():
    E = 30e6
    L = 10.0
    b_eff = 1.0
    h = 1.0
    area = 1.0
    inertia = 1.0 / 12.0

    k_prism, *_ = kernels.jit_beam_matrices(
        0.0, 0.0, L, 0.0, E, inertia, area, 0.0
    )
    k_nonprism, *_ = kernels.jit_non_prismatic_matrices(
        0.0, 0.0, L, 0.0,
        E, 0.0,
        0, 1, np.array([h, h, h], dtype=np.float64),
        area, area, b_eff,
    )

    np.testing.assert_allclose(k_nonprism, k_prism, rtol=1e-5, atol=1e-5)


def test_constant_section_numerical_fef_matches_point_load_kernel():
    E = 30e6
    L = 10.0
    b_eff = 1.0
    h = 1.0
    area = 1.0
    geom_vals = np.array([h, h, h], dtype=np.float64)

    numerical = kernels.jit_numerical_fef(
        1,
        np.array([100.0, L / 2.0], dtype=np.float64),
        L, E, 0.0,
        0, 1, geom_vals, b_eff, area,
    )
    analytical = kernels.jit_fef_point(100.0, L / 2.0, L)

    np.testing.assert_allclose(numerical, analytical, rtol=2e-2, atol=1e-5)


def test_constant_section_numerical_fef_matches_trapezoid_kernel():
    E = 30e6
    L = 10.0
    b_eff = 1.0
    h = 1.0
    area = 1.0
    geom_vals = np.array([h, h, h], dtype=np.float64)

    numerical = kernels.jit_numerical_fef(
        2,
        np.array([10.0, 10.0, 0.0, L], dtype=np.float64),
        L, E, 0.0,
        0, 1, geom_vals, b_eff, area,
    )
    analytical = kernels.jit_fef_trapezoid(10.0, 10.0, 0.0, L)

    np.testing.assert_allclose(numerical, analytical, rtol=3e-3, atol=1e-5)
