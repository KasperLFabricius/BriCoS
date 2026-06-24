"""v0.77: phi_summary / phi_base_values / phi_sls_from_base helpers.

These produce the dynamic-factor readout per analyzed limit state shared by
the sidebar and the PDF report.
"""
import pytest

import bricos_data as data

DASH = "–"  # en dash used by the range formatter


def _p(**kw):
    base = {'phi_mode': 'Manual', 'phi': 1.20, 'phi_sls_mode': 'Same', 'phi_sls': 1.05}
    base.update(kw)
    return base


def test_phi_base_values_manual_calculate_and_members():
    assert data.phi_base_values(_p(phi=1.18), {}) == [1.18]
    assert data.phi_base_values(_p(phi_mode='Calculate'), {'phi_calc': 1.142}) == [1.142]
    # Per-member values override and are de-duplicated + sorted.
    assert data.phi_base_values(_p(), {'Phi Members': {'S1': 1.25, 'S2': 1.10, 'W1': 1.25}}) == [1.10, 1.25]


def test_phi_base_values_are_unrounded():
    # The base value is NOT pre-rounded, so the SLS reduction is applied to the
    # same value the solver/report table use (the report formats at display).
    assert data.phi_base_values(_p(phi_mode='Calculate'), {'phi_calc': 1.234567}) == [1.234567]


def test_phi_summary_sls_reduces_unrounded_base():
    # Reduction on the unrounded base, then formatted - not rounded twice.
    phi = 1.234567
    s = data.phi_summary(_p(phi_mode='Calculate', phi_sls_mode='Reduced'), {'phi_calc': phi}, True, True)
    assert s['uls'] == f"{phi:.3f}"
    assert s['sls'] == f"{1.0 + (phi - 1.0) / 2.0:.3f}"


def test_phi_sls_from_base_modes():
    assert data.phi_sls_from_base(1.30, _p(phi_sls_mode='Same')) == pytest.approx(1.30)
    assert data.phi_sls_from_base(1.30, _p(phi_sls_mode='Reduced')) == pytest.approx(1.15)
    assert data.phi_sls_from_base(1.30, _p(phi_sls_mode='Manual', phi_sls=1.07)) == pytest.approx(1.07)


def test_phi_summary_both_limit_states():
    assert data.phi_summary(_p(phi=1.20, phi_sls_mode='Same'), {}, True, True) == {'uls': '1.200', 'sls': '1.200'}
    red = data.phi_summary(_p(phi=1.20, phi_sls_mode='Reduced'), {}, True, True)
    assert red['uls'] == '1.200' and red['sls'] == '1.100'
    man = data.phi_summary(_p(phi=1.20, phi_sls_mode='Manual', phi_sls=1.08), {}, True, True)
    assert man['uls'] == '1.200' and man['sls'] == '1.080'


def test_phi_summary_gates_per_limit_state():
    # ULS only -> no SLS entry.
    s = data.phi_summary(_p(phi=1.20), {}, True, False)
    assert s['uls'] == '1.200' and s['sls'] is None
    # SLS only, no reduction -> the base phi is the SLS phi; no ULS entry.
    s2 = data.phi_summary(_p(phi=1.20, phi_sls_mode='Same'), {}, False, True)
    assert s2['uls'] is None and s2['sls'] == '1.200'
    # SLS only, reduced.
    s3 = data.phi_summary(_p(phi=1.20, phi_sls_mode='Reduced'), {}, False, True)
    assert s3['uls'] is None and s3['sls'] == '1.100'


def test_phi_summary_per_member_ranges():
    raw = {'Phi Members': {'S1': 1.25, 'S2': 1.10}}
    s = data.phi_summary(_p(phi_mode='Calculate', phi_sls_mode='Reduced'), raw, True, True)
    assert s['uls'] == f"1.100{DASH}1.250"
    # Reduced per member: 1+(1.10-1)/2 = 1.05 ; 1+(1.25-1)/2 = 1.125
    assert s['sls'] == f"1.050{DASH}1.125"
