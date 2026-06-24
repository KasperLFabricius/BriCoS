"""User manual (in-app): the worked-example model, the content-block structure,
and the figure generators."""
import plotly.graph_objects as go

import bricos_data as data
import bricos_solver as solver
import bricos_manual as manual


def test_example_model_validates_and_solves():
    p = manual.example_model()
    assert data.validate_analysis_inputs(p, "Example") == []
    raw, nodes, _props, err = solver.run_raw_analysis(p)
    assert err == 0 and nodes is not None
    assert {'S1', 'S2', 'W1', 'W2', 'W3'} <= set(raw['Dead Load'].keys())


def test_manual_blocks_are_well_formed():
    blocks = manual.manual_blocks()
    assert len(blocks) > 30  # a substantial manual, not a stub
    valid = {'h1', 'h2', 'h3', 'md', 'callout', 'figure', 'table'}
    callouts = {'concept', 'theory', 'standard', 'tip', 'limit'}
    seen_types = set()
    for b in blocks:
        seen_types.add(b[0])
        assert b[0] in valid, f"unknown block type {b[0]!r}"
        if b[0] == 'figure':
            assert callable(b[1]) and isinstance(b[2], str)
        if b[0] == 'callout':
            assert b[1] in callouts
        if b[0] == 'table':
            headers, rows = b[1], b[2]
            assert headers and all(len(r) == len(headers) for r in rows)
    # the manual must actually use every structural element
    assert {'h1', 'md', 'callout', 'figure', 'table'} <= seen_types


def test_all_figure_generators_return_figures():
    # live (solve-backed) + schematic figures alike
    for gen in (manual.fig_structure, manual.fig_moment_envelope,
                manual.fig_deflection_envelope, manual.fig_soil_case,
                manual.fig_sign_convention, manual.fig_section_shapes,
                manual.fig_phi_curve):
        assert isinstance(gen(), go.Figure), gen.__name__


def test_every_figure_block_callable_builds():
    for b in manual.manual_blocks():
        if b[0] == 'figure':
            assert isinstance(b[1](), go.Figure)
