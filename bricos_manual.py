"""BriCoS user manual.

Single source of truth for the manual content. The content is authored as a
list of structured blocks (headings, markdown, callouts, figures, tables) so it
can be rendered both in the app (render_manual_streamlit, below) and - in a
later step - to a downloadable PDF over the same blocks.

Figures are generated live from one worked example (a 2-span integral frame
bridge) using the same plotting code the app uses, so they always match the
current version. Schematic figures (sign convention, load types, section
shapes, dynamic factor) are drawn with Plotly so they render identically in the
app and, later, in the PDF.

Source is kept ASCII-only: emoji are Streamlit shortcodes (e.g. :bulb:) and
mathematics is LaTeX ($...$), never literal non-ASCII glyphs.
"""
import io
import re
import threading

import numpy as np
import plotly.graph_objects as go

import bricos_data as data_mod
import bricos_solver as solver
import bricos_viz as viz

try:
    import streamlit as st
except Exception:  # pragma: no cover - manual content is importable without Streamlit
    st = None


# ==========================================================================
# WORKED EXAMPLE MODEL
# ==========================================================================

def example_model():
    """The worked example threaded through the manual: a 2-span integral
    concrete frame bridge - 2 x 12 m spans on 6 m walls, backfill (soil) on the
    outer walls, a standard classification vehicle and a Traffic UDL. It
    exercises Frame mode, walls, soil, a moving vehicle, the UDL, ULS/SLS and
    the dynamic factor - i.e. nearly every feature."""
    p = data_mod.get_def()
    p.update({
        'mode': 'Frame', 'num_spans': 2,
        'L_list': [12.0] * 10, 'h_list': [6.0] * 11,
        'Is_list': [0.8] * 10, 'Iw_list': [0.6] * 11,
        'b_eff': 6.0, 'udl_q': 2.5, 'mesh_size': 0.5, 'step_size': 0.5,
        'analyze_uls': True, 'analyze_sls': True,
        'soil': [
            {'wall_idx': 0, 'face': 'L', 'h': 6.0, 'q_top': 0.0, 'q_bot': 30.0},
            {'wall_idx': 2, 'face': 'R', 'h': 6.0, 'q_top': 0.0, 'q_bot': 30.0},
        ],
        'name': 'Example Bridge',
    })
    return data_mod.sanitize_input_data(p)


def _solve_example():
    p = example_model()
    raw, nodes, _props, _err = solver.run_raw_analysis(p)
    return p, raw, nodes


if st is not None:
    _solve_example = st.cache_data(show_spinner=False)(_solve_example)


def _example_combined(mode='Design (ULS)'):
    p, raw, nodes = _solve_example()
    return p, raw, nodes, solver.combine_results(raw, p, mode)


# ==========================================================================
# FIGURES - live result plots from the worked example (reuse the app's viz)
# ==========================================================================

def fig_structure():
    p, raw, nodes, comb = _example_combined()
    geom = raw.get('Dead Load', {})
    return viz.create_plotly_fig(
        nodes, geom, {}, 'M', target_height=2.0, title="",
        show_A=True, show_B=False, geom_A=geom, params_A=p, nodes_A=nodes,
        show_supports=True, support_size=0.6,
        show_element_names=True, structure_only=True)


def _result_fig(case, type_base, mode='Design (ULS)'):
    p, raw, nodes, comb = _example_combined(mode)
    return viz.create_plotly_fig(
        nodes, comb.get(case, {}), {}, type_base, target_height=2.0, title="",
        show_A=True, show_B=False, load_case_name=case,
        geom_A=raw.get('Dead Load'), params_A=p, nodes_A=nodes,
        show_supports=True, support_size=0.6)


def fig_moment_envelope():
    return _result_fig('Total Envelope', 'M', 'Design (ULS)')


def fig_deflection_envelope():
    return _result_fig('Total Envelope', 'Def', 'Characteristic (SLS)')


def fig_soil_case():
    return _result_fig('Soil', 'M', 'Design (ULS)')


# ==========================================================================
# FIGURES - schematics (Plotly, so they render in the app and the PDF alike)
# ==========================================================================

def _blank_axes(fig, x_range=None, y_range=None, height=300):
    fig.update_layout(
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range, scaleanchor="x", scaleratio=1),
        plot_bgcolor='white', margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False, height=height)
    return fig


def _arrow(fig, x0, y0, x1, y1, color, width=1.6):
    """Arrow from (x0,y0) to (x1,y1) in data coordinates."""
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=width,
                       arrowcolor=color, text="")


def fig_sign_convention():
    """Four-panel sign-convention reference, reproducing the report's diagram:
    sagging-positive bending drawn on the tension side, classical shear,
    tension-positive axial force, and global-axes deflection."""
    fig = go.Figure()
    RED, BLUE, GREY, GREEN = 'red', 'blue', 'grey', 'rgb(33,102,64)'

    def title(cx, text):
        fig.add_annotation(x=cx, y=10.3, text=f"<b>{text}</b>", showarrow=False,
                           font=dict(size=12, color='black'))

    def label(cx, cy, text, color, size=10):
        fig.add_annotation(x=cx, y=cy, text=text, showarrow=False, font=dict(size=size, color=color))

    # --- Panel 1: Bending M (two-span beam, moment on the tension side) ---
    title(8.0, "Bending M")
    beam_y = 6.4
    fig.add_trace(go.Scatter(x=[2, 14], y=[beam_y, beam_y], mode='lines',
                             line=dict(color='black', width=2), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[2, 8, 14], y=[beam_y - 0.25] * 3, mode='markers',
                             marker=dict(symbol='triangle-up', size=12, color='white',
                                         line=dict(color='black', width=1)), hoverinfo='skip'))
    mx, my = [], []
    for i in range(41):
        xi = i / 40.0 * 2.0
        xis = xi if xi <= 1.0 else 2.0 - xi
        m = 0.375 * xis - 0.5 * xis * xis
        mx.append(2.0 + 6.0 * xi)
        my.append(beam_y - 9.0 * m)
    fig.add_trace(go.Scatter(x=mx, y=my, mode='lines', line=dict(color=RED, width=1.8), hoverinfo='skip'))
    label(4.6, 4.6, "+M (sagging)", RED, 9)
    label(8.0, 8.1, "-M (hogging)", RED, 9)

    # --- Panel 2: Shear V (classical convention, both signs) ---
    title(22.0, "Shear V")
    fig.add_shape(type='rect', x0=18, y0=5.2, x1=21, y1=8.0, line=dict(color='black', width=1.2))
    _arrow(fig, 17.5, 5.4, 17.5, 7.8, RED)
    _arrow(fig, 21.5, 7.8, 21.5, 5.4, RED)
    label(19.5, 4.4, "+V", RED, 9)
    fig.add_shape(type='rect', x0=24, y0=5.2, x1=27, y1=8.0, line=dict(color='black', width=1.2))
    _arrow(fig, 23.5, 7.8, 23.5, 5.4, BLUE)
    _arrow(fig, 27.5, 5.4, 27.5, 7.8, BLUE)
    label(25.5, 4.4, "-V", BLUE, 9)
    label(22.5, 3.4, "+V: left face up, right face down", GREY, 8)

    # --- Panel 3: Normal force N (tension positive) ---
    title(34.0, "Normal force N")
    fig.add_shape(type='rect', x0=31, y0=7.0, x1=37, y1=7.8, fillcolor='lightgrey', line=dict(color='black', width=0.8))
    _arrow(fig, 31, 7.4, 29.3, 7.4, RED)
    _arrow(fig, 37, 7.4, 38.7, 7.4, RED)
    label(34.0, 6.4, "+N (tension)", RED, 9)
    fig.add_shape(type='rect', x0=31, y0=4.4, x1=37, y1=5.2, fillcolor='lightgrey', line=dict(color='black', width=0.8))
    _arrow(fig, 29.3, 4.8, 31, 4.8, BLUE)
    _arrow(fig, 38.7, 4.8, 37, 4.8, BLUE)
    label(34.0, 3.7, "-N (compression)", BLUE, 9)

    # --- Panel 4: Deflection (global axes) ---
    title(45.5, "Deflection")
    ox, oy = 45.0, 6.0
    _arrow(fig, ox, oy, ox, oy + 2.6, GREEN); label(ox + 0.6, oy + 2.6, "+", GREEN, 12)
    _arrow(fig, ox, oy, ox + 3.0, oy, GREEN); label(ox + 3.2, oy + 0.5, "+", GREEN, 12)
    _arrow(fig, ox, oy, ox, oy - 2.6, GREY); label(ox + 0.6, oy - 2.6, "-", GREY, 12)
    _arrow(fig, ox, oy, ox - 3.0, oy, GREY); label(ox - 3.2, oy + 0.5, "-", GREY, 12)
    label(45.0, 3.0, "up / right positive", GREY, 8)

    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 50]),
        yaxis=dict(visible=False, range=[2.5, 11]),
        plot_bgcolor='white', margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False, height=260)
    return fig


def fig_section_shapes():
    """The three Section Profiler depth shapes: constant, linear taper, 3-point."""
    fig = go.Figure()
    shapes = [
        ("Constant", [0.8, 0.8, 0.8], 0.0),
        ("Linear taper", [1.1, 0.85, 0.6], 4.5),
        ("3-point", [0.6, 1.1, 0.6], 9.0),
    ]
    for label, (h0, hm, h1), x0 in shapes:
        xs = [x0, x0 + 3.5]
        # top chord flat, bottom chord follows the depth profile
        top = [0.0, 0.0]
        bot = [-h0, -h1]
        mid_b = -hm
        px = [x0, x0 + 1.75, x0 + 3.5, x0 + 3.5, x0]
        py = [-h0, mid_b, -h1, 0.0, 0.0]
        fig.add_trace(go.Scatter(x=px, y=py, mode='lines', fill='toself',
                                 line=dict(color='#555'), fillcolor='rgba(90,90,90,0.18)',
                                 hoverinfo='skip'))
        fig.add_annotation(x=x0 + 1.75, y=0.35, text=label, showarrow=False, font=dict(size=12))
    return _blank_axes(fig, x_range=[-0.5, 13], y_range=[-1.4, 0.8], height=220)


def fig_phi_curve():
    """Dynamic factor vs influence length (DK NA A.2.3.x form used by the app)."""
    L = np.linspace(0, 60, 200)
    phi = np.clip(1.25 - (L - 5.0) / 225.0, 1.05, 1.25)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=L, y=phi, mode='lines', line=dict(color='blue', width=3),
                             hoverinfo='skip'))
    fig.add_hline(y=1.25, line=dict(color='grey', dash='dot'))
    fig.add_hline(y=1.05, line=dict(color='grey', dash='dot'))
    fig.add_annotation(x=2.5, y=1.26, text="1.25 (L <= 5 m)", showarrow=False, font=dict(size=11))
    fig.add_annotation(x=55, y=1.06, text="1.05 (L >= 50 m)", showarrow=False, font=dict(size=11))
    fig.update_layout(
        xaxis=dict(title="Influence length L [m]"),
        yaxis=dict(title="Dynamic factor"),
        plot_bgcolor='white', margin=dict(l=50, r=20, t=20, b=40),
        showlegend=False, height=320)
    return fig


def fig_support_types():
    """Schematic of the four boundary-condition types and the DOFs each restrains."""
    fig = go.Figure()
    centres = [(4.0, "Fixed", "Kx, Ky, M"),
               (12.0, "Pinned", "Kx, Ky"),
               (20.0, "Roller (X-free)", "Ky"),
               (28.0, "Spring", "Kx, Ky, M")]
    ground = 1.4
    for cx, name, dofs in centres:
        # ground hatch
        fig.add_shape(type='line', x0=cx - 2.2, y0=ground, x1=cx + 2.2, y1=ground,
                      line=dict(color='black', width=1.4))
        for hx in np.linspace(cx - 2.0, cx + 2.0, 7):
            fig.add_shape(type='line', x0=hx, y0=ground, x1=hx - 0.5, y1=ground - 0.5,
                          line=dict(color='grey', width=0.8))
        if name == "Fixed":
            fig.add_shape(type='rect', x0=cx - 0.9, y0=ground, x1=cx + 0.9, y1=ground + 1.0,
                          line=dict(color='black', width=1.2), fillcolor='rgba(90,90,90,0.18)')
        elif name == "Pinned":
            fig.add_trace(go.Scatter(x=[cx, cx - 1.0, cx + 1.0, cx], y=[ground + 1.4, ground, ground, ground + 1.4],
                                     mode='lines', line=dict(color='black', width=1.4), hoverinfo='skip'))
        elif name.startswith("Roller"):
            fig.add_trace(go.Scatter(x=[cx, cx - 1.0, cx + 1.0, cx], y=[ground + 1.4, ground + 0.45, ground + 0.45, ground + 1.4],
                                     mode='lines', line=dict(color='black', width=1.4), hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=[cx - 0.6, cx, cx + 0.6], y=[ground + 0.2] * 3, mode='markers',
                                     marker=dict(symbol='circle', size=8, color='white',
                                                 line=dict(color='black', width=1)), hoverinfo='skip'))
        else:  # Spring
            zz_y = np.linspace(ground, ground + 1.4, 9)
            zz_x = cx + 0.45 * np.array([0, 1, -1, 1, -1, 1, -1, 1, 0])
            fig.add_trace(go.Scatter(x=zz_x, y=zz_y, mode='lines', line=dict(color='black', width=1.4), hoverinfo='skip'))
        # node + member stub
        fig.add_trace(go.Scatter(x=[cx], y=[ground + 1.45], mode='markers',
                                 marker=dict(symbol='circle', size=7, color='black'), hoverinfo='skip'))
        fig.add_shape(type='line', x0=cx - 1.6, y0=ground + 1.45, x1=cx + 1.6, y1=ground + 1.45,
                      line=dict(color='#1f3b66', width=3))
        fig.add_annotation(x=cx, y=ground + 2.2, text=f"<b>{name}</b>", showarrow=False, font=dict(size=11))
        fig.add_annotation(x=cx, y=ground - 0.95, text=dofs, showarrow=False, font=dict(size=9, color='grey'))
    return _blank_axes(fig, x_range=[0, 32], y_range=[-0.6, 4.6], height=240)


def fig_udl_coupling_window():
    """How the Traffic UDL couples with a moving vehicle: the UDL fills the deck
    except a window around the vehicle (its footprint plus the clear distance)."""
    fig = go.Figure()
    deck_y = 2.0
    fig.add_shape(type='line', x0=0, y0=deck_y, x1=30, y1=deck_y, line=dict(color='black', width=2.5))
    veh_lo, veh_hi = 13.0, 17.0          # vehicle footprint
    gap = 3.0                             # clear distance each side
    w_lo, w_hi = veh_lo - gap, veh_hi + gap
    # adverse UDL fill on the loaded regions (outside the window)
    for x0, x1 in [(0, w_lo), (w_hi, 30)]:
        fig.add_shape(type='rect', x0=x0, y0=deck_y, x1=x1, y1=deck_y + 1.1,
                      fillcolor='rgba(33,102,64,0.18)', line=dict(width=0))
        for ax in np.arange(x0 + 0.4, x1, 1.2):
            _arrow(fig, ax, deck_y + 1.1, ax, deck_y + 0.15, 'rgb(33,102,64)', 1.1)
    # excluded window
    fig.add_shape(type='rect', x0=w_lo, y0=deck_y - 0.1, x1=w_hi, y1=deck_y + 1.4,
                  fillcolor='rgba(200,200,200,0.25)', line=dict(color='grey', width=0.8, dash='dot'))
    # vehicle axles
    for ax in [veh_lo + 0.4, 15.0, veh_hi - 0.4]:
        _arrow(fig, ax, deck_y + 1.6, ax, deck_y + 0.1, 'red', 2.0)
    fig.add_annotation(x=15.0, y=deck_y + 1.95, text="<b>Vehicle</b>", showarrow=False, font=dict(size=11, color='red'))
    fig.add_annotation(x=15.0, y=deck_y - 0.7, text="window: no UDL", showarrow=False, font=dict(size=9, color='grey'))
    fig.add_annotation(x=(w_lo + veh_lo) / 2, y=deck_y - 0.35, text="clear dist.", showarrow=False, font=dict(size=8, color='grey'))
    fig.add_annotation(x=5.0, y=deck_y + 1.45, text="adverse UDL", showarrow=False, font=dict(size=9, color='rgb(33,102,64)'))
    fig.add_annotation(x=25.0, y=deck_y + 1.45, text="adverse UDL", showarrow=False, font=dict(size=9, color='rgb(33,102,64)'))
    return _blank_axes(fig, x_range=[-1, 31], y_range=[0.5, 4.4], height=240)


def fig_element_dof():
    """A 2-node 2D frame element and its six degrees of freedom (u, v, theta each node)."""
    fig = go.Figure()
    xi, xj, y = 4.0, 16.0, 3.2
    fig.add_trace(go.Scatter(x=[xi, xj], y=[y, y], mode='lines',
                             line=dict(color='#1f3b66', width=4), hoverinfo='skip'))
    for x, lbl in [(xi, 'i'), (xj, 'j')]:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                 marker=dict(symbol='circle', size=10, color='black'), hoverinfo='skip'))
        _arrow(fig, x, y, x + 2.4, y, 'red', 1.8)
        fig.add_annotation(x=x + 2.7, y=y - 0.45, text=f"u<sub>{lbl}</sub>", showarrow=False,
                           font=dict(size=12, color='red'))
        _arrow(fig, x, y, x, y + 2.4, 'blue', 1.8)
        fig.add_annotation(x=x - 0.55, y=y + 2.5, text=f"v<sub>{lbl}</sub>", showarrow=False,
                           font=dict(size=12, color='blue'))
        th = np.linspace(0.25, 2.5, 30)
        fig.add_trace(go.Scatter(x=x + 1.2 * np.cos(th), y=y + 1.2 * np.sin(th), mode='lines',
                                 line=dict(color='rgb(33,102,64)', width=1.8), hoverinfo='skip'))
        # Plotly renders raw Unicode but not named HTML entities, so "&theta;"
        # would show literally - emit the theta character (U+03B8) via chr(),
        # keeping this source file ASCII-only.
        theta = chr(0x3b8)
        fig.add_annotation(x=x + 1.5, y=y + 1.5, text=f"{theta}<sub>{lbl}</sub>", showarrow=False,
                           font=dict(size=12, color='rgb(33,102,64)'))
        fig.add_annotation(x=x, y=y - 0.85, text=f"node {lbl}", showarrow=False,
                           font=dict(size=9, color='grey'))
    return _blank_axes(fig, x_range=[0, 22], y_range=[0.8, 7], height=240)


def fig_transverse_spreading():
    """A wheel/axle load spreading through surfacing and slab over an effective width, so the
    per-strip line load - and hence the design M and V - is lower than for a narrow strip."""
    fig = go.Figure()
    surf_top, slab_top, slab_bot = 6.0, 5.4, 3.8
    fig.add_shape(type='rect', x0=0, y0=slab_top, x1=20, y1=surf_top,
                  fillcolor='rgba(150,150,150,0.25)', line=dict(width=0))
    fig.add_shape(type='rect', x0=0, y0=slab_bot, x1=20, y1=slab_top,
                  fillcolor='rgba(90,90,90,0.18)', line=dict(color='#555', width=1))
    fig.add_annotation(x=19.4, y=(surf_top + slab_top) / 2, text="surfacing", showarrow=False,
                       font=dict(size=8, color='grey'), xanchor='right')
    fig.add_annotation(x=19.4, y=(slab_top + slab_bot) / 2, text="slab", showarrow=False,
                       font=dict(size=8, color='grey'), xanchor='right')
    cx = 10.0
    _arrow(fig, cx, surf_top + 1.7, cx, surf_top, 'red', 2.6)
    fig.add_annotation(x=cx, y=surf_top + 2.0, text="<b>wheel load P</b>", showarrow=False,
                       font=dict(size=11, color='red'))
    mid = (slab_top + slab_bot) / 2
    spread = surf_top - mid  # ~45 degree spread
    for s in (-1, 1):
        fig.add_trace(go.Scatter(x=[cx, cx + s * spread], y=[surf_top, mid], mode='lines',
                                 line=dict(color='red', width=1, dash='dot'), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[cx - spread, cx + spread], y=[mid, mid], mode='lines',
                             line=dict(color='rgb(33,102,64)', width=2.6), hoverinfo='skip'))
    fig.add_annotation(x=cx, y=mid - 0.7, text="effective width b<sub>eff</sub>", showarrow=False,
                       font=dict(size=10, color='rgb(33,102,64)'))
    return _blank_axes(fig, x_range=[-1, 21], y_range=[2.5, 8.4], height=260)


# ==========================================================================
# CONTENT - structured blocks (single source for app + PDF)
# ==========================================================================
# Block types:
#   ('h1'|'h2'|'h3', text)
#   ('md', markdown_text)
#   ('callout', kind, text)      kind in concept|theory|standard|tip|limit
#   ('figure', fig_callable, caption)
#   ('table', headers, rows)

_CALLOUT = {
    'concept':  (":large_blue_diamond:", "In plain terms"),
    'theory':   (":triangular_ruler:", "Theory"),
    'standard': (":blue_book:", "Standards"),
    'tip':      (":bulb:", "Tip"),
    'limit':    (":warning:", "Assumption / limitation"),
}


def _strip_num(text):
    """Drop a leading hardcoded section number so headings can be auto-numbered
    (lets a section be inserted without renumbering every following heading)."""
    return re.sub(r'^\s*\d+(?:\.\d+)*\.?\s+', '', text)


def manual_blocks():
    B = []
    h1 = lambda t: B.append(('h1', t))
    h2 = lambda t: B.append(('h2', t))
    h3 = lambda t: B.append(('h3', t))
    md = lambda t: B.append(('md', t))
    call = lambda k, t: B.append(('callout', k, t))
    fig = lambda f, c: B.append(('figure', f, c))
    table = lambda h, r: B.append(('table', h, r))
    part = lambda t: B.append(('part', t))

    md(f"*BriCoS v{data_mod.APP_VERSION} - user manual. This is the complete reference for the "
       "program: every feature and option, and the theory behind the solver. A worked example "
       "(a 2-span integral frame bridge) is carried through to illustrate.*")

    # =====================================================================
    # PART A - GET STARTED
    # =====================================================================
    part("Part A - Get started")

    # ---- 1. Introduction -------------------------------------------------
    h1("1. Introduction & purpose")
    md("**BriCoS** (Bridge Comparison Software) is a 2D finite-element tool for the "
       "rapid analysis and load rating of bridge superstructures and frames. It is built "
       "around the Danish bridge load basis (EN 1991-2 with the Danish National Annex and "
       "the *Vejledning til belastnings- og beregningsgrundlag for broer*), and it lets you "
       "model **two independent structures side by side (System A and System B)** to compare "
       "variants quickly.")
    md("A typical workflow is: build a structure, apply the permanent and traffic loads, run "
       "the analysis, read the factored envelopes, and export a report. The whole model lives "
       "in one screen so changes are immediate.")
    call('concept', "Think of BriCoS as a fast 2D frame solver with bridge load combinations "
         "built in. You draw the structure as spans and walls, attach loads, and it returns "
         "moment/shear/axial/deflection envelopes already combined to ULS and SLS.")
    call('limit', "The model is **2D and linear-elastic**. Members connect rigidly along their "
         "centrelines (no eccentricities or releases), and a single partial factor is applied "
         "to both the maximum and minimum of each permanent action - favourable/unfavourable "
         "permanent-load cases are not split automatically (verify those separately).")

    h2("What BriCoS does - at a glance")
    md("- **Two structures side by side (System A vs System B).** The core workflow: model two "
       "variants and read their results on the same diagrams - System A in blue, System B in "
       "red - to compare them directly.\n"
       "- **Fast moving-load envelopes.** Vehicles are stepped across the structure "
       "automatically and BriCoS returns the absolute maximum/minimum $M$, $V$, $N$ and "
       "deflection envelopes, so the governing design effects come back without positioning "
       "loads by hand or building a full FE model.\n"
       "- **Built-in bridge load combinations.** ULS/SLS partial factors, the dynamic factor "
       "$\\Phi$, and the coupled vehicle + Traffic UDL Total Envelope, per DS/EN 1991-2 + DK NA "
       "and the VD *Vejledning til belastnings- og beregningsgrundlag for broer*.\n"
       "- **2D frame / beam / superstructure** modelling with non-prismatic sections, soil and "
       "surcharge on walls, and flexible supports.\n"
       "- **Reports & export.** A one-click PDF report and an Excel/CSV data package.")
    call('concept', "Speed is the point. Because BriCoS envelopes the moving loads and the load "
         "combinations for you, the governing design forces come back in seconds - which is "
         "what makes it practical for side-by-side comparisons and for judging how much "
         "analysis a job actually needs (see Common use cases).")

    # ---- 2. Quick start --------------------------------------------------
    h1("2. Quick start (5 minutes)")
    md("1. **Define the structure.** In the sidebar open *Geometry, Stiffness & Static Loads*, "
       "pick the model type (Frame/Beam/Superstructure) and set the number of spans and their "
       "lengths. For a frame, set the wall heights.\n"
       "2. **Set sections.** Give each span/wall a depth (height) or use the *Section Profiler* "
       "for tapers.\n"
       "3. **Add loads.** Enter the permanent Dead Load, then any soil, surcharge, Traffic UDL "
       "and vehicles under *Vehicle Definitions*.\n"
       "4. **Choose what to analyse.** In *Analysis & Result Settings* toggle ULS and/or SLS.\n"
       "5. **Read results.** The main panel shows the bending/shear/axial/deflection diagrams "
       "and envelopes; switch load case and result type in the toolbar.\n"
       "6. **Export.** Generate a PDF report or download the Excel/CSV data package.")
    fig(fig_structure, "The worked example as BriCoS draws it: spans S1-S2 and walls W1-W3, "
        "with supports. Turn on *Element Names* to label members like this.")
    call('tip', "Everything updates as you type - there is no separate 'run' button. If a "
         "system is incompletely defined it is reported as a warning and excluded, so you can "
         "still work on the other system.")

    # ---- Common use cases (auto-numbered) -------------------------------
    h1("Common use cases")
    md("These workflows all use the same idea: put one structure (or one set of assumptions) "
       "in **System A** and a variant in **System B**, then read the governing envelopes side "
       "by side.")
    h2("Conservative design forces for a simple static system")
    md("Get bending, shear, axial and deflection **envelopes** for a simple beam or frame under "
       "the governing permanent and traffic actions - quickly, without setting up a full FE "
       "model. The 2D linear-elastic model, the adverse-only Traffic UDL (DS/EN 1991-2, "
       "4.3.2(1)(b)) and the full moving-load envelopes are inherently on the safe side, so "
       "this suits preliminary sizing or an independent check of another analysis.")
    h2("Comparing alternative static systems")
    md("Put two arrangements in A and B - for example simply-supported vs continuous, or the "
       "effect of making a joint/support continuous or adding a support - and compare the "
       "envelopes directly to see how much the static system drives the load effects.")
    h2("Normal vs conditional passage")
    md("Model the same structure twice and vary **only the partial factors and the dynamic "
       "factor $\\Phi$** between System A (normal passage) and System B (conditional or "
       "restricted passage - for example reduced speed, giving a lower $\\Phi$, or an escorted, "
       "centred single-vehicle passage). The side-by-side envelopes show what the passage "
       "conditions are worth in design terms.")
    call('standard', "The dynamic factor follows DS/EN 1991-2 + DK NA; the classification load "
         "model and partial factors follow the VD *Vejledning til belastnings- og "
         "beregningsgrundlag for broer*. Set the factors per system under *Design Factors & "
         "Type*.")
    h2("Classification vehicle vs special vehicle")
    md("Compare a standard **classification vehicle** (a built-in class) in System A against a "
       "**special / permit vehicle** (custom axle loads and spacings) in System B, to see "
       "whether and by how much the special transport governs the design.")
    call('tip', "A comparison is itself a decision tool: how much the result changes between A "
         "and B tells you how sensitive the design is to the modelling choice (system, vehicle, "
         "factors), and therefore whether a quick conservative check is enough or a more "
         "detailed analysis is warranted.")

    # =====================================================================
    # PART B - FEATURE & OPTION REFERENCE
    # =====================================================================
    part("Part B - Feature & option reference")

    # ---- The workspace ---------------------------------------------------
    h1("The workspace: systems, files and sessions")
    md("The screen is split into a **sidebar** (all inputs, grouped in expandable panels) and a "
       "**main panel** (the result diagrams and toolbar). BriCoS always holds **two independent "
       "models, System A and System B**. The **Active System** selector chooses which one the "
       "sidebar edits; both are drawn together on every diagram - **System A in blue, System B "
       "in red** - so you always see the comparison.")
    md("- **Copy Data** - clone one system into the other (A to B or B to A) as a starting point "
       "for a variant.\n"
       "- **Reset Data** - clear a system or restore the default example.\n"
       "- **File Operations (Save/Load)** - download the entire configuration of both systems "
       "as a CSV file, and re-upload it later to continue exactly where you left off.\n"
       "- **Autosave** - the session is saved automatically on each interaction (clicks, edits); "
       "it does not save while idle, so a browser refresh will not lose committed work.")
    call('tip', "To analyse a single structure, define one system and leave the other empty. An "
         "incompletely defined system is reported as a **warning and excluded** from the results "
         "and report - it never blocks the system that is complete.")

    # ---- Modelling the structure ----------------------------------------
    h1("Modelling the structure")
    h2("Model type")
    md("Set under *Design Factors & Type*:\n"
       "- **Frame** - decks (spans) carried on vertical walls/piers, with full wall-slab "
       "interaction. Soil and surcharge on the walls are available in this mode. Use it for "
       "integral and portal frames.\n"
       "- **Superstructure** - the deck alone on point supports, with no walls (a beam on "
       "supports). Use it for superstructure-only checks. Switching to Superstructure sets the "
       "wall heights and inertias to zero and removes soil/surcharge; switching back to Frame "
       "restores them.")
    h2("Geometry: spans, walls and nodes")
    md("Choose the **number of spans** and each **span length** $L$; for a frame, set each "
       "**wall height** $h$. Members are named **spans** ($S1, S2, \\ldots$, the horizontal "
       "deck) and **walls** ($W1, W2, \\ldots$, the vertical members). Each member is meshed "
       "into sub-elements for the analysis (see *Calculation precision*). Supports sit at the "
       "deck nodes (Superstructure) or the wall bases (Frame).")
    h2("Sections and the Section Profiler")
    md("Each member's cross-section is defined in one of two ways:\n"
       "- **By height** $h$ - with the strip width $b_{eff}$ this gives $I = b_{eff} h^3/12$. "
       "Required for auto self-weight and the section-depth overlay.\n"
       "- **By inertia** $I$ - enter the second moment of area directly (e.g. a cracked or "
       "transformed section). The depth-based features are then unavailable.\n"
       "The **Section Profiler (Advanced)** lets the depth vary along a member:")
    table(["Shape", "Depth along the member", "Use"],
          [["Constant", "One depth, end to end", "Prismatic members"],
           ["Linear taper", "Straight variation, start to end", "Haunched / tapered spans"],
           ["3-point", "Start, mid and end depths", "Drop panels, mid-span haunches"]])
    fig(fig_section_shapes, "The three Section Profiler depth shapes. Tapers are integrated per "
        "sub-element, so non-prismatic results converge as the mesh is refined.")
    call('concept', "$b_{eff}$ is the width of the **analysis strip**. All loads are line loads "
         "on that strip and BriCoS does not spread them transversely - you choose the width your "
         "loads and section refer to.")
    call('tip', "**Slab bridges:** because BriCoS analyses a single longitudinal strip with no "
         "transverse distribution, its section forces are **conservative** for slabs. Accounting "
         "for **transverse load spreading** - a wider effective width over which a wheel/axle "
         "load distributes across the slab - can reduce the design moments and shears "
         "**considerably**. Reflect it by choosing $b_{eff}$ and the line-load intensities to "
         "represent the spread load, or by applying a separate transverse distribution outside "
         "BriCoS.")
    h2("Boundary conditions")
    md("Each support node is one of the following. Internally, restraints use the **penalty "
       "method** (a restrained DOF is a very stiff spring, $k \\approx 10^{14}$); a Custom "
       "spring uses your stiffnesses directly to model an elastic foundation or finite bearing "
       "stiffness.")
    table(["Support", "Restrains", "Typical use"],
          [["Fixed", "$K_x$, $K_y$ and rotation", "Frame wall base (encastre)"],
           ["Pinned", "$K_x$ and $K_y$ (rotation free)", "Pinned bearing"],
           ["Roller (X-free)", "$K_y$ only", "Expansion bearing"],
           ["Custom spring", "User $K_x$, $K_y$, $K_\\theta$", "Elastic foundation / bearing"]])
    fig(fig_support_types, "The four support types and the degrees of freedom each restrains.")
    h2("Alignment")
    md("A span can be **Straight (horizontal)** or **Inclined**. Inclination is set either by "
       "**Slope [%]** or by **Delta height (end - start) [m]**, so ramped or stepped decks can "
       "be modelled. Inclination changes how the axial force $N$ and shear $V$ resolve in the "
       "member's local axes (see Part C, *Sign convention & local axes*).")

    # ---- Loads -----------------------------------------------------------
    h1("Loads")
    md("BriCoS separates **permanent** actions from **variable (traffic)** actions so they are "
       "factored independently in the combinations. The table summarises every load; the "
       "subsections give the options.")
    table(["Load", "What it is", "How it is applied"],
          [["Dead Load", "Permanent superimposed load (surfacing, ballast, parapets)",
            "Per-span line load [kN/m]"],
           ["Self-weight (auto)", "Structural self-weight from unit weight x section",
            "Computed per sub-element when enabled"],
           ["Soil", "Backfill pressure on frame walls",
            "Linear pressure profile per wall face"],
           ["Surcharge", "Traffic/area surcharge behind walls", "Pressure on the wall face"],
           ["Traffic UDL", "Uniformly distributed traffic load", "Adverse-only line load on the deck"],
           ["Vehicle", "Axle-load model(s), moving", "Stepped across the structure (A and/or B)"]])
    h2("Dead Load and self-weight")
    md("- **Dead Load** - the per-span permanent line load [kN/m]. With auto self-weight off it "
       "carries all permanent load; with it on it carries only **superimposed** actions "
       "(surfacing, ballast, parapets).\n"
       "- **Auto-calculate self-weight** (*Analysis & Result Settings*) - computes the "
       "structural self-weight of decks and walls as $\\gamma \\cdot b_{eff} \\cdot h$ per "
       "sub-element from the **unit weight** $\\gamma$ (reinforced concrete approx. 25 kN/m^3) "
       "and reports it as a separate *Self-weight* load case, factored like the Dead Load. It "
       "requires **height-defined** sections, and $b_{eff}$ is then the physical cross-section "
       "width.")
    h2("Soil and surcharge (Frame)")
    md("- **Soil** - backfill pressure on a wall face, entered as a **linear pressure profile** "
       "(top and bottom intensity) per face.\n"
       "- **Surcharge** - a traffic/area surcharge acting on the wall face.\n"
       "- **Surcharge combination** (*Analysis & Result Settings*) sets how the surcharge and "
       "the deck vehicle interact: **Exclusive** takes $\\max(\\text{Vehicle}, "
       "\\text{Surcharge})$; **Simultaneous** applies $\\text{Vehicle} + \\text{Surcharge}$ "
       "together.")
    fig(fig_soil_case, "Soil (earth-pressure) bending in the worked example - one of the "
        "permanent load cases that can be inspected on its own.")
    h2("Traffic UDL")
    md("The Traffic UDL is a uniformly distributed traffic load entered as a **line load** $q$ "
       "[kN/m] on the analysis strip - like the self-weight, you account for the loaded width "
       "yourself. It is applied **adverse-only** (placed only where it worsens the effect at "
       "each result point, per EN 1991-2:2003 4.3.2(1)(b)), and the dynamic factor $\\Phi$ is "
       "**not** applied to it because its intensity already includes the dynamic increment "
       "(DK NA A.2.3.2). Set $q = 0$ to deactivate it.")
    h2("Vehicles")
    md("Define **Vehicle A** and, optionally, **Vehicle B** under *Vehicle Definitions*. Each is "
       "either a built-in **classification vehicle** (standard LM3 classes per DS/EN 1991-2 "
       "DK NA:2017) or **Custom**:\n"
       "- **Loads** - axle loads in tonnes [t], comma-separated, e.g. `10, 10, 15`.\n"
       "- **Axle spacing** - incremental spacing in metres [m]; the first value is 0 and each "
       "subsequent value is the distance from the previous axle (not a cumulative position). "
       "The list length must equal the number of loads, e.g. `0, 1.5, 3.0`.\n"
       "- **Clear Vehicle** removes the vehicle.\n"
       "**Vehicle Direction** (*Analysis & Result Settings*, shared by both systems) is "
       "**Forward** (left to right), **Reverse** (right to left, axles inverted) or **Both** "
       "(the envelope of both directions).")
    call('standard', "The partial factor on the vehicle (e.g. $\\gamma_Q = 1.4$) follows the "
         "*Vejledning* 5.3.1 / Bilag 3; set it per system under *Design Factors & Type*.")
    h2("Coupling the UDL with the vehicle")
    md("When a system has **both** a vehicle and a Traffic UDL, three options control how the "
       "UDL accompanies the moving vehicle. They have no effect when the system has no vehicle - "
       "the UDL then simply enters as its full adverse envelope.\n"
       "- **Application in step results** - *Moving with vehicle*: the UDL fills the deck except "
       "a **window** around the vehicle; *Static (full deck)*: the UDL covers the full deck at "
       "every step.\n"
       "- **Clear distance (vehicle to UDL)** - the gap from the outermost axles to the start of "
       "the UDL, applied in front of and behind the vehicle. The excluded window is the vehicle "
       "footprint plus this gap on each side. Pick a preset or a custom distance.\n"
       "- **Apply UDL also within the vehicle window** (footprint option) - when enabled, no "
       "window is excluded and the UDL coexists with the vehicle over its footprint.")
    fig(fig_udl_coupling_window, "Moving application: the adverse UDL fills the deck except a "
        "window around the vehicle (its footprint plus the clear distance on each side).")
    call('theory', "These options change how the UDL is coupled with the vehicle in the "
         "**Total Envelope**: *Moving* excludes the window around each vehicle position, while "
         "*Static* and the *footprint* option apply the full adverse UDL everywhere, reproducing "
         "the conservative *vehicle + full UDL* superposition. The exact algorithm is in Part C, "
         "*The coupled Total Envelope*.")

    # ---- Analysis & result settings -------------------------------------
    h1("Analysis & result settings")
    md("These shared settings (under *Analysis & Result Settings*) apply to both systems.")
    h2("Limit states to analyse")
    md("Toggle **ULS (Design)** and **SLS (Characteristic)** independently. ULS applies the "
       "partial factors ($\\times K_{FI}$); SLS applies the SLS combination factors. The "
       "**Unfactored** combination (all loads $\\times 1.0$, no dynamic factor) is always "
       "available; if neither limit state is selected it is the only result produced.")
    h2("Shear deformations (Timoshenko)")
    md("Off by default. When enabled, shear flexibility is included in the stiffness of "
       "**prismatic** members (recommended for deep beams and piers); **Poisson's ratio** "
       "$\\nu$ then appears and feeds only the shear modulus $G$. For non-prismatic members "
       "shear deformation is currently not included. The strip width $b_{eff}$ is always used "
       "(shear area, axial area and auto self-weight).")
    h2("Calculation precision")
    md("- **Mesh size [m]** - the sub-element length. For **prismatic** members the internal "
       "forces $M, V, N$ are exact regardless of mesh size; for **non-prismatic** (tapered / "
       "3-point) members they **converge** as the mesh is refined rather than being exact at "
       "any size, so use a finer mesh on tapered members. **Deflections** are interpolated "
       "between nodes, so their accuracy under loads also improves with a finer mesh (the 0.5 m "
       "default keeps the error negligible).\n"
       "- **Vehicle step [m]** - the moving-load increment. Smaller steps sample the envelopes "
       "more densely (more positions evaluated), at some speed cost.")

    # ---- Design factors --------------------------------------------------
    h1("Design factors, limit states & the dynamic factor")
    md("Set per system under *Design Factors & Type*. Only the factors for the analysed limit "
       "states are shown.")
    h2("Material definition")
    md("- **Eurocode ($f_{ck}$)** - the elastic modulus $E$ is derived from the concrete "
       "strength class.\n"
       "- **Manual (E-modulus)** - enter $E$ directly.")
    h2("ULS partial factors")
    md("Applied in the Design (ULS) result mode, each multiplied by $K_{FI}$. Defaults follow "
       "the *Vejledning* Fig. B3.1.")
    table(["Factor", "Applies to", "Notes / default"],
          [["$K_{FI}$", "All loads in ULS (not SLS)", "Consequence class; soil may negate it"],
           ["$\\gamma_g$", "Dead Load and Self-weight", "Permanent action factor"],
           ["$\\gamma_j$", "Soil (earth pressure)", "Has a '1.0 (No KFI)' option for earth pressure"],
           ["$\\gamma_{veh,A}$", "Vehicle A (with $\\Phi$) and Surcharge", "Variable traffic, model A"],
           ["$\\gamma_{veh,B}$", "Vehicle B (with $\\Phi$)", "Variable traffic, model B"],
           ["$\\gamma_{UDL}$", "Traffic UDL", "0.56 as companion to vehicles; 1.40 for the UDL-alone case"]])
    call('limit', "Each permanent factor is applied to **both** the maximum and minimum result "
         "- favourable/unfavourable permanent-load combinations are not evaluated "
         "automatically. Where a permanent action is favourable (uplift, relieving earth "
         "pressure), re-run with the favourable factor. The soil '1.0 (No KFI)' option gives an "
         "effective factor of exactly 1.0 with $K_{FI}$ cancelled, as permitted for earth "
         "pressure.")
    h2("SLS combination factors")
    md("Applied in the Characteristic (SLS) result mode; $K_{FI}$ is not applied in SLS. "
       "Defaults follow the characteristic combination of *Vejledning* Fig. B3.2.")
    table(["Action", "SLS factor (default)"],
          [["Self-weight", "1.0"], ["Soil", "1.0"], ["Vehicle A", "1.0"],
           ["Vehicle B", "0.75"], ["Traffic UDL", "0.40"]])
    h2("Dynamic factor $\\Phi$")
    md("Moving-vehicle effects are amplified by $\\Phi$. Choose how it is obtained:\n"
       "- **Calculate** - from the **influence length** $L_{inf}$, with two bases:\n"
       "  - *Combined system (EN 1991-2 Tab. 6.2)* - one determinant length for the whole "
       "structure (a frame is treated as an equivalent continuous beam); generally a lower "
       "$\\Phi$ for short spans.\n"
       "  - *Per span (DK NA A.2.3.5(2))* - $L_{inf}$ is the actual span, evaluated per span; "
       "walls take the maximum $\\Phi$ of the adjacent spans.\n"
       "  Then **Phi application** is *Per member* (each span its own $\\Phi$) or *Governing* "
       "(the largest $\\Phi$ applied to every member - conservative).\n"
       "- **Manual** - enter $\\Phi$ directly, either **Global** (one value) or **Per span** (a "
       "value per span; walls take the max of adjacent spans).\n"
       "The **Phi Calculation Log** lists the resulting per-member values.")
    md("The **SLS treatment** of $\\Phi$ (shown when SLS is analysed): *Same as ULS* (no "
       "reduction), *Reduced* $\\Phi_{SLS} = 1 + (\\Phi-1)/2$ (*Vejledning* 5.4.2), or a "
       "*Manual SLS value*.")
    fig(fig_phi_curve, "Calculated dynamic factor vs influence length: 1.25 up to 5 m, "
        "decreasing to 1.05 at 50 m.")

    # ---- Results & visualization ----------------------------------------
    h1("Results & visualization")
    md("The main panel shows four diagrams - **Bending Moment, Shear, Normal Force, "
       "Deflection** - for the selected **Load Case** and **Result Type**.")
    h2("Result type")
    table(["Result Type", "What it applies"],
          [["Design (ULS)", "ULS partial factors $\\times K_{FI}$, with $\\Phi$ on the vehicle"],
           ["Characteristic (SLS)", "SLS combination factors, with the SLS $\\Phi$ treatment"],
           ["Unfactored", "All loads $\\times 1.0$, no dynamic factor"]])
    h2("Load case")
    md("The **Total Envelope** is the combined design result (the coupled vehicle + UDL + "
       "permanent envelope). Individual cases - **Dead Load, Self-weight, Soil, Surcharge, "
       "Traffic UDL, Vehicle Envelope** - are offered for checking whenever they are present in "
       "the model.")
    fig(fig_moment_envelope, "ULS bending Total Envelope of the worked example - the coupled "
        "vehicle + UDL + permanent envelope used for design.")
    fig(fig_deflection_envelope, "SLS deflection envelope of the worked example.")
    h2("Visual toggles")
    md("- **Labels** - value labels at the diagram extremes (on by default).\n"
       "- **Show Supports** / **Support Size** - support symbols at the nodes, with adjustable "
       "size.\n"
       "- **Section Depth** - overlays each member's true depth to scale (unavailable while any "
       "section is defined by inertia).\n"
       "- **Element Names** - labels each member ($S1, W2, \\ldots$) in the system colour; a "
       "single neutral label is shown where A and B coincide.\n"
       "- **Target Diagram Height [m]** - scales the diagrams (and the element-label size).")
    h2("Vehicle step viewer")
    md("For a moving vehicle you can **step through positions** and read the effect at each "
       "step, choose the **step direction** (when Both is active), and jump to the **critical "
       "position** per member. A **step-effects selector** shows the vehicle, the UDL, or their "
       "combination separately - useful for seeing how the coupled envelope is built and which "
       "vehicle location governs.")
    h2("Reactions")
    md("Support reactions $R_x, R_y, M_z$ are tabulated per node for the selected result.")

    # ---- Reports & export ------------------------------------------------
    h1("Reports & data export")
    md("**PDF report** (*Report Generation*) - a full document: basis of analysis, global "
       "settings, a per-system input summary (with an element-layout diagram), the ULS/SLS "
       "Total Envelopes, per-component results, the equilibrium check and an appendix. If only "
       "one system is validly defined, the report covers that system alone and notes the "
       "exclusion.")
    md("**Data package** (*Tabular Data*) - an Excel/CSV export with a settings sheet and one "
       "sheet per load case plus the total envelopes, for independent verification.")

    # =====================================================================
    # PART C - THEORY & METHODOLOGY
    # =====================================================================
    part("Part C - Theory & methodology")

    h1("Finite-element formulation")
    md("BriCoS is a 2D **matrix-stiffness** (displacement) solver. The structure is discretised "
       "into beam elements connecting nodes; each node has **three degrees of freedom** - "
       "horizontal $u$, vertical $v$ and rotation $\\theta$ - so a 2-node element has six.")
    fig(fig_element_dof, "A 2-node 2D frame element and its six degrees of freedom (axial $u$, "
        "transverse $v$, rotation $\\theta$ at each node).")
    md("Each element contributes a $6 \\times 6$ stiffness matrix $k$ in its local axes, which "
       "is rotated to global axes by the element's orientation ($T^T k\\,T$) and added into the "
       "global stiffness $K$. The system $K\\,u = F$ is solved for the nodal displacements $u$; "
       "element end forces follow from the local displacements, and the **internal $M, V, N$** "
       "along each member are recovered by adding the member's distributed-load (fixed-end) "
       "contributions. Behaviour is **linear-elastic** throughout - which is what makes the "
       "factored-envelope superposition in the combinations valid.")
    h2("Element stiffness: Euler-Bernoulli and Timoshenko")
    md("A prismatic member uses the closed-form 2D frame element, with the axial stiffness "
       "$EA/L$ decoupled from the bending-shear terms. With **shear deformation** enabled the "
       "element becomes **Timoshenko**, governed by the shear parameter $\\Phi_s = 12EI/(G A_s "
       "L^2)$: it scales the bending stiffness by $1/(1+\\Phi_s)$ and adjusts the rotational "
       "terms (the $4+\\Phi_s$ and $2-\\Phi_s$ coefficients). Setting $\\Phi_s = 0$ recovers "
       "**Euler-Bernoulli**. $G$ comes from $E$ and $\\nu$, and the shear area $A_s$ from "
       "$b_{eff}$. Shear deformation matters when a member is short relative to its depth (deep "
       "beams, squat piers).")
    md("Written out explicitly, with $k_a = EA/L$ the axial stiffness and "
       "$\\beta = EI/[L^3(1+\\Phi_s)]$ the bending base term, the element's local stiffness "
       "matrix - for the degrees of freedom ordered $u_i, v_i, \\theta_i, u_j, v_j, \\theta_j$ - "
       "is:")
    table(["", "$u_i$", "$v_i$", "$\\theta_i$", "$u_j$", "$v_j$", "$\\theta_j$"],
          [["$u_i$", "$k_a$", "0", "0", "$-k_a$", "0", "0"],
           ["$v_i$", "0", "$12\\beta$", "$6L\\beta$", "0", "$-12\\beta$", "$6L\\beta$"],
           ["$\\theta_i$", "0", "$6L\\beta$", "$(4+\\Phi_s)L^2\\beta$", "0", "$-6L\\beta$", "$(2-\\Phi_s)L^2\\beta$"],
           ["$u_j$", "$-k_a$", "0", "0", "$k_a$", "0", "0"],
           ["$v_j$", "0", "$-12\\beta$", "$-6L\\beta$", "0", "$12\\beta$", "$-6L\\beta$"],
           ["$\\theta_j$", "0", "$6L\\beta$", "$(2-\\Phi_s)L^2\\beta$", "0", "$-6L\\beta$", "$(4+\\Phi_s)L^2\\beta$"]])
    md("Setting $\\Phi_s = 0$ reduces this to the **Euler-Bernoulli** element "
       "($\\beta = EI/L^3$, rotational coefficients $4$ and $2$). The matrix is rotated from "
       "local to global axes by $k_{global} = T^T k\\,T$ - with $T$ built from the element's "
       "direction cosines $c_x, c_y$ - and assembled into the global $K$. A **non-prismatic** "
       "member keeps these same six degrees of freedom but replaces the bending block with the "
       "numerically integrated form below.")
    h2("Non-prismatic members")
    md("A member whose depth varies (a Section Profiler taper or 3-point profile) has no single "
       "$I$. BriCoS forms its stiffness by **displacement-based Euler-Bernoulli integration**: "
       "the standard cubic **Hermite** shape functions are used and the bending stiffness is "
       "obtained by integrating $B^T EI(x)\\,B$ numerically (Simpson's rule) along the element. "
       "A constant $EI$ integrates exactly; a varying $EI$ is approximated per sub-element, so "
       "the result **converges** as the mesh is refined. Non-prismatic **shear** deformation is "
       "not included (this path is Euler-Bernoulli only).")
    call('theory', "**Consistent nodal loads and mesh independence.** Distributed loads are not "
         "lumped at the nodes - they are converted to exact equivalent nodal forces by "
         "integrating the Hermite shape functions against the load (the fixed-end forces). "
         "Because these are **exact** for prismatic members, the recovered $M, V, N$ are "
         "**independent of the mesh size**; for non-prismatic members they converge with "
         "refinement. Deflections are the exception (next).")

    h1("Deflections")
    md("Nodal displacements come straight from the solve. The deflected shape **between** nodes "
       "is reconstructed from the nodal values with the Hermite shape functions. A load acting "
       "between two nodes adds a local sag that this internodal interpolation does not capture; "
       "the resulting underestimate is bounded by about $P\\,L_{mesh}^3/(192EI)$ per point load "
       "and **vanishes as the mesh is refined** - which is why mesh size affects deflection "
       "accuracy but not the forces.")

    h1("Moving loads")
    md("Traffic is evaluated **quasi-statically**. The vehicle is stepped across the structure "
       "in increments of the *vehicle step*; at each position its axle loads are placed (as "
       "consistent nodal loads), the structure is solved, and the **absolute maximum and "
       "minimum** of every effect at every result point is accumulated into an envelope, "
       "tracking the governing travel direction per extreme. Where both Vehicle A and Vehicle B "
       "are defined, their envelopes are **superposed independently** - the maximum from A may "
       "be added to the maximum from B even if they occur at different positions, which is "
       "conservative.")

    h1("Load combinations")
    md("Each load case is enveloped first, then the design effect is the superposition of the "
       "**factored** envelopes:")
    md("$$E_d = K_{FI}\\,(\\gamma_G E_{SW} + \\gamma_{Soil} E_{Soil} + "
       "\\gamma_Q \\Phi\\, E_{Veh} + \\gamma_{UDL} E_{UDL} + \\gamma_Q E_{Surch})$$")
    md("with the partial factors set per system (Part B). The Traffic UDL carries its own "
       "factor $\\gamma_{UDL}$ and, unlike the vehicle term, is **not** amplified by $\\Phi$. "
       "When both a vehicle and a UDL are present the two traffic terms ($\\gamma_Q \\Phi\\, "
       "E_{Veh}$ and $\\gamma_{UDL} E_{UDL}$) are **not simply added** - they are combined "
       "through the coupled Total Envelope (next). Because the model is linear, enveloping each "
       "component and then superposing the factored envelopes is valid. The **Unfactored** "
       "combination (all loads $\\times 1.0$, no $\\Phi$) is always available; if neither ULS "
       "nor SLS is selected it is the only result.")

    h1("The dynamic factor")
    md("The dynamic factor $\\Phi$ amplifies the moving-vehicle effects to account for the "
       "vehicle-structure dynamic interaction. BriCoS uses the DK NA form, which decreases with "
       "the influence length: $\\Phi = 1.25$ for $L_{inf} \\le 5$ m, then linearly "
       "$\\Phi = 1.25 - (L_{inf}-5)/225$ for $5 < L_{inf} < 50$ m, and $\\Phi = 1.05$ for "
       "$L_{inf} \\ge 50$ m (the curve is plotted in Part B).")
    md("The **influence length** is taken in one of two ways (selected in Part B):\n"
       "- **Per span** - $L_{inf}$ is the actual span length, applied per member; walls take "
       "the larger $\\Phi$ of the adjacent spans (DK NA A.2.3.5(2)).\n"
       "- **Combined system** - a single determinant length for the whole structure (EN "
       "1991-2:2003, Table 6.2). From the component lengths (the spans, plus the legs for a "
       "frame) BriCoS forms $L_{\\phi} = \\max(k\\,L_{mean}, L_{max})$, where $L_{mean}$ is "
       "their mean and the factor $k$ is $1.2, 1.3, 1.4, 1.5$ for $2, 3, 4, \\ge 5$ components "
       "(Case 5.1/5.2/5.3). A longer determinant length gives a lower $\\Phi$.")
    md("In SLS the factor may be reduced to $\\Phi_{SLS} = 1 + (\\Phi-1)/2$ (*Vejledning* "
       "5.4.2). The full per-member calculation is shown in the *Phi Calculation Log* and the "
       "report.")
    call('standard', "The Traffic UDL is **not** amplified by $\\Phi$ - its intensity already "
         "includes the dynamic increment (DK NA A.2.3.2).")

    h1("The coupled Total Envelope")
    md("When a vehicle and a Traffic UDL coexist, BriCoS does not simply add their separate "
       "envelopes - that would double-count the deck length the vehicle itself occupies. "
       "Instead, for **each vehicle position** it:\n"
       "1. excludes a **window** around the vehicle - its axle footprint extended by the clear "
       "distance on each side, snapped outward to whole mesh segments;\n"
       "2. sums the **adverse** UDL contribution over the deck **outside** that window (via "
       "per-segment prefix sums, so only the worsening segments are added at each result "
       "point);\n"
       "3. adds that to the factored vehicle effect at this position.\n"
       "The envelope over all positions is then combined with the **vehicle-absent** situation "
       "(the full adverse UDL alone) to give the **Total Envelope**. With the *Static* or "
       "*footprint* application the window is empty, so the per-position UDL equals the full "
       "adverse envelope and the result reduces to the conservative *vehicle + full UDL* "
       "superposition (Part B sets these options).")

    h1("Sign convention & local axes")
    fig(fig_sign_convention, "Bending is sagging-positive and plotted on the tension side; "
        "$+N$ is tension. Internal-force signs follow the member's local axes.")
    md("- **Spans (horizontal):** local x aligns with global X, so $N$ is horizontal axial force "
       "and $V$ is vertical shear.\n"
       "- **Walls (vertical):** local x runs along the member, so $N$ is the vertical axial load "
       "and $V$ is the horizontal shear.")

    h1("Equilibrium check")
    md("After every solve BriCoS sums, per load case, the applied loads and the support "
       "reactions and reports the **residual**. The reactions are the forces the supports exert "
       "**on** the structure, so at equilibrium they cancel the applied loads: the residual "
       "$\\sum$ applied $+ \\sum$ reactions should be at machine-zero, confirming the assembly "
       "and solve are self-consistent. It is shown in the report as a built-in QA check. "
       "Patterned cases such as the adverse-only UDL have no single applied total and are "
       "excluded from this check.")

    h1("Transverse load distribution on slab bridges")
    md("BriCoS analyses a **single longitudinal strip** of width $b_{eff}$ and does not "
       "distribute load transversely. For a beam-like deck that is exactly right. For a "
       "**slab** it is **conservative**: a wheel or axle load does not stay on one strip - it "
       "spreads through the surfacing and the slab, and is shared across an **effective width** "
       "by the slab's transverse stiffness.")
    fig(fig_transverse_spreading, "A wheel load spreads through the surfacing and slab over an "
        "effective width $b_{eff}$. The wider that width, the lower the per-strip line load - "
        "and hence the lower the design moment and shear.")
    md("The design effects per strip scale with the line-load intensity - the load divided by "
       "the width it spreads over - so modelling one narrow strip **overestimates** $M$ and $V$, "
       "and accounting for transverse distribution can reduce them **considerably**. Two "
       "practical ways to reflect it in BriCoS:\n"
       "- choose $b_{eff}$ - and convert the axle/wheel loads to line loads over that width - to "
       "represent the realistic effective width (including the roughly 45-degree spread through "
       "the surfacing and slab to mid-depth), or\n"
       "- compute a transverse distribution factor separately (a grillage, an influence-surface "
       "method, or the road-directorate guidance) and apply it to the BriCoS strip result.")
    call('tip', "Use the conservative single-strip result for a quick check or preliminary "
         "sizing; if it governs the design, a transverse-distribution analysis is often where "
         "the real capacity is found. The *Vejledning til belastnings- og beregningsgrundlag "
         "for broer* covers transverse distribution for slab decks.")

    # =====================================================================
    # PART D - REFERENCE
    # =====================================================================
    part("Part D - Reference")

    h1("Standards")
    md("- **EN 1991-2:2003** - traffic loads on bridges (UDL application 4.3.2(1)(b); dynamic "
       "factor Table 6.2).\n"
       "- **EN 1991-2 DK NA:2017** - Danish National Annex, Annex A (dynamic factor A.2.3.2; "
       "per-span influence length A.2.3.5(2)).\n"
       "- **EN 1992-1-1** - concrete: the elastic modulus $E$ used in the *Eurocode ($f_{ck}$)* "
       "material option.\n"
       "- **Vejledning til belastnings- og beregningsgrundlag for broer** - classification "
       "factors (5.3.1), SLS dynamic-factor reduction (5.4.2), partial factors (Fig. B3.1 / "
       "B3.2, Bilag 3), transverse distribution for slab decks.")
    h1("Key assumptions & limitations")
    md("- 2D, linear-elastic; rigid connections on centrelines (no releases or eccentricities).\n"
       "- Single longitudinal strip ($b_{eff}$); no transverse load distribution - conservative "
       "for slabs (see *Transverse load distribution on slab bridges*).\n"
       "- One partial factor per permanent action (max and min); favourable/unfavourable not "
       "split automatically.\n"
       "- Prismatic forces are mesh-independent; non-prismatic forces and all deflections "
       "converge with mesh refinement.\n"
       "- Non-prismatic Timoshenko shear deformation is not included.")
    h1("Glossary")
    table(["Term", "Meaning"],
          [["$K_{FI}$", "Consequence-class factor"],
           ["$\\Phi$", "Dynamic amplification factor"],
           ["$\\Phi_s$", "Timoshenko shear parameter $12EI/(G A_s L^2)$"],
           ["$L_{inf}$", "Influence length used to determine $\\Phi$"],
           ["$b_{eff}$", "Effective width of the analysis strip"],
           ["$A_s$", "Shear area (Timoshenko)"],
           ["Consistent nodal loads", "Distributed loads converted to exact equivalent nodal forces"],
           ["Adverse-only", "A load placed only where it worsens the effect at each point"],
           ["ULS / SLS", "Ultimate / Serviceability Limit State"],
           ["S# / W#", "Span / Wall element identifiers"],
           ["Total Envelope", "Coupled, factored design envelope (vehicle + UDL + permanent)"]])

    return B


# ==========================================================================
# PDF RENDERER - same content blocks, rendered with ReportLab
# ==========================================================================
# The in-app content uses Markdown + LaTeX (great for Streamlit/KaTeX). For the
# PDF the small, known subset used here is converted to ReportLab's HTML-like
# inline markup. Figures are Plotly, exported to PNG (kaleido) like the report.

_GREEK = {
    '\\Phi': '&Phi;', '\\gamma': '&gamma;', '\\theta': '&theta;', '\\nu': '&nu;',
    '\\beta': '&beta;', '\\alpha': '&alpha;',
    '\\approx': '&approx;', '\\cdot': '&middot;', '\\times': '&times;',
    '\\le': '&le;', '\\leq': '&le;', '\\ge': '&ge;', '\\geq': '&ge;',
}


def _latex_to_rl(s):
    """Convert the small LaTeX subset used in the manual to ReportLab markup."""
    s = s.replace('\\,', ' ').replace('\\;', ' ')
    for k, v in _GREEK.items():
        s = s.replace(k, v)
    s = re.sub(r'_\{([^}]*)\}', r'<sub>\1</sub>', s)
    s = re.sub(r'\^\{([^}]*)\}', r'<super>\1</super>', s)
    s = re.sub(r'_([A-Za-z0-9])', r'<sub>\1</sub>', s)
    s = re.sub(r'\^([A-Za-z0-9])', r'<super>\1</super>', s)
    return s.replace('{', '').replace('}', '').replace('\\', '')


def _inline_md_to_rl(text):
    """Inline Markdown (**bold**, *italic*, $math$) -> ReportLab inline markup.
    Special characters are escaped first so the introduced tags stay valid."""
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'\$([^$]+)\$', lambda m: _latex_to_rl(m.group(1)), text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    return text


def _png_size(png):
    """(width, height) in pixels from a PNG byte string header."""
    return int.from_bytes(png[16:20], 'big'), int.from_bytes(png[20:24], 'big')


# A single hung kaleido export would otherwise block the whole PDF build - and
# the app's UI - forever, so each figure render runs with a hard timeout.
_FIG_EXPORT_TIMEOUT_S = 30.0
_FIG_TIMED_OUT = object()


def _fig_to_png(fig_callable, timeout=_FIG_EXPORT_TIMEOUT_S):
    """Render a manual figure to PNG bytes. Returns None if the render fails,
    and the ``_FIG_TIMED_OUT`` sentinel if it does not finish within ``timeout``.

    kaleido's ``write_image`` can block indefinitely when its headless-browser
    export server is in a bad state (no launchable Chrome/Edge, or a stale sync
    server left over from a previous export). Running it in a daemon thread with
    a join timeout guarantees the PDF still completes - with placeholders -
    instead of hanging the program.
    """
    out = {}

    def _work():
        try:
            buf = io.BytesIO()
            fig_callable().write_image(buf, format='png', scale=2)
            out['png'] = buf.getvalue()
        except Exception:
            out['png'] = None

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return _FIG_TIMED_OUT
    return out.get('png')


def _render_md_pdf(text, flow, styles, Paragraph):
    """Render a Markdown block (paragraphs, '- ' bullets, '1.' numbered, or a
    standalone $$display$$ formula) to ReportLab flowables."""
    t = text.strip()
    if t.startswith('$$') and t.endswith('$$'):
        flow.append(Paragraph(_latex_to_rl(t[2:-2].strip()), styles['MMath']))
        return
    buf = []

    def flush():
        if buf:
            flow.append(Paragraph(_inline_md_to_rl(' '.join(buf).strip()), styles['MBody']))
            buf.clear()

    for line in text.split('\n'):
        s = line.strip()
        if not s:
            flush()
            continue
        mb = re.match(r'^[-*]\s+(.*)', s)
        mn = re.match(r'^(\d+)\.\s+(.*)', s)
        if mb:
            flush()
            flow.append(Paragraph("&bull;&nbsp; " + _inline_md_to_rl(mb.group(1)), styles['MBody']))
        elif mn:
            flush()
            flow.append(Paragraph(f"{mn.group(1)}.&nbsp; " + _inline_md_to_rl(mn.group(2)), styles['MBody']))
        else:
            buf.append(s)
    flush()


def build_manual_pdf(buffer):
    """Render the manual to ``buffer`` as a PDF over the same content blocks."""
    import bricos_report as report
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                    Table, TableStyle, KeepTogether)

    styles = getSampleStyleSheet()

    def _add(name, **kw):
        if name not in styles.byName:
            styles.add(ParagraphStyle(name=name, parent=styles['Normal'], **kw))

    _add('MTitle', fontSize=20, spaceAfter=6, fontName='Helvetica-Bold')
    _add('MPart', fontSize=17, spaceBefore=18, spaceAfter=8, fontName='Helvetica-Bold',
         textColor=colors.HexColor('#0d2440'))
    _add('MH1', fontSize=15, spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold',
         textColor=colors.HexColor('#1f3b66'))
    _add('MH2', fontSize=12.5, spaceBefore=9, spaceAfter=4, fontName='Helvetica-Bold')
    _add('MH3', fontSize=11, spaceBefore=6, spaceAfter=3, fontName='Helvetica-Bold')
    _add('MBody', fontSize=9.5, leading=13, spaceAfter=4)
    _add('MMath', fontSize=11, leading=15, alignment=TA_CENTER, spaceBefore=6, spaceAfter=6)
    _add('MSmall', fontSize=8, leading=10, textColor=colors.grey)

    PAGE_W = 16.5 * cm

    flow = [
        Paragraph("BriCoS User Manual", styles['MTitle']),
        Paragraph(f"Version {data_mod.APP_VERSION}", styles['MSmall']),
        Spacer(1, 0.3 * cm),
        Paragraph("How BriCoS works, the theory it applies, its features and how to use it.",
                  styles['MBody']),
        Spacer(1, 0.4 * cm),
    ]

    # Figure export shares one kaleido server (see bricos_report); the loop
    # builds the flow, rendering each Plotly figure to PNG inside the context.
    with report.persistent_image_export():
        n1 = n2 = 0
        figures_hung = False  # set once a kaleido export times out; skip the rest
        for block in manual_blocks():
            kind = block[0]
            if kind == 'part':
                flow.append(Spacer(1, 0.3 * cm))
                flow.append(Paragraph(_inline_md_to_rl(block[1]), styles['MPart']))
            elif kind == 'h1':
                n1 += 1
                n2 = 0
                flow.append(Paragraph(f"{n1}. " + _inline_md_to_rl(_strip_num(block[1])), styles['MH1']))
            elif kind == 'h2':
                n2 += 1
                flow.append(Paragraph(f"{n1}.{n2} " + _inline_md_to_rl(_strip_num(block[1])), styles['MH2']))
            elif kind == 'h3':
                flow.append(Paragraph(_inline_md_to_rl(_strip_num(block[1])), styles['MH3']))
            elif kind == 'md':
                _render_md_pdf(block[1], flow, styles, Paragraph)
            elif kind == 'callout':
                _icon, ttl = _CALLOUT.get(block[1], ('', 'Note'))
                inner = Paragraph(f"<b>{ttl}:</b> " + _inline_md_to_rl(block[2]), styles['MBody'])
                t = Table([[inner]], colWidths=[PAGE_W])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eef2f7')),
                    ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#9fb3c8')),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
                flow.append(KeepTogether([t]))
                flow.append(Spacer(1, 0.15 * cm))
            elif kind == 'figure':
                png = None
                if not figures_hung:
                    png = _fig_to_png(block[1])
                    if png is _FIG_TIMED_OUT:
                        # A hung export means kaleido is wedged; skip the
                        # remaining figures rather than waiting out the timeout
                        # on each one, so the PDF still finishes promptly.
                        figures_hung = True
                        png = None
                if png:
                    w, h = _png_size(png)
                    img_w = PAGE_W
                    img_h = img_w * (h / w) if w else 8 * cm
                    flow.append(KeepTogether([
                        Image(io.BytesIO(png), width=img_w, height=img_h),
                        Paragraph(block[2], styles['MSmall'])]))
                else:
                    flow.append(Paragraph(f"[figure unavailable] {block[2]}", styles['MSmall']))
                flow.append(Spacer(1, 0.2 * cm))
            elif kind == 'table':
                headers, rows = block[1], block[2]
                ncol = len(headers)
                data = [[Paragraph(f"<b>{_inline_md_to_rl(h)}</b>", styles['MSmall']) for h in headers]]
                data += [[Paragraph(_inline_md_to_rl(str(c)), styles['MSmall']) for c in row] for row in rows]
                t = Table(data, colWidths=[PAGE_W / ncol] * ncol)
                t.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 0.4, colors.lightgrey),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eef2f7')),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
                flow.append(t)
                flow.append(Spacer(1, 0.2 * cm))

    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title=f"BriCoS User Manual v{data_mod.APP_VERSION}")
    doc.build(flow)


def build_manual_pdf_bytes():
    buf = io.BytesIO()
    build_manual_pdf(buf)
    buf.seek(0)
    return buf.getvalue()


# ==========================================================================
# STREAMLIT RENDERER
# ==========================================================================

def render_manual_streamlit():
    """Render the manual in the app. Called from the manual view in bricos_main."""
    # PDF download (generated on demand) - at the top of the manual.
    c1, c2, _c3 = st.columns([2, 2, 6])
    if c1.button("Generate PDF manual", icon=":material/picture_as_pdf:", key="manual_gen_pdf"):
        with st.spinner("Building the PDF manual..."):
            try:
                st.session_state['manual_pdf_bytes'] = build_manual_pdf_bytes()
            except Exception as e:
                st.session_state['manual_pdf_bytes'] = None
                st.error(f"PDF build failed: {e}")
    if st.session_state.get('manual_pdf_bytes'):
        c2.download_button("Download PDF", st.session_state['manual_pdf_bytes'],
                           file_name="BriCoS_User_Manual.pdf", mime="application/pdf",
                           icon=":material/download:", key="manual_dl_pdf")

    st.markdown("# :books: BriCoS user manual")
    st.caption("How BriCoS works, the theory it applies, its features, and how to use it.")

    n1 = n2 = 0
    for block in manual_blocks():
        kind = block[0]
        if kind == 'part':
            st.divider()
            st.markdown(f"# {block[1]}")
        elif kind == 'h1':
            n1 += 1
            n2 = 0
            st.markdown(f"## {n1}. {_strip_num(block[1])}")
        elif kind == 'h2':
            n2 += 1
            st.markdown(f"### {n1}.{n2} {_strip_num(block[1])}")
        elif kind == 'h3':
            st.markdown(f"#### {_strip_num(block[1])}")
        elif kind == 'md':
            st.markdown(block[1])
        elif kind == 'callout':
            icon, title = _CALLOUT.get(block[1], (":information_source:", "Note"))
            with st.container(border=True):
                st.markdown(f"{icon} **{title}** - {block[2]}")
        elif kind == 'figure':
            try:
                st.plotly_chart(block[1](), width='stretch')
            except Exception as e:  # a broken figure must not break the manual
                st.caption(f"[figure unavailable: {e}]")
            st.caption(block[2])
        elif kind == 'table':
            header = "| " + " | ".join(block[1]) + " |"
            sep = "| " + " | ".join(["---"] * len(block[1])) + " |"
            body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in block[2])
            st.markdown(f"{header}\n{sep}\n{body}")
