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

    md(f"*BriCoS v{data_mod.APP_VERSION} - user manual. Worked example: a 2-span "
       "integral frame bridge, carried through every section below.*")

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

    # ---- 3. Modelling concepts ------------------------------------------
    h1("3. Modelling concepts")
    h2("3.1 Systems A and B")
    md("BriCoS always holds two models. Define one and leave the other empty, or define both to "
       "compare them on the same diagrams (System A in blue, System B in red). The *Copy Data* "
       "tools clone one system into the other as a starting point.")
    h2("3.2 Model type")
    md("- **Frame** - decks (spans) on vertical walls/piers; used for integral/portal frames.\n"
       "- **Beam** - a continuous beam on point supports, no walls.\n"
       "- **Superstructure** - a deck on supports, for superstructure-only checks.")
    h2("3.3 Spans, walls, nodes and supports")
    md("Members are **spans** (`S1, S2, ...`, the horizontal deck elements) and **walls** "
       "(`W1, W2, ...`, the vertical members in Frame mode). Each member is meshed into "
       "sub-elements for the analysis. Supports are defined per node as fixed, pinned, roller "
       "or an elastic spring with stiffnesses $K_x$, $K_y$, $K_\\theta$.")
    call('theory', "Supports use the penalty method: fixed/pinned restraints are very stiff "
         "springs ($k \\approx 10^{14}$), and elastic foundations use your specified spring "
         "stiffnesses directly.")
    h2("3.4 Sections - the Section Profiler")
    md("Each member has a cross-section defined either by **height** $h$ (with the effective "
       "width $b_{eff}$, giving $I = b_{eff} h^3/12$) or directly by **inertia** $I$. The "
       "*Section Profiler* lets the depth vary along a member:")
    fig(fig_section_shapes, "Section depth shapes: constant, linear taper, and a 3-point "
        "(start/mid/end) profile. Tapers are integrated per sub-element.")
    call('concept', "$b_{eff}$ is the effective width of the analysis strip. Define loads as "
         "line loads on that strip; BriCoS does not spread loads transversely - you choose the "
         "width that your loads and section refer to.")
    h2("3.5 Alignment")
    md("Spans can be inclined (by slope % or end-to-end height difference), so skewed or ramped "
       "decks can be modelled. Inclination changes how axial and shear resolve in the member's "
       "local axes (below).")

    # ---- 4. Loads --------------------------------------------------------
    h1("4. Loads")
    md("BriCoS separates **permanent** actions from **variable (traffic)** actions so they can "
       "be factored independently in the combinations.")
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
    h2("4.1 Self-weight: manual vs automatic")
    md("By default you enter all permanent load as the **Dead Load** line load. Turn on "
       "**auto self-weight** to have BriCoS compute the structural self-weight as "
       "$\\gamma \\cdot b_{eff} \\cdot h$ per sub-element from a unit weight $\\gamma$; the "
       "Dead Load then carries only superimposed actions. Auto self-weight requires "
       "height-defined sections.")
    h2("4.2 Traffic UDL")
    md("The Traffic UDL is a line load applied **adverse-only** - placed only where it worsens "
       "the effect at each result point - per EN 1991-2:2003, 4.3.2(1)(b).")
    h2("4.3 Vehicles")
    md("A vehicle is a set of axle loads and spacings (standard classification vehicles are "
       "available, or define your own). BriCoS steps the vehicle across the structure to build "
       "the moving-load envelope. Two vehicles (A and B) and travel direction (forward / reverse "
       "/ both) are supported.")
    call('standard', "Standard vehicle classes follow the Danish classification system; the "
         "partial factor on the vehicle (e.g. $\\gamma_Q = 1.4$) follows *Vejledning* 5.3.1 / "
         "Bilag 3.")

    # ---- 5. Theory & methodology ----------------------------------------
    h1("5. Theory & methodology")
    h2("5.1 Finite-element formulation")
    md("BriCoS is a 2D **matrix-stiffness** solver. Prismatic members use the "
       "**Euler-Bernoulli** beam element, or **Timoshenko** when shear deformation is enabled "
       "(via $\\Phi_s = 12EI/(GA_s L^2)$). Non-prismatic members are integrated with "
       "displacement-based Euler-Bernoulli over $EI(x)$ per sub-element. Material behaviour is "
       "linear-elastic. The global stiffness $K$ is assembled from the element stiffnesses and "
       "solved against the load vectors.")
    call('theory', "For prismatic members the internal forces $M, V, N$ are **mesh-independent** "
         "(exact fixed-end contributions are included). For non-prismatic members the section "
         "variation is represented per sub-element, so results **converge** as the mesh is "
         "refined rather than being exact at any mesh size.")
    h2("5.2 Deflections")
    md("Deflections are interpolated between nodes from the nodal displacements with Hermite "
       "shape functions. The local deflection of a load acting *between* nodes is not added; the "
       "resulting underestimate is bounded by about $P\\,L_{mesh}^3/(192EI)$ per point load and "
       "vanishes as the mesh is refined.")
    h2("5.3 Moving loads")
    md("Traffic is evaluated **quasi-statically**: the vehicle is stepped across the structure "
       "and the absolute max/min envelopes of every effect are accumulated. Where both Vehicle A "
       "and B are defined, their envelopes are superposed independently (maxima may be combined "
       "even if they occur at different positions).")
    h2("5.4 Sign convention & local axes")
    fig(fig_sign_convention, "Bending is sagging-positive and plotted on the tension side; "
        "$+N$ is tension. Internal-force signs follow the member's local axes.")
    md("- **Spans (horizontal):** local x aligns with global X, so $N$ is horizontal axial force "
       "and $V$ is vertical shear.\n"
       "- **Walls (vertical):** local x runs along the member, so $N$ is the vertical axial load "
       "and $V$ is the horizontal shear.")
    h2("5.5 Equilibrium check")
    md("After solving, BriCoS sums the applied loads against the support reactions per load case "
       "and reports the residual as a built-in QA check (shown in the report).")

    # ---- 6. Load combinations & standards -------------------------------
    h1("6. Load combinations & standards")
    md("Design effects are built by superposing **factored envelopes**:")
    md("$$E_d = K_{FI}\\,(\\gamma_G E_{SW} + \\gamma_{Soil} E_{Soil} + "
       "\\gamma_Q \\Phi\\, E_{Veh} + \\gamma_Q E_{Surch})$$")
    md("where $K_{FI}$ is the consequence-class factor and the $\\gamma$ are the partial "
       "factors set per system. The **Unfactored** combination (all loads x 1.0, no dynamic "
       "factor) is always available; if neither ULS nor SLS is selected it is the only result.")
    h2("6.1 Partial factors (ULS / SLS)")
    md("ULS uses the design partial factors (x $K_{FI}$); SLS uses the characteristic "
       "combination factors. The SLS set follows *Vejledning* Fig. B3.2 (e.g. Vehicle B 0.75, "
       "Traffic UDL 0.40).")
    h2("6.2 Dynamic factor $\\Phi$")
    md("Moving-vehicle effects are amplified by the dynamic factor $\\Phi$, calculated from the "
       "influence length (or set manually). The influence length is taken either per span "
       "(DK NA A.2.3.5(2)) or for the combined system (EN 1991-2:2003, Table 6.2); $\\Phi$ can be "
       "applied per member or as a single governing value, and reduced for SLS "
       "($1 + (\\Phi-1)/2$, *Vejledning* 5.4.2).")
    fig(fig_phi_curve, "Dynamic factor vs influence length (the form used by BriCoS): 1.25 up to "
        "5 m, decreasing to 1.05 at 50 m.")
    call('standard', "The Traffic UDL is **not** amplified by $\\Phi$ - its intensity already "
         "includes the dynamic increment (DK NA A.2.3.2).")
    h2("6.3 The coupled Total Envelope")
    md("When a vehicle and a Traffic UDL coexist, BriCoS does not simply add their separate "
       "envelopes. For each vehicle position it combines the factored vehicle effect with the "
       "factored **adverse UDL outside that position's clear-distance window**, and envelopes "
       "that together with the vehicle-absent situation (full adverse UDL alone). This coupled "
       "result is the **Total Envelope**.")
    fig(fig_moment_envelope, "ULS bending Total Envelope of the worked example - the coupled "
        "vehicle + UDL + permanent envelope used for design.")

    # ---- 7. Results & visualization -------------------------------------
    h1("7. Results & visualization")
    md("The main panel shows four diagrams - **Bending Moment, Shear, Normal Force, "
       "Deflection** - for the selected **Load Case** and **Result Type** (Design (ULS), "
       "Characteristic (SLS), or Unfactored). The *Total Envelope* case is the combined design "
       "result; individual cases (Dead Load, Soil, Surcharge, Traffic UDL, Vehicle Envelope, "
       "Self-weight) are available for checking.")
    fig(fig_deflection_envelope, "SLS deflection envelope of the worked example.")
    h2("7.1 Visual toggles")
    md("- **Labels** - value labels at the diagram extremes.\n"
       "- **Show Supports** - support symbols at the nodes.\n"
       "- **Section Depth** - overlays each member's true depth to scale.\n"
       "- **Element Names** - labels each member (S1, W2, ...) in the system colour; a single "
       "neutral label is shown where A and B coincide.\n"
       "- **Target Diagram Height** - scales the diagrams (and the element-label size).")
    h2("7.2 Vehicle step viewer")
    md("For a moving vehicle you can step through positions and see the effect at each step, "
       "and jump to the critical position per member - useful for understanding which vehicle "
       "location governs.")
    h2("7.3 Reactions")
    md("Support reactions ($R_x, R_y, M_z$) are tabulated per node for the selected result.")

    # ---- 8. Reports & export --------------------------------------------
    h1("8. Reports & data export")
    md("**PDF report** (*Report Generation*): a full document with the basis of analysis, "
       "global settings, per-system input summary (including an element-layout diagram), the "
       "ULS/SLS Total Envelopes, per-component results, the equilibrium check and an appendix. "
       "If only one system is validly defined, the report covers that system alone and notes the "
       "exclusion.")
    md("**Data package** (*Tabular Data*): an Excel/CSV export with a settings sheet and one "
       "sheet per load case plus the total envelopes, for independent verification.")

    # ---- 9. Session management ------------------------------------------
    h1("9. Saving & loading")
    md("*File Operations* downloads the full configuration as a CSV you can re-upload later. "
       "BriCoS also autosaves the session periodically (configurable), so work is not lost on a "
       "refresh.")

    # ---- 10. Reference ---------------------------------------------------
    h1("10. Reference")
    h2("10.1 Standards")
    md("- **EN 1991-2:2003** - traffic loads on bridges (UDL application 4.3.2(1)(b); dynamic "
       "factor Table 6.2).\n"
       "- **EN 1991-2 DK NA:2017** - Danish National Annex, Annex A (dynamic factor A.2.3.2; "
       "per-span influence length A.2.3.5(2)).\n"
       "- **Vejledning til belastnings- og beregningsgrundlag for broer** - classification "
       "factors (5.3.1), SLS dynamic-factor reduction (5.4.2), partial factors (Fig. B3.1 / "
       "B3.2, Bilag 3).")
    h2("10.2 Key assumptions")
    md("- 2D, linear-elastic; rigid connections on centrelines.\n"
       "- One partial factor per permanent action (max and min); favourable/unfavourable not "
       "split automatically.\n"
       "- Non-prismatic results converge with mesh refinement; prismatic results are "
       "mesh-independent.\n"
       "- Non-prismatic Timoshenko shear deformation is not included.")
    h2("10.3 Glossary")
    table(["Term", "Meaning"],
          [["$K_{FI}$", "Consequence-class factor"],
           ["$\\Phi$", "Dynamic amplification factor"],
           ["$b_{eff}$", "Effective width of the analysis strip"],
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


def _fig_to_png(fig_callable):
    try:
        buf = io.BytesIO()
        fig_callable().write_image(buf, format='png', scale=2)
        return buf.getvalue()
    except Exception:
        return None


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
        for block in manual_blocks():
            kind = block[0]
            if kind == 'h1':
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
                png = _fig_to_png(block[1])
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
        if kind == 'h1':
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
