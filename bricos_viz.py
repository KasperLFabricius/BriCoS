import plotly.graph_objects as go
import numpy as np
import bricos_kernels as kernels

# ==========================================
# ANNOTATION SOLVER & HELPERS
# ==========================================

def solve_annotations(annotations, extent_x=None, font_scale=1.0):
    """
    Optimizes label placement to prevent overlaps using a rigid-body physics approach.
    """
    if not annotations: return []
    n = len(annotations)
    data_arr = np.zeros((n, 6))

    # Label footprint in DATA units. Labels render at a fixed pixel size, so
    # their data-unit size grows with the plotted extent; the legacy
    # constants (0.15 per char, 0.40 high) match a ~15 m structure at font
    # scale 1 and understate the footprint on longer decks, letting the
    # solver leave labels stacked on top of each other.
    if extent_x is not None and extent_x > 1e-6:
        char_w = 0.0094 * extent_x * font_scale
        box_h = 0.025 * extent_x * font_scale
    else:
        char_w, box_h = 0.15, 0.40

    # Pack data for Numba kernel
    for i, ann in enumerate(annotations):
        ann['w'] = len(ann['text']) * char_w
        ann['h'] = box_h
        data_arr[i, :] = [ann['x'], ann['y'], ann['w'], ann['h'], ann['perp_x'], ann['perp_y']]
        
    # Run the solver (Pure Math)
    result_arr = kernels.jit_annotation_solver(data_arr)
    
    # Unpack results
    for i, ann in enumerate(annotations):
        ann['x'] = result_arr[i, 0]
        ann['y'] = result_arr[i, 1]
    return annotations

def structure_extent_x(sources):
    """Horizontal extent [m] spanned by the element dicts' ni/nj global
    coordinates, or None when no usable geometry is present."""
    xs = []
    for src in sources:
        if not src:
            continue
        for dat in src.values():
            if isinstance(dat, dict) and 'ni' in dat and 'nj' in dat:
                xs.append(dat['ni'][0])
                xs.append(dat['nj'][0])
    if not xs:
        return None
    ext = max(xs) - min(xs)
    return ext if ext > 1e-6 else None


def _add_support_icon(fig, x, y, supp_type, size, color='black'):
    """
    Helper to draw classical boundary condition icons at (x,y).
    """
    s = size
    line_width = 3.0  # Thicker for better report visibility
    
    # 1. FIXED SUPPORT
    if supp_type == 'Fixed':
        # Main rigid plate
        fig.add_trace(go.Scatter(
            x=[x - s, x + s], 
            y=[y, y],
            mode='lines',
            line=dict(color=color, width=line_width),
            hoverinfo='skip', showlegend=False
        ))
        # Hatching
        h_spacing = (2 * s) / 4.0
        h_height = s * 0.6
        for i in range(5):
            hx_start = (x - s) + i * h_spacing
            fig.add_trace(go.Scatter(
                x=[hx_start, hx_start - h_height * 0.5],
                y=[y, y - h_height],
                mode='lines',
                line=dict(color=color, width=1.5),
                hoverinfo='skip', showlegend=False
            ))

    # 2. PINNED SUPPORT
    elif supp_type == 'Pinned':
        fig.add_trace(go.Scatter(
            x=[x, x - s/1.5, x + s/1.5, x],
            y=[y, y - s, y - s, y],
            mode='lines',
            fill='toself', fillcolor='white', 
            line=dict(color=color, width=line_width),
            hoverinfo='skip', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(color='white', line=dict(color=color, width=1.5), size=4),
            hoverinfo='skip', showlegend=False
        ))

    # 3. ROLLER (X-Free)
    elif supp_type == 'Roller (X-Free)':
        fig.add_trace(go.Scatter(
            x=[x, x - s/1.5, x + s/1.5, x],
            y=[y, y - s, y - s, y],
            mode='lines',
            fill='toself', fillcolor='white',
            line=dict(color=color, width=line_width),
            hoverinfo='skip', showlegend=False
        ))
        wheel_r = s * 0.2
        wheel_y = y - s - wheel_r
        fig.add_trace(go.Scatter(
            x=[x - s/3.0, x + s/3.0],
            y=[wheel_y, wheel_y],
            mode='markers',
            marker=dict(symbol='circle-open', color=color, size=6, line=dict(width=1.5)),
            hoverinfo='skip', showlegend=False
        ))
        ground_y = wheel_y - wheel_r
        fig.add_trace(go.Scatter(
            x=[x - s, x + s],
            y=[ground_y, ground_y],
            mode='lines',
            line=dict(color=color, width=1.5),
            hoverinfo='skip', showlegend=False
        ))
    
    # 4. ROLLER (Y-Free)
    elif supp_type == 'Roller (Y-Free)':
        fig.add_trace(go.Scatter(
            x=[x, x], y=[y, y], 
            mode='markers', 
            marker=dict(symbol='square-open', color=color, size=10, line=dict(width=2)),
            hoverinfo='skip', showlegend=False
        ))
        fig.add_trace(go.Scatter(
             x=[x - s/2, x - s/2], y=[y - s, y + s],
             mode='lines', line=dict(color=color, width=1.5),
             hoverinfo='skip', showlegend=False
        ))
        fig.add_trace(go.Scatter(
             x=[x + s/2, x + s/2], y=[y - s, y + s],
             mode='lines', line=dict(color=color, width=1.5),
             hoverinfo='skip', showlegend=False
        ))

    # 5. CUSTOM
    else:
        fig.add_trace(go.Scatter(
            x=[x - s/2, x + s/2, x + s/2, x - s/2, x - s/2],
            y=[y, y, y - s, y - s, y],
            mode='lines',
            line=dict(color=color, width=line_width, dash='dot'),
            hoverinfo='skip', showlegend=False
        ))

# ==========================================
# SECTION-DEPTH (THICKNESS) OVERLAY
# ==========================================

def _section_height_at(value, val_type, b_eff):
    """Section depth h at a sampled section value. Height input is used
    directly; an inertia input is mapped to the equivalent rectangular depth
    h = (12 I / b_eff)^(1/3) (only reached defensively - the UI gates the
    overlay off whenever any section is inertia-defined)."""
    v = max(float(value), 0.0)
    if val_type == 0:
        be = b_eff if b_eff > 1e-6 else 1.0
        return (12.0 * v / be) ** (1.0 / 3.0)
    return v


def section_band_polygon(ni, nj, L, val_type, shape, vals, b_eff, n_samples=3):
    """Closed polygon (xs, ys) of a member's true section depth, centred on the
    member axis ni->nj. The depth h(x) is offset +-h/2 along the member normal,
    so the band follows the centreline and is drawn to real geometric scale
    (the figure uses equal aspect). Returns ([], []) for a degenerate member.
    Pure helper - no Plotly - so it is unit-testable.
    """
    L = float(L)
    if L <= 1e-9:
        return [], []
    dx, dy = (nj[0] - ni[0]) / L, (nj[1] - ni[1]) / L
    nx, ny = -dy, dx  # unit normal to the member axis
    n = max(int(n_samples), 2)
    # Typed array matches the solver's call convention (a Python list triggers
    # Numba's reflected-list deprecation warning).
    vals_arr = np.asarray(vals, dtype=np.float64)
    top_x, top_y, bot_x, bot_y = [], [], [], []
    for i in range(n):
        f = i / (n - 1)
        h = _section_height_at(
            kernels.get_section_value_at_x(f * L, L, vals_arr, int(shape)),
            val_type, b_eff)
        px = ni[0] + (nj[0] - ni[0]) * f
        py = ni[1] + (nj[1] - ni[1]) * f
        top_x.append(px + nx * h / 2.0); top_y.append(py + ny * h / 2.0)
        bot_x.append(px - nx * h / 2.0); bot_y.append(py - ny * h / 2.0)
    xs = top_x + bot_x[::-1] + [top_x[0]]
    ys = top_y + bot_y[::-1] + [top_y[0]]
    return xs, ys


def _section_profile_for(params, eid):
    """(val_type, shape, vals) for a parent element id (e.g. 'S1', 'W2') from
    the system params, falling back to the simple height list when no Section
    Profiler geometry is stored."""
    idx = int(eid[1:]) - 1
    if eid.startswith('S'):
        geom, fallback = params.get(f'span_geom_{idx}'), params.get('Is_list', [])
    else:
        geom, fallback = params.get(f'wall_geom_{idx}'), params.get('Iw_list', [])
    if isinstance(geom, dict) and geom.get('vals'):
        return int(geom.get('type', 1)), int(geom.get('shape', 0)), [float(v) for v in geom['vals']]
    fv = float(fallback[idx]) if 0 <= idx < len(fallback) else 0.0
    return 1, 0, [fv, fv, fv]


def _draw_thickness_bands(fig, geom_source, params, is_sys_A, legend_flags):
    """Overlay each member's true section depth as a tinted greyscale band,
    drawn beneath the result diagrams. Geometry (axis, length) comes from the
    Dead Load result; the depth profile from the system params."""
    if not geom_source or not params:
        return
    b_eff = float(params.get('b_eff', 1.0) or 1.0)
    # System A slightly darker than B so overlapping bands stay distinguishable.
    fillcolor = 'rgba(60,60,60,0.16)' if is_sys_A else 'rgba(130,130,130,0.16)'
    for eid in sorted(geom_source.keys(), key=lambda x: (x[0], int(x[1:]))):
        dat = geom_source[eid]
        if 'ni' not in dat or 'nj' not in dat:
            continue
        val_type, shape, vals = _section_profile_for(params, eid)
        xs, ys = section_band_polygon(
            dat['ni'], dat['nj'], dat.get('L', 0.0), val_type, shape, vals, b_eff)
        if not xs:
            continue
        show_legend = not legend_flags.get('thickness', False)
        legend_flags['thickness'] = True
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill='toself', fillcolor=fillcolor,
            line=dict(width=0), mode='none', hoverinfo='skip',
            name="Section depth", showlegend=show_legend
        ))


def _element_coords_match(da, db, tol=1e-6):
    """True when two element definitions share the same start/end node
    coordinates, i.e. the element occupies the same place in both systems."""
    for key in ('ni', 'nj'):
        pa, pb = da.get(key), db.get(key)
        if pa is None or pb is None:
            return False
        if abs(pa[0] - pb[0]) > tol or abs(pa[1] - pb[1]) > tol:
            return False
    return True


def _label_at(fig, dat, text, color, font_size):
    """Draw one element-name chip: white bold text on a solid colour fill. The
    filled-chip style is deliberately distinct from the section-force value
    labels (white box, coloured text) so the two are not confused when both are
    shown. Spans are nudged just above their axis and walls just to the side so
    the chip does not sit directly on the member line."""
    if 'ni' not in dat or 'nj' not in dat:
        return
    ni, nj = dat['ni'], dat['nj']
    mx = (ni[0] + nj[0]) / 2.0
    my = (ni[1] + nj[1]) / 2.0
    is_wall = str(text).startswith('W')
    nudge = max(6.0, 0.9 * font_size)
    fig.add_annotation(
        x=mx, y=my, text=text, showarrow=False,
        xshift=(nudge if is_wall else 0.0),
        yshift=(0.0 if is_wall else nudge),
        font=dict(color='white', size=font_size, family="Arial", weight="bold"),
        bgcolor=color, bordercolor=color, borderwidth=0, borderpad=3
    )


def _draw_element_labels(fig, geom_source, color, font_size):
    """Element-name chips for a single system, in its colour. A QA aid toggled
    from the UI."""
    if not geom_source:
        return
    for eid in sorted(geom_source.keys(), key=lambda x: (x[0], int(x[1:]))):
        _label_at(fig, geom_source[eid], eid, color, font_size)


def _draw_element_labels_overlay(fig, gsrc_a, gsrc_b, font_size, tol=1e-6):
    """Element-name chips for an overlaid A/B plot, decided per element: an
    element that occupies the same place in both systems gets a single neutral
    chip (it overlaps, so one identifier is enough); an element that differs or
    exists in only one system keeps that system's colour (blue = A, red = B).
    This way W1 is merged when it coincides even if S1 (and hence W2) is offset
    between the systems."""
    gsrc_a = gsrc_a or {}
    gsrc_b = gsrc_b or {}
    all_ids = sorted(set(gsrc_a) | set(gsrc_b), key=lambda x: (x[0], int(x[1:])))
    for eid in all_ids:
        da, db = gsrc_a.get(eid), gsrc_b.get(eid)
        a_ok = bool(da and 'ni' in da and 'nj' in da)
        b_ok = bool(db and 'ni' in db and 'nj' in db)
        if a_ok and b_ok and _element_coords_match(da, db, tol):
            _label_at(fig, da, eid, '#333333', font_size)
        else:
            if a_ok:
                _label_at(fig, da, eid, 'blue', font_size)
            if b_ok:
                _label_at(fig, db, eid, 'red', font_size)


# ==========================================
# MAIN PLOTTING FUNCTION
# ==========================================

def create_plotly_fig(
    nodes, sysA_data, sysB_data, type_base='M', target_height=2.0, title="", 
    show_A=True, show_B=True, annotate=True, load_case_name="", 
    name_A="System A", name_B="System B", 
    geom_A=None, geom_B=None,
    params_A=None, params_B=None,
    show_supports=False, support_size=0.5,
    font_scale=1.0,
    nodes_A=None, nodes_B=None,
    show_thickness=False,
    show_element_names=False,
    structure_only=False
):
    fig = go.Figure()
    
    # Safety Defaults
    if params_A is None: params_A = {}
    if params_B is None: params_B = {}
    if nodes_A is None: nodes_A = nodes
    if nodes_B is None: nodes_B = nodes
    
    # --- 1. DETERMINE SCALING FACTOR (DIAGRAMS) ---
    max_val = 0.0
    for data_set in [sysA_data, sysB_data]:
        if not data_set: continue
        for eid, d in data_set.items():
            keys_to_scan = []
            if type_base == 'Def':
                if eid.startswith('W'): keys_to_scan = ['def_x', 'def_x_max', 'def_x_min']
                else: keys_to_scan = ['def_y', 'def_y_max', 'def_y_min']
            elif type_base == 'M': keys_to_scan = ['M', 'M_max', 'M_min']
            elif type_base == 'V': keys_to_scan = ['V', 'V_max', 'V_min']
            elif type_base == 'N': keys_to_scan = ['N', 'N_max', 'N_min']
            
            for k in keys_to_scan:
                if k in d:
                    # Sanitize before checking max
                    raw_v = np.nan_to_num(d[k], nan=0.0, posinf=0.0, neginf=0.0)
                    val = np.max(np.abs(raw_v))
                    if type_base == 'Def': val *= 1000.0 
                    max_val = max(max_val, val)
    
    scale = 1.0
    if max_val > 1e-5: scale = target_height / max_val

    # --- 2. DETERMINE SCALING FACTOR (LOADS) ---
    max_P_veh = 1.0
    max_q_sw = 1.0
    max_q_soil = 1.0
    max_q_surch = 1.0

    for data_set in [sysA_data, sysB_data]:
        if not data_set: continue
        for d in data_set.values():
            if 'loads' in d:
                for load in d['loads']:
                    l_type = load.get('type', '')
                    params = load.get('params', [])
                    
                    if not params: continue
                    if l_type == 'point': 
                        max_P_veh = max(max_P_veh, abs(params[0]))
                    elif l_type == 'distributed_trapezoid':
                        q_max = max(abs(params[0]), abs(params[1])) if len(params) > 1 else abs(params[0])
                        if load.get('is_gravity', False): max_q_sw = max(max_q_sw, q_max)
                        elif load_case_name == "Surcharge": max_q_surch = max(max_q_surch, q_max)
                        else: max_q_soil = max(max_q_soil, q_max)

    unit_map = {'M': 'kNm', 'V': 'kN', 'N': 'kN', 'Def': 'mm'}
    unit = unit_map.get(type_base, '')
    
    ann_candidates = []
    legend_flags = {'struct': False, 'A': False, 'B': False, 'thickness': False}

    # --- FONT SIZING & LAYOUT ADJUSTMENTS ---
    base_font_size = 10 * font_scale
    marker_size = 8 * font_scale
    veh_load_font_size = 10 * (1 + (font_scale - 1) * 0.4) 
    margin_base = 30
    top_margin = 40 
    
    # --- DRAW SUPPORTS ---
    if show_supports:
        def render_system_supports(params, sys_nodes_dict, color_override):
            if not params: return
            supp_list = params.get('supports', [])
            mode = params.get('mode', 'Frame')
            num_spans = params.get('num_spans', 1)
            num_supp = num_spans + 1
            
            base_idx = 100 if mode == 'Frame' else 200
            
            for i in range(num_supp):
                nid = base_idx + i
                pos_x, pos_y = None, None
                
                # Check if we have valid nodes for this system
                if sys_nodes_dict and nid in sys_nodes_dict:
                    pos_x, pos_y = sys_nodes_dict[nid]
                else:
                    # Fallback logic removed/guarded by caller check on geom availability
                    # But if we are here, we attempt fallback in case node dict is partial
                    current_x = 0.0
                    L_list = params.get('L_list', [])
                    for span_i in range(i):
                        if span_i < len(L_list): current_x += L_list[span_i]
                    current_y = 0.0
                    if mode == 'Frame':
                        h_list = params.get('h_list', [])
                        if i < len(h_list): current_y = -h_list[i]
                    pos_x, pos_y = current_x, current_y
                
                s_type = 'Fixed'
                if i < len(supp_list):
                    s_type = supp_list[i].get('type', 'Fixed')
                else:
                    if mode == 'Frame': s_type = 'Fixed'
                    else: s_type = 'Pinned' if i==0 else 'Roller (X-Free)'
                
                _add_support_icon(fig, pos_x, pos_y, s_type, support_size, color_override)

        # UPDATED: Only draw supports if geometry data (geom_A/geom_B) is present and non-empty.
        # geom_A/B are typically the 'Dead Load' result dictionaries containing element definitions.
        # If the solver returns an error state, these will be empty.
        
        if show_A and params_A and geom_A: 
             render_system_supports(params_A, nodes_A, 'blue')
             
        if show_B and params_B and geom_B: 
             render_system_supports(params_B, nodes_B, 'red')

    def add_traces(sys_data, sys_name, color, line_style):
        if not sys_data: return
        is_sys_A = (sys_name == name_A)
        sys_key = 'A' if is_sys_A else 'B'
        geom_source = geom_A if (is_sys_A and geom_A) else (geom_B if (not is_sys_A and geom_B) else sys_data)

        # 1. Structure Geometry Trace (Background)
        x_struct, y_struct = [], []
        if geom_source:
            sorted_ids = sorted(geom_source.keys(), key=lambda x: (x[0], int(x[1:])))
            for eid in sorted_ids:
                if eid not in geom_source: continue
                dat = geom_source[eid]
                ni, nj = dat['ni'], dat['nj']
                x_struct.extend([ni[0], nj[0], None])
                y_struct.extend([ni[1], nj[1], None])
        
        show_struct = False
        if not legend_flags['struct'] and x_struct:
            show_struct = True
            legend_flags['struct'] = True
            
        fig.add_trace(go.Scatter(
            x=x_struct, y=y_struct, mode='lines+markers', 
            line=dict(color='grey', width=3 if is_sys_A else 1.5),
            marker=dict(size=4, color='grey'),
            name="Structure Geometry", opacity=0.5,
            hoverinfo='skip', showlegend=show_struct
        ))

        # Structure-only mode (e.g. the report's element-layout diagram): draw
        # the geometry and stop before any result diagram or load arrows.
        if structure_only:
            return

        # 2. Result Traces Aggregation
        # We accumulate all coordinates into single lists (separated by None) to create ONE trace per type.
        # This significantly improves Plotly rendering performance.
        
        # Accumulators
        X_pos_list, Y_pos_list = [], []
        X_neg_list, Y_neg_list = [], []
        C_pos_list, C_neg_list = [], [] # customdata
        
        # Fill Accumulators (Closed polygons for fill)
        X_fill_list, Y_fill_list = [], []
        
        is_envelope = False
        sorted_eids = sorted(sys_data.keys(), key=lambda x: (x[0], int(x[1:])))
        
        # Prepare "None" break arrays
        nan_pt = np.array([None])
        nan_cust = np.array([[None, None]])

        htemp_max = f"<b>{sys_name} (Max)</b><br>Loc: %{{customdata[0]:.2f}} m<br>Val: %{{customdata[1]:.1f}} {unit}<extra></extra>"
        htemp_min = f"<b>{sys_name} (Min)</b><br>Loc: %{{customdata[0]:.2f}} m<br>Val: %{{customdata[1]:.1f}} {unit}<extra></extra>"
        htemp_step = f"<b>{sys_name}</b><br>Loc: %{{customdata[0]:.2f}} m<br>Val: %{{customdata[1]:.1f}} {unit}<extra></extra>"

        for eid in sorted_eids:
            data = sys_data[eid]
            if 'x' not in data: continue 
            
            L = data['L']
            c, s = data['cx'], data['cy']
            ni = data['ni']
            x_local = data['x']
            x_glob = ni[0] + c * x_local
            y_glob = ni[1] + s * x_local
            
            vals_pos = None; vals_neg = None; fill_mode = False
            inv = 1.0 
            
            if type_base == 'Def':
                if eid.startswith('W'): key_base, inv = 'def_x', -1.0
                else: key_base, inv = 'def_y', 1.0
                
                if f'{key_base}_max' in data:
                    vals_pos = data[f'{key_base}_max'] * 1000
                    vals_neg = data[f'{key_base}_min'] * 1000
                    fill_mode = True
                else:
                    vals_pos = data[key_base] * 1000
                    vals_neg = vals_pos
                    fill_mode = False
            else:
                key = type_base
                if type_base == 'M':
                    # Sagging-positive sign convention (v0.58): keep drawing
                    # the moment diagram on the tension side (sagging below
                    # the member), as before the flip.
                    inv = -1.0
                if f'{key}_max' in data:
                    vals_pos = data[f'{key}_max']
                    vals_neg = data[f'{key}_min']
                    fill_mode = True
                else:
                    vals_pos = data[key]
                    vals_neg = vals_pos
                    fill_mode = False

            # Sanitization
            vals_pos = np.nan_to_num(vals_pos, nan=0.0, posinf=0.0, neginf=0.0)
            vals_neg = np.nan_to_num(vals_neg, nan=0.0, posinf=0.0, neginf=0.0)

            nx, ny = -s, c
            x_plot_pos = x_glob + nx * vals_pos * scale * inv
            y_plot_pos = y_glob + ny * vals_pos * scale * inv
            x_plot_neg = x_glob + nx * vals_neg * scale * inv
            y_plot_neg = y_glob + ny * vals_neg * scale * inv
            
            custom_pos = np.stack((x_local, vals_pos), axis=-1)
            custom_neg = np.stack((x_local, vals_neg), axis=-1)
            
            # --- ACCUMULATE TRACE DATA ---
            # Max Line (or Single Line)
            X_pos_list.append(x_plot_pos); X_pos_list.append(nan_pt)
            Y_pos_list.append(y_plot_pos); Y_pos_list.append(nan_pt)
            C_pos_list.append(custom_pos); C_pos_list.append(nan_cust)

            if fill_mode:
                is_envelope = True
                # Min Line
                X_neg_list.append(x_plot_neg); X_neg_list.append(nan_pt)
                Y_neg_list.append(y_plot_neg); Y_neg_list.append(nan_pt)
                C_neg_list.append(custom_neg); C_neg_list.append(nan_cust)
                
                # Fill Polygon (Max -> Min Reverse)
                poly_x = np.concatenate([x_plot_pos, x_plot_neg[::-1]])
                poly_y = np.concatenate([y_plot_pos, y_plot_neg[::-1]])
                X_fill_list.append(poly_x); X_fill_list.append(nan_pt)
                Y_fill_list.append(poly_y); Y_fill_list.append(nan_pt)
            else:
                # Step Fill (Geometry -> Value Reverse)
                poly_x = np.concatenate([x_glob, x_plot_pos[::-1]])
                poly_y = np.concatenate([y_glob, y_plot_pos[::-1]])
                X_fill_list.append(poly_x); X_fill_list.append(nan_pt)
                Y_fill_list.append(poly_y); Y_fill_list.append(nan_pt)
            
            # --- ANNOTATION CANDIDATES ---
            # (Annotations are sparse, so we keep this logic here)
            if annotate:
                threshold = max_val * 0.05
                if fill_mode:
                    idx_max = np.argmax(vals_pos)
                    idx_min = np.argmin(vals_neg)
                    if abs(vals_pos[idx_max]) > threshold:
                        ann_candidates.append({
                            'x': x_plot_pos[idx_max], 'y': y_plot_pos[idx_max], 
                            'text': f"{vals_pos[idx_max]:.1f}", 'color': color,
                            'perp_x': nx, 'perp_y': ny
                        })
                    if abs(vals_neg[idx_min]) > threshold and idx_min != idx_max:
                        ann_candidates.append({
                            'x': x_plot_neg[idx_min], 'y': y_plot_neg[idx_min], 
                            'text': f"{vals_neg[idx_min]:.1f}", 'color': color,
                            'perp_x': nx, 'perp_y': ny
                        })
                else:
                    vals_abs = np.abs(vals_pos)
                    idx = np.argmax(vals_abs)
                    val = vals_pos[idx]
                    if abs(val) > threshold:
                         ann_candidates.append({
                            'x': x_plot_pos[idx], 'y': y_plot_pos[idx], 
                            'text': f"{val:.1f}", 'color': color,
                            'perp_x': nx, 'perp_y': ny
                        })

            # --- TRAFFIC UDL EXTENT (step views) ---
            # Translucent band along the deck regions carrying the UDL for
            # this step; the window around the vehicle stays visibly empty.
            if load_case_name == "Vehicle Steps" and data.get('udl_loaded_ranges'):
                band_h = 0.45
                first_band = True
                for (r_a, r_b) in data['udl_loaded_ranges']:
                    ax0 = ni[0] + c * r_a; ay0 = ni[1] + s * r_a
                    ax1 = ni[0] + c * r_b; ay1 = ni[1] + s * r_b
                    tx0 = ax0 - s * band_h; ty0 = ay0 + c * band_h
                    tx1 = ax1 - s * band_h; ty1 = ay1 + c * band_h
                    fig.add_trace(go.Scatter(
                        x=[ax0, ax1, tx1, tx0, ax0],
                        y=[ay0, ay1, ty1, ty0, ay0],
                        fill='toself', fillcolor='seagreen', opacity=0.25,
                        mode='none', hoverinfo='skip', showlegend=False
                    ))
                    if first_band and (r_b - r_a) > 0.5:
                        fig.add_annotation(
                            x=(ax0 + ax1) / 2, y=(ay0 + ay1) / 2 + band_h,
                            text="UDL", showarrow=False, yshift=4 * font_scale,
                            font=dict(color='seagreen', size=marker_size, weight="bold"))
                        first_band = False

            # --- LOADS (Cannot be easily merged due to diverse shapes) ---
            if 'Envelope' not in load_case_name and 'loads' in data:
                for i_load, load in enumerate(data['loads']):
                    l_type = load['type']
                    
                    if load_case_name == "Vehicle Steps" and l_type == 'point':
                        p_val = load['params'][0]
                        lx = load['params'][1]
                        bas_x = ni[0] + c * lx; bas_y = ni[1] + s * lx
                        dx, dy = 0.0, -1.0 
                        base_len = 2.0
                        tail_len = max(base_len * (abs(p_val) / max_P_veh), 1.0) 
                        tail_x = bas_x - dx * tail_len; tail_y = bas_y - dy * tail_len
                        
                        fig.add_annotation(
                            x=bas_x, y=bas_y, ax=tail_x, ay=tail_y,
                            xref='x', yref='y', axref='x', ayref='y',
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2.5, 
                            arrowcolor='orange', opacity=1.0
                        )
                        # Three stagger levels: with closely spaced axles
                        # (e.g. 1.4 m bogies) two levels still place every
                        # second label on the same row, where they overlap
                        # on longer decks.
                        shift_val = (10 + (i_load % 3) * 15) * font_scale
                        load_text = f"{int(round(abs(p_val)))} kN"
                        fig.add_annotation(
                            x=tail_x, y=tail_y, text=load_text, 
                            showarrow=False, yshift=shift_val, 
                            font=dict(color='orange', size=veh_load_font_size, weight="bold")
                        )

                    elif load_case_name == "Dead Load" and l_type == 'distributed_trapezoid' and load.get('is_gravity', False):
                        q_val = load['params'][0]
                        h_vis = 0.6 * (abs(q_val) / max_q_sw)
                        if h_vis < 0.1: h_vis = 0.1
                        x_st = ni[0] - s * h_vis; y_st = ni[1] + c * h_vis
                        x_et = (ni[0] + c * L) - s * h_vis; y_et = (ni[1] + s * L) + c * h_vis
                        
                        fig.add_trace(go.Scatter(
                            x=[ni[0], ni[0]+c*L, x_et, x_st, ni[0]],
                            y=[ni[1], ni[1]+s*L, y_et, y_st, ni[1]],
                            fill='toself', fillcolor='orange', opacity=0.3, mode='none',
                            hoverinfo='skip', showlegend=False
                        ))
                        xm = (x_st + x_et) / 2; ym = (y_st + y_et) / 2
                        fig.add_annotation(
                            x=xm, y=ym, text=f"{q_val:.1f}", showarrow=False, 
                            font=dict(color='orange', size=marker_size, weight="bold"), yshift=5*font_scale
                        )

                    elif load_case_name == "Soil" and l_type == 'distributed_trapezoid' and not load.get('is_gravity', False):
                        q_bot, q_top, x_s, x_e = load['params']
                        target_width = 1.5 
                        w_bot = target_width * (abs(q_bot) / max_q_soil)
                        w_top = target_width * (abs(q_top) / max_q_soil)
                        
                        b_x_bot = ni[0] + c * x_s; b_y_bot = ni[1] + s * x_s
                        b_x_top = ni[0] + c * x_e; b_y_top = ni[1] + s * x_e
                        
                        dir_sign = 1.0 if q_bot >= 0 else -1.0
                        draw_dir_x = dir_sign * (-s); draw_dir_y = dir_sign * c
                        
                        t_x_bot = b_x_bot + draw_dir_x * w_bot; t_y_bot = b_y_bot + draw_dir_y * w_bot
                        t_x_top = b_x_top + draw_dir_x * w_top; t_y_top = b_y_top + draw_dir_y * w_top
                        
                        fig.add_trace(go.Scatter(
                            x=[b_x_bot, b_x_top, t_x_top, t_x_bot, b_x_bot],
                            y=[b_y_bot, b_y_top, t_y_top, t_y_bot, b_y_bot],
                            fill='toself', fillcolor='orange', opacity=0.4, mode='none',
                            hoverinfo='skip', showlegend=False
                        ))
                        fig.add_annotation(x=t_x_bot, y=t_y_bot, text=f"{abs(q_bot):.1f}", showarrow=False, font=dict(color='orange', size=marker_size*0.9, weight="bold"))
                        fig.add_annotation(x=t_x_top, y=t_y_top, text=f"{abs(q_top):.1f}", showarrow=False, font=dict(color='orange', size=marker_size*0.9, weight="bold"))

                    elif load_case_name == "Surcharge" and l_type == 'distributed_trapezoid' and not load.get('is_gravity', False):
                        q_bot, q_top, x_s, x_e = load['params']
                        target_width = 1.0 
                        w_vis = target_width * (abs(q_bot) / max_q_surch)
                        if w_vis < 0.1: w_vis = 0.1
                        b_x_bot = ni[0] + c * x_s; b_y_bot = ni[1] + s * x_s
                        b_x_top = ni[0] + c * x_e; b_y_top = ni[1] + s * x_e
                        dir_sign = 1.0 if q_bot >= 0 else -1.0
                        draw_dir_x = dir_sign * (-s); draw_dir_y = dir_sign * c
                        t_x_bot = b_x_bot + draw_dir_x * w_vis; t_y_bot = b_y_bot + draw_dir_y * w_vis
                        t_x_top = b_x_top + draw_dir_x * w_vis; t_y_top = b_y_top + draw_dir_y * w_vis
                        
                        fig.add_trace(go.Scatter(
                            x=[b_x_bot, b_x_top, t_x_top, t_x_bot, b_x_bot],
                            y=[b_y_bot, b_y_top, t_y_top, t_y_bot, b_y_bot],
                            fill='toself', fillcolor='purple', opacity=0.4, mode='none',
                            hoverinfo='skip', showlegend=False
                        ))
                        fig.add_annotation(x=t_x_bot, y=t_y_bot, text=f"{abs(q_bot):.1f}", showarrow=False, font=dict(color='purple', size=marker_size*0.9, weight="bold"))

        # 3. Add Merged Traces to Figure
        show_leg = False
        if not legend_flags[sys_key]:
            show_leg = True
            legend_flags[sys_key] = True

        if X_pos_list:
            # Flatten accumulators
            X_pos_all = np.concatenate(X_pos_list)
            Y_pos_all = np.concatenate(Y_pos_list)
            C_pos_all = np.concatenate(C_pos_list)
            
            # Fill Trace (Background)
            if X_fill_list:
                X_fill_all = np.concatenate(X_fill_list)
                Y_fill_all = np.concatenate(Y_fill_list)
                
                # Attach legend to fill for consistency with envelope mode, or main line
                # Original code put legend on Fill for fill_mode, line for step.
                # Here we use one consolidated legend entry on the fill.
                fig.add_trace(go.Scatter(
                    x=X_fill_all, y=Y_fill_all,
                    fill='toself', fillcolor=color, opacity=0.2 if is_envelope else 0.1, line=dict(width=0),
                    name=f"{sys_name}", showlegend=show_leg, hoverinfo='skip'
                ))
                show_leg = False # Don't duplicate legend on lines
            
            if is_envelope:
                X_neg_all = np.concatenate(X_neg_list)
                Y_neg_all = np.concatenate(Y_neg_list)
                C_neg_all = np.concatenate(C_neg_list)
                
                # Max Line
                fig.add_trace(go.Scatter(
                    x=X_pos_all, y=Y_pos_all, mode='lines',
                    line=dict(color=color, width=2.5, dash=line_style), showlegend=False,
                    customdata=C_pos_all, hovertemplate=htemp_max
                ))
                # Min Line
                fig.add_trace(go.Scatter(
                    x=X_neg_all, y=Y_neg_all, mode='lines',
                    line=dict(color=color, width=2.5, dash=line_style), showlegend=False,
                    customdata=C_neg_all, hovertemplate=htemp_min
                ))
            else:
                # Single Line (Step)
                # If fill didn't exist (unlikely in this logic), legend goes here
                fig.add_trace(go.Scatter(
                    x=X_pos_all, y=Y_pos_all, mode='lines',
                    line=dict(color=color, width=3.0, dash=line_style), 
                    name=f"{sys_name}", showlegend=show_leg,
                    customdata=C_pos_all, hovertemplate=htemp_step
                ))

    # Section-depth overlay first, so the bands sit beneath every diagram.
    if show_thickness:
        if show_A: _draw_thickness_bands(fig, geom_A, params_A, True, legend_flags)
        if show_B: _draw_thickness_bands(fig, geom_B, params_B, False, legend_flags)

    if show_A: add_traces(sysA_data, name_A, "blue", "solid")
    if show_B: add_traces(sysB_data, name_B, "red", "dash")

    # Horizontal structure extent, for sizing annotation footprints in data
    # units (labels render at fixed pixel size, so their data-unit size
    # scales with the extent shown). Only the SHOWN systems count: callers
    # (e.g. report critical-step plots) pass both geometries with a single
    # show flag, and a hidden longer deck must not inflate the sizing.
    ext_sources = []
    if show_A:
        ext_sources.extend((geom_A, sysA_data))
    if show_B:
        ext_sources.extend((geom_B, sysB_data))
    extent_x = structure_extent_x(ext_sources)

    solved = solve_annotations(ann_candidates, extent_x=extent_x, font_scale=font_scale)
    for ann in solved:
        fig.add_annotation(
            x=ann['x'], y=ann['y'], text=ann['text'], showarrow=False,
            font=dict(color=ann['color'], size=base_font_size, family="Arial", weight="bold"), 
            bgcolor="rgba(255,255,255,0.7)", bordercolor=ann['color'], borderwidth=1, borderpad=2
        )

    # Element-name labels (S1, W2, ...): a QA aid sized to the chosen diagram
    # height. Only label a system whose result is actually drawn - add_traces
    # renders a system only when its sys_data is non-empty, regardless of the
    # show_A/show_B UI selection (e.g. a static case defined for one system
    # only leaves the other side's data empty). Per element, an element that
    # overlaps in both plotted systems gets a single neutral label; otherwise
    # each system's labels take its own colour (blue = A, red = B).
    if show_element_names:
        elem_font = float(np.clip(11.0 * (float(target_height) / 2.0) ** 0.5 * font_scale, 9.0, 26.0))
        gsrc_A = geom_A if geom_A else sysA_data
        gsrc_B = geom_B if geom_B else sysB_data
        plot_A = bool(show_A and sysA_data)
        plot_B = bool(show_B and sysB_data)
        if plot_A and plot_B:
            _draw_element_labels_overlay(fig, gsrc_A, gsrc_B, elem_font)
        elif plot_A:
            _draw_element_labels(fig, gsrc_A, 'blue', elem_font)
        elif plot_B:
            _draw_element_labels(fig, gsrc_B, 'red', elem_font)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14*font_scale)),
        yaxis=dict(scaleanchor="x", scaleratio=1, visible=False),
        xaxis=dict(visible=False),
        plot_bgcolor='white',
        margin=dict(l=margin_base, r=margin_base, t=top_margin, b=margin_base),
        showlegend=True,
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            font=dict(size=10*font_scale)
        )
    )
    
    return fig
