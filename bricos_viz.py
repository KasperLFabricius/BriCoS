import plotly.graph_objects as go
import numpy as np
import bricos_kernels as kernels

# ... (Previous helper functions solve_annotations and _add_support_icon remain unchanged) ...

def solve_annotations(annotations):
    """
    Optimizes label placement to prevent overlaps using a rigid-body physics approach.
    """
    if not annotations: return []
    n = len(annotations)
    data_arr = np.zeros((n, 6))
    
    # Pack data for Numba kernel
    for i, ann in enumerate(annotations):
        # Heuristic width calculation based on character count
        ann['w'] = len(ann['text']) * 0.15 
        ann['h'] = 0.40 # Fixed height assumption
        data_arr[i, :] = [ann['x'], ann['y'], ann['w'], ann['h'], ann['perp_x'], ann['perp_y']]
        
    # Run the solver (Pure Math)
    result_arr = kernels.jit_annotation_solver(data_arr)
    
    # Unpack results
    for i, ann in enumerate(annotations):
        ann['x'] = result_arr[i, 0]
        ann['y'] = result_arr[i, 1]
    return annotations

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


def create_plotly_fig(
    nodes, sysA_data, sysB_data, type_base='M', target_height=2.0, title="", 
    show_A=True, show_B=True, annotate=True, load_case_name="", 
    name_A="System A", name_B="System B", 
    geom_A=None, geom_B=None,
    params_A=None, params_B=None,
    show_supports=False, support_size=0.5,
    font_scale=1.0 
):
    fig = go.Figure()
    
    # Safety Defaults
    if params_A is None: params_A = {}
    if params_B is None: params_B = {}
    
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
                    val = np.max(np.abs(d[k]))
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
    legend_flags = {'struct': False, 'A': False, 'B': False}

    # Font Sizes & Margin Logic
    base_font_size = 12 * font_scale
    marker_size = 10 * font_scale # Slightly smaller markers
    
    # Increase margins significantly if scaling font to prevent cut-off
    margin_base = 30 * font_scale
    top_margin = 50 * font_scale # Extra room for title
    
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
                
                if sys_nodes_dict and nid in sys_nodes_dict:
                    pos_x, pos_y = sys_nodes_dict[nid]
                else:
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

        if show_A and params_A: render_system_supports(params_A, nodes, 'blue')
        if show_B and params_B: render_system_supports(params_B, nodes, 'red')

    def add_traces(sys_data, sys_name, color, line_style, offset_dir):
        if not sys_data: return
        is_sys_A = (sys_name == name_A)
        sys_key = 'A' if is_sys_A else 'B'
        geom_source = geom_A if (is_sys_A and geom_A) else (geom_B if (not is_sys_A and geom_B) else sys_data)

        x_struct, y_struct = [], []
        if geom_source:
            sorted_ids = sorted(geom_source.keys(), key=lambda x: int(x[1:]))
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

        for eid, data in sys_data.items():
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
                if f'{key}_max' in data:
                    vals_pos = data[f'{key}_max']
                    vals_neg = data[f'{key}_min']
                    fill_mode = True
                else:
                    vals_pos = data[key]
                    vals_neg = vals_pos
                    fill_mode = False

            nx, ny = -s, c
            x_plot_pos = x_glob + nx * vals_pos * scale * inv
            y_plot_pos = y_glob + ny * vals_pos * scale * inv
            x_plot_neg = x_glob + nx * vals_neg * scale * inv
            y_plot_neg = y_glob + ny * vals_neg * scale * inv
            
            custom_pos = np.stack((x_local, vals_pos), axis=-1)
            custom_neg = np.stack((x_local, vals_neg), axis=-1)
            
            htemp_max = f"<b>{sys_name} (Max)</b><br>Loc: %{{customdata[0]:.2f}} m<br>Val: %{{customdata[1]:.1f}} {unit}<extra></extra>"
            htemp_min = f"<b>{sys_name} (Min)</b><br>Loc: %{{customdata[0]:.2f}} m<br>Val: %{{customdata[1]:.1f}} {unit}<extra></extra>"
            htemp_step = f"<b>{sys_name}</b><br>Loc: %{{customdata[0]:.2f}} m<br>Val: %{{customdata[1]:.1f}} {unit}<extra></extra>"

            show_leg = False
            if not legend_flags[sys_key]:
                show_leg = True
                legend_flags[sys_key] = True

            if fill_mode:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_plot_pos, x_plot_neg[::-1]]),
                    y=np.concatenate([y_plot_pos, y_plot_neg[::-1]]),
                    fill='toself', fillcolor=color, opacity=0.2, line=dict(width=0),
                    name=f"{sys_name}", showlegend=show_leg, hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=x_plot_pos, y=y_plot_pos, mode='lines', 
                    line=dict(color=color, width=2.5, dash=line_style), showlegend=False, 
                    customdata=custom_pos, hovertemplate=htemp_max
                ))
                fig.add_trace(go.Scatter(
                    x=x_plot_neg, y=y_plot_neg, mode='lines', 
                    line=dict(color=color, width=2.5, dash=line_style), showlegend=False, 
                    customdata=custom_neg, hovertemplate=htemp_min
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=x_plot_pos, y=y_plot_pos, mode='lines', 
                    line=dict(color=color, width=3.0, dash=line_style), 
                    name=f"{sys_name}", showlegend=show_leg,
                    customdata=custom_pos, hovertemplate=htemp_step
                ))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_glob, x_plot_pos[::-1]]),
                    y=np.concatenate([y_glob, y_plot_pos[::-1]]),
                    fill='toself', fillcolor=color, opacity=0.1, line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))

            if 'Envelope' not in load_case_name and 'loads' in data:
                for load in data['loads']:
                    l_type = load['type']
                    
                    if load_case_name == "Vehicle Steps" and l_type == 'point':
                        p_val = load['params'][0]
                        lx = load['params'][1]
                        bas_x = ni[0] + c * lx; bas_y = ni[1] + s * lx
                        dx, dy = 0.0, -1.0 
                        base_len = 2.0
                        tail_len = base_len * (abs(p_val) / max_P_veh)
                        tail_x = bas_x - dx * tail_len; tail_y = bas_y - dy * tail_len
                        
                        fig.add_annotation(
                            x=bas_x, y=bas_y, ax=tail_x, ay=tail_y,
                            xref='x', yref='y', axref='x', ayref='y',
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2.5, 
                            arrowcolor='orange', opacity=1.0
                        )
                        fig.add_annotation(
                            x=tail_x, y=tail_y, text=f"{abs(p_val):.1f} kN", 
                            showarrow=False, yshift=10 * font_scale, 
                            font=dict(color='orange', size=marker_size, weight="bold")
                        )

                    elif load_case_name == "Selfweight" and l_type == 'distributed_trapezoid' and load.get('is_gravity', False):
                        q_val = load['params'][0]
                        nx, ny = -s, c
                        h_vis = 0.6 * (abs(q_val) / max_q_sw)
                        if h_vis < 0.1: h_vis = 0.1
                        x_start = ni[0]; y_start = ni[1]
                        x_end = ni[0] + c * L; y_end = ni[1] + s * L
                        x_st = x_start + nx * h_vis; y_st = y_start + ny * h_vis
                        x_et = x_end + nx * h_vis; y_et = y_end + ny * h_vis
                        
                        fig.add_trace(go.Scatter(
                            x=[x_start, x_end, x_et, x_st, x_start],
                            y=[y_start, y_end, y_et, y_st, y_start],
                            fill='toself', fillcolor='orange', opacity=0.3, mode='none',
                            hoverinfo='skip', showlegend=False
                        ))
                        xm = (x_st + x_et) / 2; ym = (y_st + y_et) / 2
                        fig.add_annotation(
                            x=xm, y=ym, text=f"{q_val:.1f}", showarrow=False, 
                            font=dict(color='orange', size=marker_size, weight="bold"), yshift=5*font_scale
                        )

                    elif load_case_name == "Soil" and l_type == 'distributed_trapezoid' and not load.get('is_gravity', False):
                        q_bot, q_top, x_s, L_load = load['params']
                        nx, ny = -s, c
                        target_width = 1.5 
                        w_bot = target_width * (abs(q_bot) / max_q_soil)
                        w_top = target_width * (abs(q_top) / max_q_soil)
                        
                        b_x_bot = ni[0] + c * x_s; b_y_bot = ni[1] + s * x_s
                        b_x_top = ni[0] + c * (x_s + L_load); b_y_top = ni[1] + s * (x_s + L_load)
                        
                        dir_sign = 1.0 if q_bot >= 0 else -1.0
                        draw_dir_x = dir_sign * nx; draw_dir_y = dir_sign * ny
                        
                        t_x_bot = b_x_bot + draw_dir_x * w_bot; t_y_bot = b_y_bot + draw_dir_y * w_bot
                        t_x_top = b_x_top + draw_dir_x * w_top; t_y_top = b_y_top + draw_dir_y * w_top
                        
                        fig.add_trace(go.Scatter(
                            x=[b_x_bot, b_x_top, t_x_top, t_x_bot, b_x_bot],
                            y=[b_y_bot, b_y_top, t_y_top, t_y_bot, b_y_bot],
                            fill='toself', fillcolor='orange', opacity=0.4, mode='none',
                            hoverinfo='skip', showlegend=False
                        ))
                        for k in [0.25, 0.5, 0.75]:
                            bx = b_x_bot + k*(b_x_top - b_x_bot); by = b_y_bot + k*(b_y_top - b_y_bot)
                            tx = t_x_bot + k*(t_x_top - t_x_bot); ty = t_y_bot + k*(t_y_top - t_y_bot)
                            fig.add_annotation(
                                x=bx, y=by, ax=tx, ay=ty, xref='x', yref='y', axref='x', ayref='y',
                                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor='orange', opacity=0.6
                            )
                        fig.add_annotation(
                            x=t_x_bot, y=t_y_bot, text=f"{abs(q_bot):.1f}", showarrow=False, 
                            font=dict(color='orange', size=marker_size*0.9, weight="bold")
                        )
                        fig.add_annotation(
                            x=t_x_top, y=t_y_top, text=f"{abs(q_top):.1f}", showarrow=False, 
                            font=dict(color='orange', size=marker_size*0.9, weight="bold")
                        )

                    elif load_case_name == "Surcharge" and l_type == 'distributed_trapezoid' and not load.get('is_gravity', False):
                        q_bot, q_top, x_s, L_load = load['params']
                        nx, ny = -s, c
                        target_width = 1.0 
                        w_vis = target_width * (abs(q_bot) / max_q_surch)
                        if w_vis < 0.1: w_vis = 0.1
                        b_x_bot = ni[0] + c * x_s; b_y_bot = ni[1] + s * x_s
                        b_x_top = ni[0] + c * (x_s + L_load); b_y_top = ni[1] + s * (x_s + L_load)
                        dir_sign = 1.0 if q_bot >= 0 else -1.0
                        draw_dir_x = dir_sign * nx; draw_dir_y = dir_sign * ny
                        t_x_bot = b_x_bot + draw_dir_x * w_vis; t_y_bot = b_y_bot + draw_dir_y * w_vis
                        t_x_top = b_x_top + draw_dir_x * w_vis; t_y_top = b_y_top + draw_dir_y * w_vis
                        
                        fig.add_trace(go.Scatter(
                            x=[b_x_bot, b_x_top, t_x_top, t_x_bot, b_x_bot],
                            y=[b_y_bot, b_y_top, t_y_top, t_y_bot, b_y_bot],
                            fill='toself', fillcolor='purple', opacity=0.4, mode='none',
                            hoverinfo='skip', showlegend=False
                        ))
                        for k in [0.5]:
                            bx = b_x_bot + k*(b_x_top - b_x_bot); by = b_y_bot + k*(b_y_top - b_y_bot)
                            tx = t_x_bot + k*(t_x_top - t_x_bot); ty = t_y_bot + k*(t_y_top - t_y_bot)
                            fig.add_annotation(
                                x=bx, y=by, ax=tx, ay=ty, xref='x', yref='y', axref='x', ayref='y',
                                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor='purple', opacity=0.6
                            )
                        fig.add_annotation(
                            x=t_x_bot, y=t_y_bot, text=f"{abs(q_bot):.1f}", showarrow=False, 
                            font=dict(color='purple', size=marker_size*0.9, weight="bold")
                        )

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

    if show_A: add_traces(sysA_data, name_A, "blue", "solid", 1)
    if show_B: add_traces(sysB_data, name_B, "red", "dash", -1)
    
    solved = solve_annotations(ann_candidates)
    for ann in solved:
        fig.add_annotation(
            x=ann['x'], y=ann['y'], text=ann['text'], showarrow=False,
            font=dict(color=ann['color'], size=base_font_size, family="Arial", weight="bold"), 
            bgcolor="rgba(255,255,255,0.7)", bordercolor=ann['color'], borderwidth=1, borderpad=2
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14*font_scale)),
        yaxis=dict(scaleanchor="x", scaleratio=1, visible=False),
        xaxis=dict(visible=False),
        plot_bgcolor='white',
        # INCREASED MARGINS TO PREVENT CUT-OFF
        margin=dict(l=margin_base, r=margin_base, t=top_margin, b=margin_base),
        showlegend=True,
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            font=dict(size=10*font_scale)
        )
    )
    
    return fig