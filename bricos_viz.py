import plotly.graph_objects as go
import numpy as np
import bricos_kernels as kernels

# ==========================================
# VISUALIZATION ENGINE
# ==========================================

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

def create_plotly_fig(nodes, sysA_data, sysB_data, type_base='M', target_height=2.0, title="", show_A=True, show_B=True, annotate=True, load_case_name="", name_A="System A", name_B="System B", geom_A=None, geom_B=None):
    fig = go.Figure()
    
    # --- 1. DETERMINE SCALING FACTOR (ISOLATED BY TYPE & ELEMENT) ---
    max_val = 0.0
    
    # Logic:
    # Moments/Shear/Normal: Simple Key Lookup
    # Deflection: Depends on Element Type (Wall=X, Span=Y)
    
    for data_set in [sysA_data, sysB_data]:
        if not data_set: continue
        # Must iterate items() to access Element ID
        for eid, d in data_set.items():
            
            # Determine Keys to Scan
            keys_to_scan = []
            if type_base == 'Def':
                # If Wall ('W'), look for Horizontal X. Else Vertical Y.
                if eid.startswith('W'): keys_to_scan = ['def_x', 'def_x_max', 'def_x_min']
                else: keys_to_scan = ['def_y', 'def_y_max', 'def_y_min']
            elif type_base == 'M': keys_to_scan = ['M', 'M_max', 'M_min']
            elif type_base == 'V': keys_to_scan = ['V', 'V_max', 'V_min']
            elif type_base == 'N': keys_to_scan = ['N', 'N_max', 'N_min']
            
            for k in keys_to_scan:
                if k in d:
                    val = np.max(np.abs(d[k]))
                    # Convert Deflection to mm for scaling check
                    if type_base == 'Def': val *= 1000.0 
                    max_val = max(max_val, val)
    
    scale = 1.0
    if max_val > 1e-5: scale = target_height / max_val
    
    # Unit Label Map
    unit_map = {'M': 'kNm', 'V': 'kN', 'N': 'kN', 'Def': 'mm'}
    unit = unit_map.get(type_base, '')
    
    ann_candidates = []
    legend_flags = {'struct': False, 'A': False, 'B': False}

    def add_traces(sys_data, sys_name, color, line_style, offset_dir):
        if not sys_data: return
        
        is_sys_A = (sys_name == name_A)
        sys_key = 'A' if is_sys_A else 'B'
        
        # Use geometry source for structure drawing
        geom_source = geom_A if (is_sys_A and geom_A) else (geom_B if (not is_sys_A and geom_B) else sys_data)

        # 2. Draw Structure (Geometry)
        x_struct, y_struct = [], []
        sorted_ids = sorted(geom_source.keys(), key=lambda x: int(x[1:]))
        
        for eid in sorted_ids:
            if eid not in geom_source: continue
            dat = geom_source[eid]
            ni, nj = dat['ni'], dat['nj']
            x_struct.extend([ni[0], nj[0], None])
            y_struct.extend([ni[1], nj[1], None])
        
        # Add Structure Trace (Grey)
        show_struct = False
        if not legend_flags['struct']:
            show_struct = True
            legend_flags['struct'] = True
            
        fig.add_trace(go.Scatter(
            x=x_struct, y=y_struct, mode='lines+markers', 
            line=dict(color='grey', width=3 if is_sys_A else 1),
            marker=dict(size=4, color='grey'),
            name="Structure Geometry", opacity=0.5, 
            hoverinfo='skip', showlegend=show_struct
        ))

        # 3. Draw Diagrams
        for eid, data in sys_data.items():
            if 'x' not in data: continue # Skip empty
            
            # Reconstruct global coordinates
            L = data['L']
            c, s = data['cx'], data['cy']
            ni = data['ni']
            
            x_local = data['x']
            x_glob = ni[0] + c * x_local
            y_glob = ni[1] + s * x_local
            
            # Diagram Values Selection
            vals_pos = None
            vals_neg = None
            fill_mode = False
            
            # Invert Logic
            # M: Inverted to show tension side (Standard convention)
            # Def: 
            #   - Spans (Normal=Up): +DefY is Up. No Inversion.
            #   - Walls (Normal=Left): +DefX is Right. Need Inversion (-1.0) to plot Right.
            inv = 1.0 
            if type_base == 'M': inv = 1.0 # Maintain M convention
            
            if type_base == 'Def':
                # Switch based on Element Type
                if eid.startswith('W'):
                    # Wall: Plot X
                    key_base = 'def_x'
                    inv = -1.0 # Invert because Wall Normal points Left (-X)
                else:
                    # Span: Plot Y
                    key_base = 'def_y'
                    inv = 1.0
                
                if f'{key_base}_max' in data: # Envelope
                    vals_pos = data[f'{key_base}_max'] * 1000 # Convert to mm
                    vals_neg = data[f'{key_base}_min'] * 1000
                    fill_mode = True
                else: # Step
                    vals_pos = data[key_base] * 1000
                    vals_neg = vals_pos
                    fill_mode = False
                    
            else:
                # Forces (M, V, N)
                key = type_base
                if f'{key}_max' in data: # Envelope
                    vals_pos = data[f'{key}_max']
                    vals_neg = data[f'{key}_min']
                    fill_mode = True
                else:
                    vals_pos = data[key]
                    vals_neg = vals_pos
                    fill_mode = False

            nx, ny = -s, c
            
            # Calculate Diagram Points
            x_plot_pos = x_glob + nx * vals_pos * scale * inv
            y_plot_pos = y_glob + ny * vals_pos * scale * inv
            x_plot_neg = x_glob + nx * vals_neg * scale * inv
            y_plot_neg = y_glob + ny * vals_neg * scale * inv
            
            # Prepare Custom Data for Hover
            custom_pos = np.stack((x_local, vals_pos), axis=-1)
            custom_neg = np.stack((x_local, vals_neg), axis=-1)
            
            htemp_max = f"<b>{sys_name} (Max)</b><br>Loc: %{{customdata[0]:.2f}} m<br>Val: %{{customdata[1]:.1f}} {unit}<extra></extra>"
            htemp_min = f"<b>{sys_name} (Min)</b><br>Loc: %{{customdata[0]:.2f}} m<br>Val: %{{customdata[1]:.1f}} {unit}<extra></extra>"
            htemp_step = f"<b>{sys_name}</b><br>Loc: %{{customdata[0]:.2f}} m<br>Val: %{{customdata[1]:.1f}} {unit}<extra></extra>"

            show_leg = False
            if not legend_flags[sys_key]:
                show_leg = True
                legend_flags[sys_key] = True

            # Plot Diagrams
            if fill_mode:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_plot_pos, x_plot_neg[::-1]]),
                    y=np.concatenate([y_plot_pos, y_plot_neg[::-1]]),
                    fill='toself', fillcolor=color, opacity=0.2, line=dict(width=0),
                    name=f"{sys_name}", showlegend=show_leg, hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=x_plot_pos, y=y_plot_pos, mode='lines', 
                    line=dict(color=color, width=1.5, dash=line_style), 
                    showlegend=False, customdata=custom_pos, hovertemplate=htemp_max
                ))
                fig.add_trace(go.Scatter(
                    x=x_plot_neg, y=y_plot_neg, mode='lines', 
                    line=dict(color=color, width=1.5, dash=line_style), 
                    showlegend=False, customdata=custom_neg, hovertemplate=htemp_min
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=x_plot_pos, y=y_plot_pos, mode='lines', 
                    line=dict(color=color, width=2, dash=line_style), 
                    name=f"{sys_name}", showlegend=show_leg,
                    customdata=custom_pos, hovertemplate=htemp_step
                ))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_glob, x_plot_pos[::-1]]),
                    y=np.concatenate([y_glob, y_plot_pos[::-1]]),
                    fill='toself', fillcolor=color, opacity=0.1, line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))

            # 4. LOADS VISUALIZATION LOGIC
            # Only visualize loads if NOT an Envelope case
            if 'Envelope' not in load_case_name:
                if 'loads' in data and len(data['loads']) > 0:
                    
                    # Pre-calculate Max Load for Scaling (Vehicle Steps Only)
                    max_P = 1.0
                    if load_case_name == "Vehicle Steps":
                        # Safely find max absolute point load
                        pts = [abs(l['params'][0]) for l in data['loads'] if l['type']=='point']
                        if pts: max_P = max(pts)

                    for load in data['loads']:
                        l_type = load['type']
                        
                        # --- A. VEHICLE STEPS (Point Loads) ---
                        if load_case_name == "Vehicle Steps" and l_type == 'point':
                            p_val = load['params'][0]
                            lx = load['params'][1]
                            
                            bas_x = ni[0] + c * lx
                            bas_y = ni[1] + s * lx
                            
                            # Gravity Down
                            dx, dy = 0.0, -1.0
                            
                            # Scaled Arrow
                            base_len = 2.0
                            tail_len = base_len * (abs(p_val) / max_P)
                            
                            tail_x = bas_x - dx * tail_len
                            tail_y = bas_y - dy * tail_len
                            
                            fig.add_annotation(
                                x=bas_x, y=bas_y, ax=tail_x, ay=tail_y,
                                xref='x', yref='y', axref='x', ayref='y',
                                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, 
                                arrowcolor='orange', opacity=1.0
                            )
                            fig.add_annotation(
                                x=tail_x, y=tail_y, 
                                text=f"{abs(p_val):.1f} kN", 
                                showarrow=False, yshift=10,
                                font=dict(color='orange', size=10, weight="bold")
                            )

                        # --- B. SELFWEIGHT (UDL Block) ---
                        elif load_case_name == "Selfweight" and l_type == 'distributed_trapezoid' and load.get('is_gravity', False):
                            q_val = load['params'][0]
                            
                            nx, ny = -s, c
                            h_block = 0.4
                            
                            x_start = ni[0]
                            y_start = ni[1]
                            x_end = ni[0] + c * L
                            y_end = ni[1] + s * L
                            
                            x_start_top = x_start + nx * h_block
                            y_start_top = y_start + ny * h_block
                            x_end_top = x_end + nx * h_block
                            y_end_top = y_end + ny * h_block
                            
                            fig.add_trace(go.Scatter(
                                x=[x_start, x_end, x_end_top, x_start_top, x_start],
                                y=[y_start, y_end, y_end_top, y_start_top, y_start],
                                fill='toself', fillcolor='orange', opacity=0.3, mode='none',
                                hoverinfo='skip', showlegend=False
                            ))
                            
                            xm = (x_start_top + x_end_top) / 2
                            ym = (y_start_top + y_end_top) / 2
                            fig.add_annotation(
                                x=xm, y=ym, text=f"{q_val:.1f} kN/m", 
                                showarrow=False, font=dict(color='orange', size=10), yshift=5
                            )

                        # --- C. SOIL LOAD (Block) ---
                        elif load_case_name == "Soil" and l_type == 'distributed_trapezoid' and not load.get('is_gravity', False):
                            q_bot, q_top, x_s, L_load = load['params']
                            
                            nx, ny = -s, c
                            vis_scale = 1.5 / max(abs(q_bot)+0.1, abs(q_top)+0.1)
                            w_bot = abs(q_bot) * vis_scale
                            w_top = abs(q_top) * vis_scale
                            
                            b_x_bot = ni[0] + c * x_s
                            b_y_bot = ni[1] + s * x_s
                            b_x_top = ni[0] + c * (x_s + L_load)
                            b_y_top = ni[1] + s * (x_s + L_load)
                            
                            dir_sign = 1.0 if q_bot >= 0 else -1.0
                            draw_dir_x = -dir_sign * nx
                            draw_dir_y = -dir_sign * ny
                            
                            t_x_bot = b_x_bot + draw_dir_x * w_bot
                            t_y_bot = b_y_bot + draw_dir_y * w_bot
                            t_x_top = b_x_top + draw_dir_x * w_top
                            t_y_top = b_y_top + draw_dir_y * w_top
                            
                            fig.add_trace(go.Scatter(
                                x=[b_x_bot, b_x_top, t_x_top, t_x_bot, b_x_bot],
                                y=[b_y_bot, b_y_top, t_y_top, t_y_bot, b_y_bot],
                                fill='toself', fillcolor='orange', opacity=0.4, mode='none',
                                hoverinfo='skip', showlegend=False
                            ))
                            
                            for k in [0.25, 0.5, 0.75]:
                                bx = b_x_bot + k*(b_x_top - b_x_bot)
                                by = b_y_bot + k*(b_y_top - b_y_bot)
                                tx = t_x_bot + k*(t_x_top - t_x_bot)
                                ty = t_y_bot + k*(t_y_top - t_y_bot)
                                fig.add_annotation(
                                    x=bx, y=by, ax=tx, ay=ty,
                                    xref='x', yref='y', axref='x', ayref='y',
                                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, 
                                    arrowcolor='orange', opacity=0.6
                                )
                                
                            fig.add_annotation(
                                x=t_x_bot, y=t_y_bot, text=f"{abs(q_bot):.1f}", 
                                showarrow=False, font=dict(color='orange', size=9), xshift=0
                            )
                            fig.add_annotation(
                                x=t_x_top, y=t_y_top, text=f"{abs(q_top):.1f}", 
                                showarrow=False, font=dict(color='orange', size=9), xshift=0
                            )

            # 5. ANNOTATIONS SELECTION (Max/Min Labels)
            if annotate:
                # Calculate threshold relative to the active plot type
                threshold = max_val * 0.05
                
                if fill_mode:
                    idx_max = np.argmax(vals_pos)
                    idx_min = np.argmin(vals_neg)
                    
                    if abs(vals_pos[idx_max]) > threshold:
                        perp_x, perp_y = nx, ny 
                        tx = x_plot_pos[idx_max]
                        ty = y_plot_pos[idx_max]
                        ann_candidates.append({
                            'x': tx, 'y': ty, 'text': f"{vals_pos[idx_max]:.1f}", 'color': color,
                            'perp_x': perp_x, 'perp_y': perp_y
                        })

                    if abs(vals_neg[idx_min]) > threshold and idx_min != idx_max:
                        perp_x, perp_y = nx, ny 
                        tx = x_plot_neg[idx_min]
                        ty = y_plot_neg[idx_min]
                        ann_candidates.append({
                            'x': tx, 'y': ty, 'text': f"{vals_neg[idx_min]:.1f}", 'color': color,
                            'perp_x': perp_x, 'perp_y': perp_y
                        })
                else:
                    vals_abs = np.abs(vals_pos)
                    idx = np.argmax(vals_abs)
                    val = vals_pos[idx]
                    if abs(val) > threshold:
                         tx = x_plot_pos[idx]
                         ty = y_plot_pos[idx]
                         ann_candidates.append({
                            'x': tx, 'y': ty, 'text': f"{val:.1f}", 'color': color,
                            'perp_x': nx, 'perp_y': ny
                        })

    if show_A: add_traces(sysA_data, name_A, "blue", "solid", 1)
    if show_B: add_traces(sysB_data, name_B, "red", "dash", -1)
    
    # 6. Solve Annotation Overlaps
    solved = solve_annotations(ann_candidates)
    for ann in solved:
        fig.add_annotation(
            x=ann['x'], y=ann['y'], text=ann['text'], showarrow=False,
            font=dict(color=ann['color'], size=11, family="Arial", weight="bold"), 
            bgcolor="rgba(255,255,255,0.7)", bordercolor=ann['color'], borderwidth=1, borderpad=2
        )

    fig.update_layout(
        title=title,
        yaxis=dict(scaleanchor="x", scaleratio=1, visible=False),
        xaxis=dict(visible=False),
        plot_bgcolor='white',
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig