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
    """
    Generates the comparative Plotly figure for System A and System B.
    geom_A/geom_B: Optional static datasets to define the structure skeleton (useful if sysA_data is empty for a step).
    """
    fig = go.Figure()
    
    # 1. Draw Geometry (Skeleton) - independently for both systems
    def add_structure_trace(g_data, sys_label, visible):
        if not visible or not g_data: return
        geom_x, geom_y = [], []
        for eid, data in g_data.items():
            ni, nj = data['ni'], data['nj']
            geom_x.extend([ni[0], nj[0], None])
            geom_y.extend([ni[1], nj[1], None])
        
        fig.add_trace(go.Scatter(
            x=geom_x, y=geom_y, 
            mode='lines', 
            name=f'Structure {sys_label}',
            legendgroup='Structure',
            line=dict(color='black', width=3), 
            opacity=0.2, 
            hoverinfo='skip',
            showlegend=True
        ))

    # Use specific geometry source if provided, otherwise fallback to the result data
    source_A = geom_A if geom_A else sysA_data
    source_B = geom_B if geom_B else sysB_data
    
    # Draw both if requested
    add_structure_trace(source_A, "(A)", show_A)
    # Avoid drawing B if it's identical to A? No, draw both to be safe (they overlay)
    add_structure_trace(source_B, "(B)", show_B)
    
    ann_candidates = []
    
    # 2. Determine Scale Factor (Auto-scaling based on max value across active systems)
    max_val_plot = 0.0
    systems_to_check = []
    if show_A: systems_to_check.append(sysA_data)
    if show_B: systems_to_check.append(sysB_data)
    
    for sys_d in systems_to_check:
        if not sys_d: continue
        for eid, data in sys_d.items():
            keys_to_check = []
            if type_base == 'Def':
                keys_to_check = ['def_x_max', 'def_x_min', 'def_y_max', 'def_y_min', 'def_x', 'def_y']
            else:
                keys_to_check = [f"{type_base}_max", f"{type_base}_min", type_base]
            
            for k in keys_to_check:
                if k in data:
                    val = data[k] * 1000.0 if type_base == 'Def' else data[k]
                    max_val_plot = max(max_val_plot, np.max(np.abs(val)))

    scale_factor = 1.0
    if max_val_plot > 1e-5:
        scale_factor = target_height / max_val_plot
    
    # 3. Determine Load Scale Factor
    max_P_load = 1.0
    max_q_load = 1.0
    
    show_loads = "Envelope" not in load_case_name and load_case_name != ""
    
    if show_loads:
        for sys_data in [sysA_data, sysB_data]:
            if not sys_data: continue
            for eid, data in sys_data.items():
                for load in data.get('loads', []):
                    if load['type'] == 'point':
                        max_P_load = max(max_P_load, abs(load['params'][0]))
                    elif load['type'] == 'distributed_trapezoid':
                        max_q_load = max(max_q_load, abs(load['params'][0]), abs(load['params'][1]))
    
    # Helper: Draw Loads
    def draw_loads(res_data, sys_name):
        if not res_data: return
        vis_h_load = 1.5
        q_fac = vis_h_load / max_q_load if max_q_load > 0 else 1.0

        for eid, data in res_data.items():
            loads = data.get('loads', [])
            if not loads: continue
            ni, nj = data['ni'], data['nj']
            cx, cy = data['cx'], data['cy']
            perp_x, perp_y = -cy, cx 
            
            for load in loads:
                if load['type'] == 'point':
                    P, a = load['params']
                    px = ni[0] + a * cx
                    py = ni[1] + a * cy
                    tail_len = abs(P) / max_P_load * 2.0 
                    if tail_len < 0.3: tail_len = 0.3
                    vx = -1.0 * np.sign(P) * perp_x 
                    vy = -1.0 * np.sign(P) * perp_y
                    tx = px - vx * tail_len
                    ty = py - vy * tail_len
                    
                    fig.add_annotation(
                        x=px, y=py, ax=tx, ay=ty, xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='orange',
                        text=f"{abs(P):.1f} kN", font=dict(color='orange', size=10), yshift=5
                    )
                    
                elif load['type'] == 'distributed_trapezoid':
                    qs, qe, hs, he = load['params']
                    x_s, y_s = ni[0] + hs * cx, ni[1] + hs * cy
                    x_e, y_e = ni[0] + he * cx, ni[1] + he * cy
                    v_qs, v_qe = qs * q_fac, qe * q_fac
                    tx_s, ty_s = x_s + v_qs * perp_x, y_s + v_qs * perp_y
                    tx_e, ty_e = x_e + v_qe * perp_x, y_e + v_qe * perp_y
                    
                    fig.add_trace(go.Scatter(
                        x=[x_s, x_e, tx_e, tx_s, x_s],
                        y=[y_s, y_e, ty_e, ty_s, y_s],
                        fill='toself', fillcolor='rgba(255, 165, 0, 0.2)',
                        mode='lines', line=dict(color='orange', width=1),
                        hoverinfo='skip', showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=[tx_s, tx_e], y=[ty_s, ty_e],
                        mode='text', text=[f"{abs(qs):.1f}", f"{abs(qe):.1f}"],
                        textposition="top center", textfont=dict(color='orange', size=10),
                        showlegend=False, hoverinfo='skip'
                    ))

    if show_loads:
        if show_A: draw_loads(sysA_data, "A")
        if show_B: draw_loads(sysB_data, "B")

    def get_fill_color(color, alpha=0.2):
        if color == 'blue': return f'rgba(0, 0, 255, {alpha})'
        if color == 'red': return f'rgba(255, 0, 0, {alpha})'
        return color

    # Helper: Add Results Traces
    def add_traces(res_data, sys_name, color, dash, offset_dir):
        if not res_data: return
        
        # Flags to ensure we only show one legend entry per group
        showed_legend_max = False
        showed_legend_min = False
        
        for eid, data in res_data.items():
            vals_max, vals_min = None, None
            if type_base == 'Def':
                is_vertical = abs(data['cy']) > abs(data['cx'])
                comp = 'def_x' if is_vertical else 'def_y'
                # Check for envelope keys first, then static keys
                if f"{comp}_max" in data:
                    vals_max = data[f"{comp}_max"] * 1000.0
                    vals_min = data[f"{comp}_min"] * 1000.0
                elif comp in data:
                    vals_max = data[comp] * 1000.0
                    vals_min = data[comp] * 1000.0
                else: continue
            else:
                max_key, min_key = f"{type_base}_max", f"{type_base}_min"
                if max_key in data:
                    vals_max, vals_min = data[max_key], data[min_key]
                elif type_base in data:
                    vals_max, vals_min = data[type_base], data[type_base]
                else: continue

            dx, dy = data['cx'], data['cy']
            perp_x, perp_y = -dy, dx 
            base_x = data['ni'][0] + data['x'] * dx
            base_y = data['ni'][1] + data['x'] * dy
            
            # Trace Max
            hover_text_max = [
                f"<b>{sys_name} - {eid} (Max)</b><br>Loc x: {x:.2f}m<br>{type_base}: {v:.2f}"
                for x, v in zip(data['x'], vals_max)
            ]
            px_max = base_x + vals_max * scale_factor * perp_x
            py_max = base_y + vals_max * scale_factor * perp_y
            
            # Determine legend visibility for this specific segment
            show_leg_max = not showed_legend_max
            
            fig.add_trace(go.Scatter(
                x=px_max, y=py_max, mode='lines', line=dict(color=color, width=2, dash=dash),
                name=f"{sys_name} Max", 
                legendgroup=f"{sys_name} Max",
                showlegend=show_leg_max,
                text=hover_text_max, hoverinfo='text'
            ))
            showed_legend_max = True
            
            # Trace Min
            px_min = base_x + vals_min * scale_factor * perp_x
            py_min = base_y + vals_min * scale_factor * perp_y
            
            hover_text_min = [
                f"<b>{sys_name} - {eid} (Min)</b><br>Loc x: {x:.2f}m<br>{type_base}: {v:.2f}"
                for x, v in zip(data['x'], vals_min)
            ]
            
            show_leg_min = not showed_legend_min
            
            fig.add_trace(go.Scatter(
                x=px_min, y=py_min, mode='lines', line=dict(color=color, width=2, dash=dash),
                fill='tonexty', fillcolor=get_fill_color(color),
                name=f"{sys_name} Min", 
                legendgroup=f"{sys_name} Min",
                showlegend=show_leg_min,
                text=hover_text_min, hoverinfo='text'
            ))
            showed_legend_min = True
            
            # Candidate Annotations (Min/Max peaks)
            if annotate:
                idx_max = np.argmax(vals_max)
                idx_min = np.argmin(vals_min)
                indices_to_label = {idx_max, idx_min}

                for idx in indices_to_label:
                    for v_arr, is_max in [(vals_max, True), (vals_min, False)]:
                        if (is_max and idx == idx_max) or (not is_max and idx == idx_min):
                            val = v_arr[idx]
                            # Simple filter to avoid labeling zeros
                            cutoff = 0.5 if type_base != 'Def' else 0.05
                            if abs(val) > cutoff: 
                                tx = px_max[idx] if is_max else px_min[idx]
                                ty = py_max[idx] if is_max else py_min[idx]
                                x_loc = data['x'][idx]
                                L_el = data['L']
                                dir_fac = 1.0 if x_loc < L_el * 0.5 else -1.0
                                long_shift = min(L_el * 0.15, 0.8)
                                push = 0.2 if is_max else -0.2
                                tx += push * perp_x + long_shift * dir_fac * dx
                                ty += push * perp_y + long_shift * dir_fac * dy
                                
                                ann_candidates.append({
                                    'x': tx, 'y': ty, 'text': f"{val:.1f}", 'color': color,
                                    'perp_x': perp_x, 'perp_y': perp_y
                                })

    if show_A: add_traces(sysA_data, name_A, "blue", "solid", 1)
    if show_B: add_traces(sysB_data, name_B, "red", "dash", -1)
    
    # 4. Solve Annotation Overlaps
    solved = solve_annotations(ann_candidates)
    for ann in solved:
        fig.add_annotation(
            x=ann['x'], y=ann['y'], text=ann['text'], showarrow=False,
            font=dict(color=ann['color'], size=11, family="Arial", weight="bold"), 
            bgcolor="rgba(255,255,255,0.7)", borderpad=2
        )

    fig.update_layout(
        title=title, yaxis=dict(visible=False, scaleanchor="x"), xaxis=dict(visible=False),
        plot_bgcolor='white', margin=dict(l=10, r=10, t=30, b=10), height=550
    )
    return fig