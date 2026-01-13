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
    
    # Determine Scaling Factor
    max_val = 0.0
    for data_set in [sysA_data, sysB_data]:
        if not data_set: continue
        for d in data_set.values():
            # Check for both Envelope keys and Standard keys
            keys_to_check = ['M', 'V', 'N', 'M_max', 'M_min', 'V_max', 'V_min', 'N_max', 'N_min', 'def_x', 'def_y']
            for k in keys_to_check:
                if k in d: max_val = max(max_val, np.max(np.abs(d[k])))
    
    scale = 1.0
    if max_val > 1e-5: scale = target_height / max_val
    
    ann_candidates = []

    def add_traces(sys_data, sys_name, color, line_style, offset_dir):
        if not sys_data: return
        
        # Use geometry source (geom_A/B) for drawing the structure lines if available (contains true lengths/coords), 
        # otherwise fallback to results data (sys_data).
        geom_source = geom_A if (sys_name == name_A and geom_A) else (geom_B if (sys_name == name_B and geom_B) else sys_data)

        # 1. Draw Structure (Geometry)
        x_struct, y_struct = [], []
        # Sort by ID to ensure drawing order
        sorted_ids = sorted(geom_source.keys(), key=lambda x: int(x[1:]))
        
        for eid in sorted_ids:
            if eid not in geom_source: continue
            dat = geom_source[eid]
            ni, nj = dat['ni'], dat['nj']
            
            # Start/End coords
            x_struct.extend([ni[0], nj[0], None])
            y_struct.extend([ni[1], nj[1], None])
        
        fig.add_trace(go.Scatter(
            x=x_struct, y=y_struct, mode='lines+markers', 
            line=dict(color='black', width=3 if sys_name==name_A else 1),
            marker=dict(size=4, color='black'),
            name=f"{sys_name} Structure", opacity=0.3, showlegend=False, hoverinfo='skip'
        ))

        # 2. Draw Diagrams
        for eid, data in sys_data.items():
            if 'x' not in data: continue # Skip empty
            
            # Reconstruct global coordinates for the diagram line
            # Local x runs from 0 to L
            L = data['L']
            c, s = data['cx'], data['cy']
            ni = data['ni']
            
            # Coordinate transformation: Global = Ni + R * Local
            x_glob = ni[0] + c * data['x']
            y_glob = ni[1] + s * data['x']
            
            # Diagram Values
            # Handle Envelopes vs Single Steps
            if type_base == 'Def':
                # Deformations are vectors
                if 'def_x_max' in data: # Envelope
                    vals_pos = data['def_y_max'] * 1000 # mm
                    vals_neg = data['def_y_min'] * 1000
                    fill_mode = True
                else: # Step
                    vals_pos = data['def_y'] * 1000
                    vals_neg = vals_pos
                    fill_mode = False
            else:
                key = type_base
                if f'{key}_max' in data: # Envelope
                    vals_pos = data[f'{key}_max']
                    vals_neg = data[f'{key}_min']
                    fill_mode = True
                else:
                    vals_pos = data[key]
                    vals_neg = vals_pos
                    fill_mode = False

            # Invert M for engineering convention
            # Previous setting: -1.0. User requested reversal -> 1.0
            inv = 1.0 if type_base == 'M' else 1.0
            
            # Determine Local Normal Vector for plotting diagram perpendicular to element
            # Element vector is (c, s). Normal is (-s, c).
            nx, ny = -s, c
            
            # Calculate Diagram Points
            # Pos side
            x_plot_pos = x_glob + nx * vals_pos * scale * inv
            y_plot_pos = y_glob + ny * vals_pos * scale * inv
            
            # Neg side
            x_plot_neg = x_glob + nx * vals_neg * scale * inv
            y_plot_neg = y_glob + ny * vals_neg * scale * inv
            
            # Plot
            if fill_mode:
                # Fill between Max and Min
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_plot_pos, x_plot_neg[::-1]]),
                    y=np.concatenate([y_plot_pos, y_plot_neg[::-1]]),
                    fill='toself', fillcolor=color, opacity=0.2, line=dict(width=0),
                    name=f"{sys_name} {type_base}", showlegend=False, hoverinfo='skip'
                ))
                # Add Lines
                fig.add_trace(go.Scatter(x=x_plot_pos, y=y_plot_pos, mode='lines', line=dict(color=color, width=1.5, dash=line_style), showlegend=False))
                fig.add_trace(go.Scatter(x=x_plot_neg, y=y_plot_neg, mode='lines', line=dict(color=color, width=1.5, dash=line_style), showlegend=False))
            else:
                # Single Line (Step)
                fig.add_trace(go.Scatter(
                    x=x_plot_pos, y=y_plot_pos, mode='lines', 
                    line=dict(color=color, width=2, dash=line_style), 
                    name=f"{sys_name} {type_base}"
                ))
                # Fill to structure
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_glob, x_plot_pos[::-1]]),
                    y=np.concatenate([y_glob, y_plot_pos[::-1]]),
                    fill='toself', fillcolor=color, opacity=0.1, line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))

            # 3. Loads Visualization (Arrows)
            # Only for step views or if specific load case
            if 'loads' in data and len(data['loads']) > 0:
                for load in data['loads']:
                    if load['type'] == 'point':
                        # Draw Arrow
                        p_val = load['params'][0]
                        lx = load['params'][1]
                        
                        # Arrow Base on Structure
                        bas_x = ni[0] + c * lx
                        bas_y = ni[1] + s * lx
                        
                        # Direction Logic
                        if load.get('is_gravity', False):
                            # Vertical Down
                            dx, dy = 0.0, -1.0
                        else:
                            # Perpendicular to Element (Local Y)
                            dx, dy = -s, c
                            
                        # Arrow Length (Scaled)
                        arr_len = 1.5 
                        
                        # Arrow Tail
                        tail_x = bas_x - dx * arr_len
                        tail_y = bas_y - dy * arr_len
                        
                        fig.add_annotation(
                            x=bas_x, y=bas_y, ax=tail_x, ay=tail_y,
                            xref='x', yref='y', axref='x', ayref='y',
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=color,
                            opacity=0.8
                        )
            
            # 4. Annotations Selection
            if annotate and fill_mode:
                # Find Peak Indices
                idx_max = np.argmax(vals_pos)
                idx_min = np.argmin(vals_neg)
                
                # Filter small values
                threshold = max_val * 0.05
                
                # Max Label
                if abs(vals_pos[idx_max]) > threshold:
                    # Direction vector for label shift (normal to element)
                    # We push label away from the diagram
                    perp_x, perp_y = nx, ny 
                    # Coordinate of the peak on the diagram
                    tx = x_plot_pos[idx_max]
                    ty = y_plot_pos[idx_max]
                    
                    ann_candidates.append({
                        'x': tx, 'y': ty, 'text': f"{vals_pos[idx_max]:.1f}", 'color': color,
                        'perp_x': perp_x, 'perp_y': perp_y
                    })

                # Min Label
                if abs(vals_neg[idx_min]) > threshold and idx_min != idx_max:
                    perp_x, perp_y = nx, ny 
                    tx = x_plot_neg[idx_min]
                    ty = y_plot_neg[idx_min]
                     
                    ann_candidates.append({
                        'x': tx, 'y': ty, 'text': f"{vals_neg[idx_min]:.1f}", 'color': color,
                        'perp_x': perp_x, 'perp_y': perp_y
                    })
            elif annotate and not fill_mode:
                # Just Max abs
                vals_abs = np.abs(vals_pos)
                idx = np.argmax(vals_abs)
                val = vals_pos[idx]
                if abs(val) > max_val * 0.05:
                     # Check direction to offset label
                     is_max = (val > 0)
                     tx = x_plot_pos[idx]
                     ty = y_plot_pos[idx]
                     
                     perp_x, perp_y = nx, ny
                     
                     # Simple collision avoidance push
                     push_dist = 0.5
                     if not is_max: push_dist *= -1
                     
                     ann_candidates.append({
                        'x': tx, 'y': ty, 'text': f"{val:.1f}", 'color': color,
                        'perp_x': perp_x, 'perp_y': perp_y
                    })

    if show_A: add_traces(sysA_data, name_A, "blue", "solid", 1)
    if show_B: add_traces(sysB_data, name_B, "red", "dash", -1)
    
    # 5. Solve Annotation Overlaps
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig