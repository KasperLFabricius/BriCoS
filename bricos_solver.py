import streamlit as st
import numpy as np
import copy
import bricos_kernels as kernels

# ==========================================
# 1. CORE FEM CLASSES & FUNCTIONS
# ==========================================

class FrameElement:
    def __init__(self, node_i, node_j, E, geom_data, shear_config):
        # geom_data: {'type': 0(I)/1(H), 'shape': 0(Const)/1(Lin)/2(3pt), 'vals': [v1, v2, v3]}
        # shear_config: {'use': bool, 'b_eff': float, 'nu': float}
        
        # Calculate Geometry
        dx = node_j[0] - node_i[0]
        dy = node_j[1] - node_i[1]
        self.L_calc = np.sqrt(dx**2 + dy**2)
        if self.L_calc < 1e-6: self.L_calc = 1e-6
        
        # --- PROPERTIES & GEOMETRY ESTIMATION ---
        # Handling sub-element interpolation for variable sections
        parent_vals = np.array(geom_data.get('vals', [1.0, 1.0, 1.0]), dtype=np.float64)
        v_type = int(geom_data.get('type', 1)) # Default to 1 (Height) now
        
        # If this is a sub-element of a variable member, we must interpolate properties 
        # based on the parent's full length and this element's local offset.
        local_off = geom_data.get('local_offset', 0.0)
        parent_L = geom_data.get('parent_L', self.L_calc)
        
        # Helper to interpolate at specific relative position
        def interp_val(rel_x):
            shape = int(geom_data.get('shape', 0))
            if shape == 0: return parent_vals[0]
            if shape == 1: 
                # BUGFIX: Linear taper uses index 0 (Start) and index 2 (End).
                # Index 1 is reserved for Mid point in 3-point shapes.
                return parent_vals[0]*(1-rel_x) + parent_vals[2]*rel_x 
            if shape == 2: # 3-Point
                if rel_x <= 0.5:
                    s = rel_x * 2.0
                    return parent_vals[0]*(1-s) + parent_vals[1]*s
                else:
                    s = (rel_x - 0.5) * 2.0
                    return parent_vals[1]*(1-s) + parent_vals[2]*s
            return parent_vals[0]

        # Calculate interpolated start/mid/end for THIS sub-element
        rel_start = local_off / parent_L
        rel_end = (local_off + self.L_calc) / parent_L
        rel_mid = (rel_start + rel_end) / 2.0
        
        # Effective values for this sub-element
        v_start = interp_val(rel_start)
        v_end = interp_val(rel_end)
        
        # --- FIX: DETECT "FAKE" TAPER ---
        # If the user defined a Linear Taper but Start == End, treat as Constant (Prismatic).
        # This forces the use of Exact Analytical Kernels instead of Numerical Integration,
        # preventing integration errors (aliasing) for point loads on constant sections.
        is_effectively_constant = (abs(v_start - v_end) < 1e-6)
        
        eff_vals = [v_start, v_end, v_end] 
        # Default shape is linear (1) unless overridden
        eff_shape = 1 if geom_data.get('shape', 0) != 0 else 0
        
        if is_effectively_constant:
            eff_shape = 0
            eff_vals = [v_start, v_start, v_start]

        # Defaults
        b_eff = shear_config.get('b_eff', 1.0)
        if b_eff < 0.01: b_eff = 1.0
        
        if v_type == 1:
            # Input was H (Height).
            h_avg = (v_start + v_end)/2.0
            I_avg = (b_eff * h_avg**3) / 12.0
            A_approx = b_eff * h_avg
        else:
            # Input was I.
            I_avg = (v_start + v_end)/2.0
            h_est = (12.0 * I_avg / b_eff)**(1.0/3.0)
            A_approx = b_eff * h_est

        # --- SAFETY FLOOR ---
        MIN_I = 1e-9  # m^4
        MIN_A = 1e-6  # m^2
        if I_avg < MIN_I: I_avg = MIN_I
        if A_approx < MIN_A: A_approx = MIN_A
        if v_start < 1e-6: v_start = 1e-6
        if v_end < 1e-6: v_end = 1e-6

        # --- SHEAR DEFORMATION LOGIC ---
        phi_s = 0.0
        G_val = 0.0
        h_shear_calc = (12.0 * I_avg / b_eff)**(1.0/3.0)
        As_avg = (5.0/6.0) * b_eff * h_shear_calc

        if shear_config.get('use', False):
            nu = shear_config.get('nu', 0.2)
            G_val = float(E) / (2.0 * (1.0 + nu))
            denom = (G_val * As_avg * self.L_calc**2)
            if denom > 1e-9:
                phi_s = (12.0 * float(E) * I_avg) / denom

        # --- STORE GEOM DATA FOR LOAD KERNEL ---
        self.eff_shape = eff_shape
        self.v_type = v_type
        self.eff_vals = np.array([v_start, v_end, 0.0])
        self.b_eff = float(b_eff)
        self.As_avg = float(As_avg)
        self.G_val = float(G_val)

        # --- KERNEL SELECTION ---
        if eff_shape == 0:
            # PRISMATIC (CONSTANT) PATH
            if v_type == 1: 
                # If constant height, I is constant
                I_c = (b_eff * v_start**3) / 12.0
            else: 
                I_c = v_start
            
            if I_c < MIN_I: I_c = MIN_I
            
            k_loc, k_glob, T, L, cx, cy = kernels.jit_beam_matrices(
                node_i[0], node_i[1], node_j[0], node_j[1], float(E), I_c, float(A_approx), phi_s
            )
            self.I = I_c
        else:
            # NON-PRISMATIC (TAPERED) PATH
            k_loc, k_glob, T, L, cx, cy = kernels.jit_non_prismatic_matrices(
                node_i[0], node_i[1], node_j[0], node_j[1], float(E), float(G_val),
                1, v_type, self.eff_vals, float(A_approx), float(As_avg),
                float(b_eff) 
            )
            # For variable elements, store start I as representative
            if v_type == 1:
                self.I = (b_eff * v_start**3) / 12.0
            else:
                self.I = v_start
            
            if self.I < MIN_I: self.I = MIN_I

        self.E = E
        self.k_local = k_loc
        self.k_global = k_glob
        self.T = T
        self.L = L
        self.cx = cx
        self.cy = cy

    def get_fixed_end_forces(self, load_type, params):
        # 1. Determine Load Type Integer
        l_int = 0
        if load_type == 'point': l_int = 1
        elif load_type == 'distributed_trapezoid': l_int = 2
        
        # 2. Pad Params to 4 for numba kernel consistency
        p_arr = np.zeros(4)
        n_p = len(params)
        for i in range(min(4, n_p)): p_arr[i] = params[i]
        
        # 3. Choose Kernel based on Element Shape
        if self.eff_shape == 0:
            # Use Exact Analytical Formulas for Prismatic
            if load_type == 'point':
                return kernels.jit_fef_point(params[0], params[1], self.L)
            elif load_type == 'distributed_trapezoid':
                return kernels.jit_fef_trapezoid(params[0], params[1], params[2], params[3], self.L)
        else:
            # Use Numerical Integration for Non-Prismatic
            return kernels.jit_numerical_fef(
                l_int, p_arr, self.L, float(self.E), self.G_val,
                self.eff_shape, self.v_type, self.eff_vals, self.b_eff, self.As_avg
            )
            
        return np.zeros(6)

def build_stiffness_matrix(nodes, elements, restraints_stiffness, shear_config):
    node_keys = sorted(nodes.keys())
    node_map = {nid: i for i, nid in enumerate(node_keys)}
    NDOF = 3 * len(nodes)
    K_global = np.zeros((NDOF, NDOF))
    elem_objects = []
    
    # 1. Assemble Element Stiffness
    for el_data in elements:
        ni_id, nj_id = el_data['nodes']
        el = FrameElement(nodes[ni_id], nodes[nj_id], el_data['E'], el_data, shear_config)
        elem_objects.append(el)
        idx_i, idx_j = node_map[ni_id]*3, node_map[nj_id]*3
        indices = [idx_i, idx_i+1, idx_i+2, idx_j, idx_j+1, idx_j+2]
        K_global[np.ix_(indices, indices)] += el.k_global
        
    # 2. Apply Boundary Springs / Restraints
    for nid, stiff in restraints_stiffness.items():
        if nid in node_map:
            idx = node_map[nid]*3
            for i in range(3):
                if stiff[i] is not None: 
                    K_global[idx+i, idx+i] += stiff[i]
                
    return K_global, node_map, elem_objects, NDOF

def get_detailed_results_optimized(elem_objects, elements_source_data, nodes, D_total, loads_map, mesh_size=0.5):
    node_keys = sorted(nodes.keys())
    node_map = {nid: i for i, nid in enumerate(node_keys)}
    results = {}
    
    for i, el_data in enumerate(elements_source_data):
        el = elem_objects[i]
        ni, nj = el_data['nodes']
        idx_i, idx_j = node_map[ni]*3, node_map[nj]*3
        d_glob = np.concatenate([D_total[idx_i:idx_i+3], D_total[idx_j:idx_j+3]])
        d_loc = el.T @ d_glob
        
        f_start = el.k_local @ d_loc
        active_loads = loads_map.get(i, [])
        
        for load in active_loads:
            p_raw = load['params']
            p_final = list(p_raw)
            if load.get('is_gravity', False):
                p_final[0] *= el.cx
                if load['type'] == 'distributed_trapezoid':
                    p_final[1] *= el.cx
            f_start += el.get_fixed_end_forces(load['type'], p_final)
            
        load_arr = np.zeros((len(active_loads), 5))
        for k, load in enumerate(active_loads):
            p_raw = load['params']
            val_1 = p_raw[0]
            val_2 = p_raw[1] if len(p_raw) > 1 else 0.0
            if load.get('is_gravity', False):
                val_1 *= el.cx
                val_2 *= el.cx
            p = load['params']
            if load['type'] == 'point':
                load_arr[k, :] = [1, val_1, p[1], 0.0, 0.0]
            elif load['type'] == 'distributed_trapezoid':
                load_arr[k, :] = [2, val_1, val_2, p[2], p[3]]

        num_pts = max(3, int(el.L / mesh_size) + 1)
        x_vals, M_vals, V_vals, N_vals = kernels.jit_internal_forces(
            el.L, f_start[:3], num_pts, load_arr
        )
        ux, uy = kernels.jit_disp_shape(d_loc, el.L, num_pts)
        def_x = el.cx * ux - el.cy * uy
        def_y = el.cy * ux + el.cx * uy
        
        results[el_data['id']] = {
            'x': x_vals, 
            'M': M_vals, 'V': V_vals, 'N': N_vals,
            'def_x': def_x, 'def_y': def_y,
            'L': el.L, 'ni': nodes[ni], 'nj': nodes[nj], 'cx': el.cx, 'cy': el.cy,
            'loads': active_loads,
            'f_start_local': np.array([N_vals[0], V_vals[0], M_vals[0]]),
            'f_end_local': np.array([-N_vals[-1], -V_vals[-1], -M_vals[-1]]),
            'ni_id': ni, 'nj_id': nj,
            'parent': el_data.get('parent'),
            'local_offset': el_data.get('local_offset', 0.0)
        }
    return results

def aggregate_member_results(detailed_results, global_loads_override=None):
    agg_res = {}
    grouped = {}
    for eid, res in detailed_results.items():
        parent = res.get('parent', eid)
        if parent not in grouped: grouped[parent] = []
        grouped[parent].append(res)
    
    for parent, parts in grouped.items():
        parts.sort(key=lambda x: x['local_offset'])
        base = parts[0]
        last = parts[-1]
        total_L = sum(p['L'] for p in parts)
        
        x_all = []
        M_all, V_all, N_all = [], [], []
        dx_all, dy_all = [], []
        raw_loads_all = []
        
        ni_final = base['ni']
        nj_final = last['nj']
        ni_id_final = base['ni_id']
        nj_id_final = last['nj_id']
        
        for p in parts:
            offset = p['local_offset']
            x_all.append(p['x'] + offset)
            M_all.append(p['M'])
            V_all.append(p['V'])
            N_all.append(p['N'])
            dx_all.append(p['def_x'])
            dy_all.append(p['def_y'])
            
            if not global_loads_override or parent not in global_loads_override:
                for load in p['loads']:
                    new_load = copy.deepcopy(load)
                    params = new_load['params']
                    if load['type'] == 'point':
                        params[1] += offset
                    elif load['type'] == 'distributed_trapezoid':
                        params[2] += offset
                        params[3] += offset
                    raw_loads_all.append(new_load)
        
        final_loads = []
        if global_loads_override and parent in global_loads_override:
            final_loads = global_loads_override[parent]
        else:
            def get_sort_tuple(d):
                p = d['params']
                if d['type'] == 'point': x_s = p[1]
                elif d['type'] == 'distributed_trapezoid': x_s = p[2]
                else: x_s = 0.0
                return (d['type'], d.get('is_gravity', False), x_s)
            raw_loads_all.sort(key=get_sort_tuple)

            for curr in raw_loads_all:
                if not final_loads:
                    final_loads.append(curr)
                    continue
                prev = final_loads[-1]
                is_trap = (curr['type'] == 'distributed_trapezoid') and (prev['type'] == 'distributed_trapezoid')
                same_grav = (curr['is_gravity'] == prev['is_gravity'])
                
                if is_trap and same_grav:
                    p_prev = prev['params']
                    p_curr = curr['params']
                    x_match = abs(p_prev[3] - p_curr[2]) < 1e-4
                    q_match = abs(p_prev[1] - p_curr[0]) < 1e-4
                    len_prev = max(1e-6, p_prev[3] - p_prev[2])
                    len_curr = max(1e-6, p_curr[3] - p_curr[2])
                    slope_prev = (p_prev[1] - p_prev[0]) / len_prev
                    slope_curr = (p_curr[1] - p_curr[0]) / len_curr
                    slope_match = abs(slope_prev - slope_curr) < 1e-4
                    if x_match and q_match and slope_match:
                        p_prev[1] = p_curr[1] # Update q_end
                        p_prev[3] = p_curr[3] # Update x_end
                        continue 
                final_loads.append(curr)

        agg_res[parent] = {
            'x': np.concatenate(x_all),
            'M': np.concatenate(M_all),
            'V': np.concatenate(V_all),
            'N': np.concatenate(N_all),
            'def_x': np.concatenate(dx_all),
            'def_y': np.concatenate(dy_all),
            'L': total_L,
            'cx': base['cx'], 'cy': base['cy'],
            'ni': ni_final, 'nj': nj_final,
            'ni_id': ni_id_final, 'nj_id': nj_id_final,
            'loads': final_loads, 
            'f_start_local': base['f_start_local'],
            'f_end_local': last['f_end_local']
        }
        
    return agg_res

def calculate_reactions(nodes, detailed_results):
    reactions = {nid: np.zeros(3) for nid in nodes}
    for eid, data in detailed_results.items():
        if 'f_start_local' not in data or 'f_end_local' not in data: continue
        c, s = data['cx'], data['cy']
        T_trans = np.array([[c, -s, 0],[s,  c, 0],[0,  0, 1]])
        f_start_glob = T_trans @ data['f_start_local'] 
        reactions[data['ni_id']] += f_start_glob
        f_end_glob = T_trans @ data['f_end_local']
        reactions[data['nj_id']] += f_end_glob
    return reactions

def get_safe_error_result():
    empty_dict = {} 
    return {
        'Selfweight': empty_dict, 'Soil': empty_dict, 'Surcharge': empty_dict,
        'Vehicle Envelope A': empty_dict, 'Vehicle Envelope B': empty_dict,
        'Vehicle Steps A': [], 'Vehicle Steps B': [],
        'phi_calc': 1.0, 'phi_log': ["System Unstable or Empty"]
    }

# ==========================================
# 2. MAIN SOLVER CONTROLLER
# ==========================================

@st.cache_data(show_spinner=False)
def run_raw_analysis(params, phi_val_override=None):
    phi_mode = params.get('phi_mode', 'Calculate')
    
    # --- PHI CALCULATION ---
    phi_log = []
    calc_phi = 1.0
    
    if phi_mode == "Manual":
        calc_phi = params.get('phi', 1.0)
        phi_log.append(f"Manual input: {calc_phi}")
    else:
        lengths_for_phi = []
        comp_desc = []
        
        if params['mode'] == 'Frame':
            h_left_total = params['h_list'][0]
            lengths_for_phi.append(h_left_total)
            comp_desc.append(f"LeftLeg({h_left_total:.2f}m)")
            for i in range(params['num_spans']):
                L = params['L_list'][i]
                lengths_for_phi.append(L)
                comp_desc.append(f"Span{i+1}({L:.2f}m)")
            end_idx = params['num_spans']
            h_right_total = params['h_list'][end_idx]
            lengths_for_phi.append(h_right_total)
            comp_desc.append(f"RightLeg({h_right_total:.2f}m)")
            
        else: 
            for i in range(params['num_spans']):
                L = params['L_list'][i]
                lengths_for_phi.append(L)
                comp_desc.append(f"Span{i+1}({L:.2f}m)")

        if lengths_for_phi:
            n = len(lengths_for_phi)
            if n == 1:
                L_phi = lengths_for_phi[0]
                k_fac = 1.0
                phi_log.append(f"Case 5.1 (Single Element)")
            else:
                if n == 2: k_fac = 1.2
                elif n == 3: k_fac = 1.3
                elif n == 4: k_fac = 1.4
                else: k_fac = 1.5
                L_max = max(lengths_for_phi)
                L_mean = sum(lengths_for_phi) / n
                L_phi_calc = k_fac * L_mean
                L_phi = max(L_phi_calc, L_max)
                phi_log.append(f"Case 5.2/5.3 (n={n}, k={k_fac})")
                phi_log.append(f"Components: {', '.join(comp_desc)}")
                phi_log.append(f"L_mean = {sum(lengths_for_phi):.2f}/{n} = {L_mean:.2f} m")
                phi_log.append(f"L_phi = {k_fac} * {L_mean:.2f} = {L_phi_calc:.3f} m (Min {L_max:.2f})")

            if L_phi <= 5.0:
                calc_phi = 1.25
                phi_log.append(f"L_phi <= 5.0m: Phi set to upper limit (1.25)")
            elif L_phi >= 50.0:
                calc_phi = 1.05
                phi_log.append(f"L_phi >= 50.0m: Phi set to lower limit (1.05)")
            else:
                raw_val = 1.25 - (L_phi - 5.0) / 225.0
                calc_phi = raw_val
                phi_log.append(f"5.0 < L_phi < 50.0: Calc Formula applied.")
                phi_log.append(f"Phi = 1.25 - ({L_phi:.3f}-5)/225 = {raw_val:.3f}")
            phi_log.append(f"Final Phi = {calc_phi:.3f}")
        else:
            phi_log.append("Geometry invalid/empty. Phi=1.0")

    phi = phi_val_override if phi_val_override is not None else calc_phi

    nodes = {}
    elems_base = []
    restraints = {}
    model_props = {'Spans': {}, 'Walls': {}}
    
    num_spans = params['num_spans']
    num_supp = num_spans + 1
    supports_cfg = params.get('supports', [])
    E_default = params.get('E', 30e6)
    E_spans = params.get('E_span_list', [E_default] * num_spans)
    E_walls = params.get('E_wall_list', [E_default] * num_supp)
    mesh_size = params.get('mesh_size', 0.5)

    def get_geom_data(idx, prefix_list_key, prefix_new_key):
        adv_key = f"{prefix_new_key}_{idx}"
        if adv_key in params:
            return params[adv_key]
        val = float(params[prefix_list_key][idx])
        return {'type': 1, 'shape': 0, 'vals': [val, val, val]}

    next_node_id = 1000 

    def create_member_mesh(start_xy, end_xy, start_node_id, end_node_id, props, member_type):
        nonlocal next_node_id
        dx = end_xy[0] - start_xy[0]
        dy = end_xy[1] - start_xy[1]
        L_total = np.sqrt(dx**2 + dy**2)
        if L_total < 1e-6: return []

        n_seg = max(1, int(np.ceil(L_total / mesh_size)))
        d_vec = np.array([dx, dy]) / n_seg
        
        nodes[start_node_id] = start_xy
        nodes[end_node_id] = end_xy
        
        sub_elems = []
        prev_id = start_node_id
        
        for i in range(n_seg):
            is_last = (i == n_seg - 1)
            curr_node_id = end_node_id if is_last else next_node_id
            
            if not is_last:
                pos = np.array(start_xy) + d_vec * (i + 1)
                nodes[curr_node_id] = tuple(pos)
                next_node_id += 1
                
            sub_id = f"{props['parent']}_{i}"
            local_off = i * (L_total / n_seg)
            
            elem_data = {
                **props['geom'],
                'id': sub_id,
                'nodes': (prev_id, curr_node_id),
                'E': props['E'],
                'parent': props['parent'],
                'local_offset': local_off,
                'parent_L': L_total
            }
            sub_elems.append(elem_data)
            prev_id = curr_node_id
            
        return sub_elems

    # --- GEOMETRY CONSTRUCTION ---
    top_node_coords = {} 
    curr_x = 0.0
    curr_y = 0.0
    top_node_coords[0] = (0.0, 0.0) 
    
    for i in range(num_spans):
        L_horiz = params['L_list'][i]
        g_data = get_geom_data(i, 'Is_list', 'span_geom')
        d_y = 0.0
        align_type = g_data.get('align_type', 0)
        
        if align_type == 1:
            mode = g_data.get('incline_mode', 0)
            val = float(g_data.get('incline_val', 0.0))
            if mode == 0: d_y = (val / 100.0) * L_horiz
            else: d_y = val
        
        next_x = curr_x + L_horiz
        next_y = curr_y + d_y
        top_node_coords[i+1] = (next_x, next_y)
        curr_x = next_x
        curr_y = next_y

    TOLERANCE = 1e-4

    # 1. BUILD WALLS
    if params['mode'] == 'Frame':
        for i in range(num_supp):
            h = params['h_list'][i]
            nid_b, nid_t = 100+i, 200+i
            if i < len(supports_cfg): k_vec = supports_cfg[i]['k']
            else: k_vec = [1e14, 1e14, 1e14] 
            
            if h < TOLERANCE:
                t_x, t_y = top_node_coords[i]
                nodes[nid_t] = (t_x, t_y)
                restraints[nid_t] = k_vec
                continue

            t_x, t_y = top_node_coords[i]
            base_xy = (t_x, t_y - h)
            top_xy = (t_x, t_y)
            e_val = E_walls[i] if i < len(E_walls) else E_default
            g_data = get_geom_data(i, 'Iw_list', 'wall_geom')
            
            walls_subs = create_member_mesh(
                base_xy, top_xy, nid_b, nid_t, 
                {'parent': f'W{i+1}', 'E': e_val, 'geom': g_data}, 'Wall'
            )
            elems_base.extend(walls_subs)
            model_props['Walls'][f'W{i+1}'] = {'E': e_val}
            restraints[nid_b] = k_vec
            
    else: 
        for i in range(num_supp):
            nid_t = 200+i
            t_x, t_y = top_node_coords[i]
            nodes[nid_t] = (t_x, t_y)
            if i < len(supports_cfg): k_vec = supports_cfg[i]['k']
            else:
                if i == 0: k_vec = [1e14, 1e14, 0.0]
                else: k_vec = [0.0, 1e14, 0.0]
            restraints[nid_t] = k_vec

    # 2. BUILD SPANS
    for i in range(num_spans):
        if params['L_list'][i] < TOLERANCE: continue
        nid_s, nid_e = 200+i, 200+i+1
        start_xy = top_node_coords[i]
        end_xy = top_node_coords[i+1]
        e_val = E_spans[i] if i < len(E_spans) else E_default
        g_data = get_geom_data(i, 'Is_list', 'span_geom')
        
        span_subs = create_member_mesh(
            start_xy, end_xy, nid_s, nid_e,
            {'parent': f'S{i+1}', 'E': e_val, 'geom': g_data}, 'Span'
        )
        elems_base.extend(span_subs)
        model_props['Spans'][f'S{i+1}'] = {'E': e_val}

    if len(elems_base) == 0:
        return get_safe_error_result(), {}, {}, "No valid structural elements defined."

    shear_config = {
        'use': params.get('use_shear_def', False),
        'b_eff': params.get('b_eff', 1.0),
        'nu': params.get('nu', 0.2)
    }

    K_glob, node_map, elem_objects, NDOF = build_stiffness_matrix(nodes, elems_base, restraints, shear_config)
    
    try:
        K_inv = np.linalg.inv(K_glob)
    except np.linalg.LinAlgError:
        raise ValueError("Structural Instability Detected: The model is insufficiently constrained (Mechanism). Please check boundary conditions.")

    def solve_static(loads_dict):
        F = np.zeros(NDOF)
        for idx, load_list in loads_dict.items():
            el = elem_objects[idx]
            for load in load_list:
                p_raw = load['params']
                p_final = list(p_raw)
                if load.get('is_gravity', False):
                    p_final[0] *= el.cx
                    if load['type'] == 'distributed_trapezoid':
                        p_final[1] *= el.cx
                f_loc = el.get_fixed_end_forces(load['type'], p_final)
                f_glob = el.T.T @ f_loc
                ni, nj = elems_base[idx]['nodes']
                idx_i, idx_j = node_map[ni]*3, node_map[nj]*3
                indices = [idx_i, idx_i+1, idx_i+2, idx_j, idx_j+1, idx_j+2]
                F[indices] -= f_glob
        D = K_inv @ F
        return get_detailed_results_optimized(elem_objects, elems_base, nodes, D, loads_dict, params.get('mesh_size', 0.5))

    def add_member_load(target_map, parent_id, load_type, is_gravity, params_list):
        indices = [k for k, el in enumerate(elems_base) if el['parent'] == parent_id]
        if not indices: return
        indices.sort(key=lambda k: elems_base[k]['local_offset'])
        
        for idx in indices:
            el_data = elems_base[idx]
            el_obj = elem_objects[idx]
            loc_start = el_data['local_offset']
            loc_end = loc_start + el_obj.L
            
            if load_type == 'distributed_trapezoid':
                # FIXED: Unpack correctly so variable names match logic below
                q_s_glob, q_e_glob, x_s_glob, L_load = params_list
                x_e_glob = x_s_glob + L_load
                overlap_start = max(loc_start, x_s_glob)
                overlap_end = min(loc_end, x_e_glob)
                
                if overlap_end > overlap_start + 1e-6:
                    local_x_s = overlap_start - loc_start
                    len_glob = x_e_glob - x_s_glob
                    if len_glob > 1e-9:
                        q_sub_start = q_s_glob + (q_e_glob - q_s_glob) * (overlap_start - x_s_glob) / len_glob
                        q_sub_end = q_s_glob + (q_e_glob - q_s_glob) * (overlap_end - x_s_glob) / len_glob
                    else:
                        q_sub_start = q_s_glob
                        q_sub_end = q_e_glob
                    if idx not in target_map: target_map[idx] = []
                    target_map[idx].append({
                        'type': 'distributed_trapezoid',
                        'is_gravity': is_gravity,
                        'params': [q_sub_start, q_sub_end, local_x_s, overlap_end - overlap_start]
                    })
                    
            elif load_type == 'point':
                P_val, x_pos_glob = params_list
                if loc_start <= x_pos_glob <= loc_end:
                    local_x = x_pos_glob - loc_start
                    if idx not in target_map: target_map[idx] = []
                    target_map[idx].append({
                        'type': 'point',
                        'is_gravity': is_gravity,
                        'params': [P_val, local_x]
                    })

    # 1. Selfweight
    sw_loads_map = {}
    sw_global_loads = {} 

    for i in range(num_spans):
        val = params['sw_list'][i]
        if val != 0:
            pid = f'S{i+1}'
            indices = [k for k, el in enumerate(elems_base) if el['parent'] == pid]
            true_L = sum(elem_objects[k].L for k in indices)
            
            if pid not in sw_global_loads: sw_global_loads[pid] = []
            sw_global_loads[pid].append({
                'type': 'distributed_trapezoid', 'is_gravity': True,
                'params': [val, val, 0, true_L]
            })
            for idx in indices:
                el_L = elem_objects[idx].L
                if idx not in sw_loads_map: sw_loads_map[idx] = []
                sw_loads_map[idx].append({
                    'type': 'distributed_trapezoid', 'is_gravity': True, 
                    'params': [val, val, 0, el_L]
                })

    res_sw_detailed = solve_static(sw_loads_map)
    res_sw = aggregate_member_results(res_sw_detailed, sw_global_loads)

    # 2. Soil
    soil_loads_map = {}
    soil_global_loads = {}
    
    if params['mode'] == 'Frame':
        for s in params.get('soil', []):
            if s['wall_idx'] < num_supp:
                pid = f'W{s["wall_idx"]+1}'
                sign = 1.0 if s['face'] == 'L' else -1.0 
                if pid not in soil_global_loads: soil_global_loads[pid] = []
                soil_global_loads[pid].append({
                    'type': 'distributed_trapezoid', 'is_gravity': False,
                    'params': [sign*s['q_bot'], sign*s['q_top'], 0.0, s['h']]
                })
                add_member_load(soil_loads_map, pid, 'distributed_trapezoid', False, 
                                [sign*s['q_bot'], sign*s['q_top'], 0.0, s['h']])
                                
    res_soil = aggregate_member_results(solve_static(soil_loads_map), soil_global_loads)

    # 3. Surcharge
    surch_loads_map = {}
    surch_global_loads = {}
    
    if params['mode'] == 'Frame':
        for sur in params.get('surcharge', []):
            if sur['wall_idx'] < num_supp:
                pid = f'W{sur["wall_idx"]+1}'
                sign = -1.0 if sur['face'] == 'L' else 1.0
                if pid not in surch_global_loads: surch_global_loads[pid] = []
                surch_global_loads[pid].append({
                    'type': 'distributed_trapezoid', 'is_gravity': False,
                    'params': [sign*sur['q'], sign*sur['q'], 0.0, sur['h']]
                })
                add_member_load(surch_loads_map, pid, 'distributed_trapezoid', False,
                                [sign*sur['q'], sign*sur['q'], 0.0, sur['h']])
                                
    res_surch = aggregate_member_results(solve_static(surch_loads_map), surch_global_loads)

    def get_empty_env():
        res_empty_det = solve_static({})
        res_empty_agg = aggregate_member_results(res_empty_det)
        env = {}
        if res_empty_agg:
            for eid, data in res_empty_agg.items():
                env[eid] = {
                    'M_max': np.zeros_like(data['M']), 'M_min': np.zeros_like(data['M']), 
                    'V_max': np.zeros_like(data['V']), 'V_min': np.zeros_like(data['V']),
                    'N_max': np.zeros_like(data['N']), 'N_min': np.zeros_like(data['N']),
                    'def_x_max': np.zeros_like(data['def_x']), 'def_x_min': np.zeros_like(data['def_x']),
                    'def_y_max': np.zeros_like(data['def_y']), 'def_y_min': np.zeros_like(data['def_y']),
                    'base': data
                }
        return env

    veh_env_A = get_empty_env()
    veh_env_B = get_empty_env()
    
    def run_stepping(vehicle_key, env_to_fill):
        v_loads_raw = np.array(params[vehicle_key]['loads']) * 9.81
        v_dists_raw = np.cumsum(params[vehicle_key]['spacing'])
        veh_dir = params.get('vehicle_direction', 'Forward')
        directions_to_run = ['Forward', 'Reverse'] if veh_dir == 'Both' else [veh_dir]
        steps_out = {'Forward': [], 'Reverse': []}
        
        sp_start_x = []
        sp_lens = []
        sp_el_indices = []
        c_x = 0
        sp_elems_info = [] 
        
        for sp_i in range(num_spans):
            pid = f'S{sp_i+1}'
            indices = [k for k, el in enumerate(elems_base) if el['parent'] == pid]
            indices.sort(key=lambda k: elems_base[k]['local_offset'])
            for k in indices:
                true_L = elem_objects[k].L
                sp_start_x.append(c_x)
                sp_lens.append(true_L)
                sp_el_indices.append(k)
                sp_elems_info.append((c_x, true_L, k))
                c_x += true_L

        sp_start_x = np.array(sp_start_x, dtype=np.float64)
        sp_lens = np.array(sp_lens, dtype=np.float64)
        sp_el_indices = np.array(sp_el_indices, dtype=np.int32)

        total_structure_len = c_x 
        n_elems = len(elem_objects)
        el_L = np.zeros(n_elems)
        el_T = np.zeros((n_elems, 6, 6))
        el_k_local = np.zeros((n_elems, 6, 6))
        el_dof_indices = np.zeros((n_elems, 6), dtype=np.int32)
        
        for k in range(n_elems):
            el_obj = elem_objects[k]
            el_L[k] = el_obj.L
            el_T[k] = el_obj.T
            el_k_local[k] = el_obj.k_local
            ni, nj = elems_base[k]['nodes']
            idx_i, idx_j = node_map[ni]*3, node_map[nj]*3
            el_dof_indices[k] = [idx_i, idx_i+1, idx_i+2, idx_j, idx_j+1, idx_j+2]
        
        mesh_sz = params.get('mesh_size', 0.5)
        max_len = np.max(el_L) if len(el_L) > 0 else 1.0
        n_pts_kernel = max(3, int(max_len / mesh_sz) + 1)
        
        S_matrices = kernels.jit_precompute_stress_recovery(n_elems, n_pts_kernel, el_L, el_k_local)
        env_results_accum = np.zeros((n_elems, n_pts_kernel, 10))
        is_first_run = True

        for current_dir in directions_to_run:
            if len(v_loads_raw) == 0: continue
            max_d = max(v_dists_raw) if len(v_dists_raw) > 0 else 0
            step_val = params.get('step_size', 0.2)
            
            if current_dir == 'Forward':
                x_steps = np.arange(-max_d-1.0, total_structure_len + max_d + 1.0, step_val)
                v_dists_run = v_dists_raw
            else:
                x_steps = np.arange(total_structure_len + max_d + 1.0, -max_d-1.0, -step_val)
                v_dists_run = -v_dists_raw
                
            total_steps = len(x_steps)
            v_steps_res_list = []
            CHUNK_SIZE = 2000
            
            for start_idx in range(0, total_steps, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, total_steps)
                x_chunk = x_steps[start_idx:end_idx]
                n_chunk = len(x_chunk)
                is_init_chunk = (is_first_run and start_idx == 0)
                
                F_chunk = kernels.jit_build_batch_F(NDOF, n_chunk, x_chunk, v_loads_raw, v_dists_run, sp_start_x, sp_lens, sp_el_indices, el_L, el_T, el_dof_indices)
                D_chunk = K_inv @ F_chunk
                
                kernels.jit_envelope_batch_parallel(
                    n_chunk, n_elems, n_pts_kernel,
                    x_chunk, v_loads_raw, v_dists_run,
                    sp_start_x, sp_lens, sp_el_indices,
                    D_chunk, el_dof_indices, el_T, el_L,
                    S_matrices,
                    env_results_accum, 
                    is_init_chunk
                )
                
                for i_local in range(n_chunk):
                    x_front = x_chunk[i_local]
                    D_step = D_chunk[:, i_local]
                    step_loads_map = {}
                    has_loads = False
                    for ax_i, d in enumerate(v_dists_run):
                        ax_x = x_front - d
                        for (start_x, L_span, el_idx) in sp_elems_info:
                            local_x = ax_x - start_x
                            if 0 <= local_x <= L_span:
                                P_val = v_loads_raw[ax_i]
                                if el_idx not in step_loads_map: step_loads_map[el_idx] = []
                                load_p = {'type': 'point', 'is_gravity': True, 'params': [P_val, local_x]}
                                step_loads_map[el_idx].append(load_p)
                                has_loads = True
                                break
                    if has_loads:
                        raw_step = get_detailed_results_optimized(elem_objects, elems_base, nodes, D_step, step_loads_map, params.get('mesh_size', 0.5))
                        agg_step = aggregate_member_results(raw_step)
                        v_steps_res_list.append({'x': x_front, 'res': agg_step})
            
            steps_out[current_dir] = v_steps_res_list
            is_first_run = False 

        if len(v_loads_raw) > 0 and env_to_fill:
            parent_map = {}
            for k, el_data in enumerate(elems_base):
                pid = el_data['parent']
                if pid not in parent_map: parent_map[pid] = []
                parent_map[pid].append((k, el_data['local_offset'], el_L[k]))
            
            for pid, parts in parent_map.items():
                if pid not in env_to_fill: continue
                target_env = env_to_fill[pid]
                parts.sort(key=lambda x: x[1])
                M_max_list, M_min_list = [], []
                V_max_list, V_min_list = [], []
                N_max_list, N_min_list = [], []
                dx_max_list, dx_min_list = [], []
                dy_max_list, dy_min_list = [], []
                
                for (idx, offset, L_sub) in parts:
                    valid_res = env_results_accum[idx, :n_pts_kernel, :]
                    M_max_list.append(valid_res[:, 0])
                    M_min_list.append(valid_res[:, 1])
                    V_max_list.append(valid_res[:, 2])
                    V_min_list.append(valid_res[:, 3])
                    N_max_list.append(valid_res[:, 4])
                    N_min_list.append(valid_res[:, 5])
                    dx_max_list.append(valid_res[:, 6])
                    dx_min_list.append(valid_res[:, 7])
                    dy_max_list.append(valid_res[:, 8])
                    dy_min_list.append(valid_res[:, 9])
                    
                target_env['M_max'] = np.concatenate(M_max_list)
                target_env['M_min'] = np.concatenate(M_min_list)
                target_env['V_max'] = np.concatenate(V_max_list)
                target_env['V_min'] = np.concatenate(V_min_list)
                target_env['N_max'] = np.concatenate(N_max_list)
                target_env['N_min'] = np.concatenate(N_min_list)
                target_env['def_x_max'] = np.concatenate(dx_max_list)
                target_env['def_x_min'] = np.concatenate(dx_min_list)
                target_env['def_y_max'] = np.concatenate(dy_max_list)
                target_env['def_y_min'] = np.concatenate(dy_min_list)

        return steps_out

    steps_A = run_stepping('vehicle', veh_env_A)
    steps_B = run_stepping('vehicleB', veh_env_B)

    return {
        'Selfweight': res_sw,
        'Soil': res_soil,
        'Surcharge': res_surch,
        'Vehicle Envelope A': veh_env_A,
        'Vehicle Envelope B': veh_env_B,
        'Vehicle Steps A': steps_A['Forward'],
        'Vehicle Steps A_Rev': steps_A['Reverse'],
        'Vehicle Steps B': steps_B['Forward'],
        'Vehicle Steps B_Rev': steps_B['Reverse'],
        'phi_calc': calc_phi,
        'phi_log': phi_log,
        'Reactions': calculate_reactions(nodes, res_sw) 
    }, nodes, model_props, 0

def combine_results(raw_res, params, result_mode="Design (ULS)"):
    KFI = params.get('KFI', 1.0)
    gamma_g = params.get('gamma_g', 1.0) 
    gamma_j = params.get('gamma_j', 1.0)
    gamma_vA = params.get('gamma_veh', 1.0) 
    gamma_vB = params.get('gamma_vehB', 1.0) 
    phi = raw_res['phi_calc'] if params.get('phi_mode') == 'Calculate' else params.get('phi', 1.0)
    
    if result_mode == "Design (ULS)":
        f_sw = KFI * gamma_g 
        f_soil = KFI * gamma_j
        f_vehA = KFI * gamma_vA * phi
        f_vehB = KFI * gamma_vB * phi
        f_surch = KFI * gamma_vA 
    elif result_mode == "Characteristic (SLS)":
        f_sw = 1.0; f_soil = 1.0; f_vehA = 1.0 * phi; f_vehB = 1.0 * phi; f_surch = 1.0
    else:
        f_sw = 1.0; f_soil = 1.0; f_vehA = 1.0; f_vehB = 1.0; f_surch = 1.0

    def factor_res(res, f):
        out = {}
        if not res: return out
        for eid, dat in res.items():
            new_loads = []
            for l in dat.get('loads', []):
                new_l = copy.deepcopy(l)
                if new_l['type'] == 'point': new_l['params'][0] *= f
                elif new_l['type'] == 'distributed_trapezoid': 
                    new_l['params'][0] *= f
                    new_l['params'][1] *= f
                new_loads.append(new_l)
            out[eid] = {
                **dat, 'loads': new_loads,
                'M': dat['M']*f, 'V': dat['V']*f, 'N': dat['N']*f,
                'M_max': dat['M']*f, 'M_min': dat['M']*f,
                'V_max': dat['V']*f, 'V_min': dat['V']*f,
                'N_max': dat['N']*f, 'N_min': dat['N']*f,
                'def_x': dat['def_x']*f, 'def_x_min': dat['def_x']*f,
                'def_y_max': dat['def_y']*f, 'def_y_min': dat['def_y']*f,
                'def_x_max': dat['def_x']*f, 'def_x_min': dat['def_x']*f,
                'def_y_max': dat['def_y']*f, 'def_y_min': dat['def_y']*f
            }
        return out

    out_sw = factor_res(raw_res['Selfweight'], f_sw)
    out_soil = factor_res(raw_res['Soil'], f_soil)
    out_surch = factor_res(raw_res['Surcharge'], f_surch) 
    
    raw_env_A = raw_res['Vehicle Envelope A']
    raw_env_B = raw_res['Vehicle Envelope B']
    out_veh_env = {}
    
    if raw_env_A:
        for eid in raw_env_A:
            dA = raw_env_A[eid]
            dB = raw_env_B.get(eid, dA) 
            base = dA['base']
            out_veh_env[eid] = {
                **base,
                'M_max': dA['M_max']*f_vehA + dB['M_max']*f_vehB,
                'M_min': dA['M_min']*f_vehA + dB['M_min']*f_vehB,
                'V_max': dA['V_max']*f_vehA + dB['V_max']*f_vehB,
                'V_min': dA['V_min']*f_vehA + dB['V_min']*f_vehB,
                'N_max': dA['N_max']*f_vehA + dB['N_max']*f_vehB,
                'N_min': dA['N_min']*f_vehA + dB['N_min']*f_vehB,
                'def_x_max': dA['def_x_max']*f_vehA + dB['def_x_max']*f_vehB,
                'def_x_min': dA['def_x_min']*f_vehA + dB['def_x_min']*f_vehB,
                'def_y_max': dA['def_y_max']*f_vehA + dB['def_y_max']*f_vehB,
                'def_y_min': dA['def_y_min']*f_vehA + dB['def_y_min']*f_vehB
            }

    out_total = {}
    all_ids = set(out_sw.keys()) | set(out_veh_env.keys()) | set(out_soil.keys()) | set(out_surch.keys())
    combine_surcharge_vehicle = params.get('combine_surcharge_vehicle', False)

    for eid in all_ids:
        ref = out_sw.get(eid) or out_soil.get(eid) or out_veh_env.get(eid) or out_surch.get(eid)
        if not ref: continue
        n_p = len(ref['M_max'])
        z = np.zeros(n_p)
        sw = out_sw.get(eid, {'M_max':z, 'M_min':z, 'V_max':z, 'V_min':z, 'N_max':z, 'N_min':z, 'def_x_max':z, 'def_x_min':z, 'def_y_max':z, 'def_y_min':z})
        sl = out_soil.get(eid, {'M_max':z, 'M_min':z, 'V_max':z, 'V_min':z, 'N_max':z, 'N_min':z, 'def_x_max':z, 'def_x_min':z, 'def_y_max':z, 'def_y_min':z})
        ve = out_veh_env.get(eid, {'M_max':z, 'M_min':z, 'V_max':z, 'V_min':z, 'N_max':z, 'N_min':z, 'def_x_max':z, 'def_x_min':z, 'def_y_max':z, 'def_y_min':z})
        su = out_surch.get(eid, {'M_max':z, 'M_min':z, 'V_max':z, 'V_min':z, 'N_max':z, 'N_min':z, 'def_x_max':z, 'def_x_min':z, 'def_y_max':z, 'def_y_min':z})

        M_perm_max = sw['M_max'] + sl['M_max']
        M_perm_min = sw['M_min'] + sl['M_min']
        V_perm_max = sw['V_max'] + sl['V_max']
        V_perm_min = sw['V_min'] + sl['V_min']
        N_perm_max = sw['N_max'] + sl['N_max']
        N_perm_min = sw['N_min'] + sl['N_min']
        def_x_perm_max = sw['def_x_max'] + sl['def_x_max']
        def_x_perm_min = sw['def_x_min'] + sl['def_x_min']
        def_y_perm_max = sw['def_y_max'] + sl['def_y_max']
        def_y_perm_min = sw['def_y_min'] + sl['def_y_min']

        if combine_surcharge_vehicle:
            M_tot_max = M_perm_max + su['M_max'] + ve['M_max']
            M_tot_min = M_perm_min + su['M_min'] + ve['M_min']
            V_tot_max = V_perm_max + su['V_max'] + ve['V_max']
            V_tot_min = V_perm_min + su['V_min'] + ve['V_min']
            N_tot_max = N_perm_max + su['N_max'] + ve['N_max']
            N_tot_min = N_perm_min + su['N_min'] + ve['N_min']
            def_x_tot_max = def_x_perm_max + su['def_x_max'] + ve['def_x_max']
            def_x_tot_min = def_x_perm_min + su['def_x_min'] + ve['def_x_min']
            def_y_tot_max = def_y_perm_max + su['def_y_max'] + ve['def_y_max']
            def_y_tot_min = def_y_perm_min + su['def_y_min'] + ve['def_y_min']
        else:
            M_tot_max = np.maximum(M_perm_max + ve['M_max'], M_perm_max + su['M_max'])
            M_tot_min = np.minimum(M_perm_min + ve['M_min'], M_perm_min + su['M_min'])
            V_tot_max = np.maximum(V_perm_max + ve['V_max'], V_perm_max + su['V_max'])
            V_tot_min = np.minimum(V_perm_min + ve['V_min'], V_perm_min + su['V_min'])
            N_tot_max = np.maximum(N_perm_max + ve['N_max'], N_perm_max + su['N_max'])
            N_tot_min = np.minimum(N_perm_min + ve['N_min'], N_perm_min + su['N_min'])
            def_x_tot_max = np.maximum(def_x_perm_max + ve['def_x_max'], def_x_perm_max + su['def_x_max'])
            def_x_tot_min = np.minimum(def_x_perm_min + ve['def_x_min'], def_x_perm_min + su['def_x_min'])
            def_y_tot_max = np.maximum(def_y_perm_max + ve['def_y_max'], def_y_perm_max + su['def_y_max'])
            def_y_tot_min = np.minimum(def_y_perm_min + ve['def_y_min'], def_y_perm_min + su['def_y_min'])

        out_total[eid] = {
            **sw, 
            'M_max': M_tot_max, 'M_min': M_tot_min,
            'V_max': V_tot_max, 'V_min': V_tot_min,
            'N_max': N_tot_max, 'N_min': N_tot_min,
            'def_x_max': def_x_tot_max, 'def_x_min': def_x_tot_min,
            'def_y_max': def_y_tot_max, 'def_y_min': def_y_tot_min
        }
    
    return {
        'Selfweight': out_sw, 'Soil': out_soil, 'Surcharge': out_surch,
        'Vehicle Envelope': out_veh_env, 'Total Envelope': out_total,
        'Vehicle Steps A': raw_res.get('Vehicle Steps A', []),
        'Vehicle Steps A_Rev': raw_res.get('Vehicle Steps A_Rev', []),
        'Vehicle Steps B': raw_res.get('Vehicle Steps B', []),
        'Vehicle Steps B_Rev': raw_res.get('Vehicle Steps B_Rev', []),
        'f_vehA': f_vehA, 'f_vehB': f_vehB,
        'phi_calc': phi, 'phi_log': raw_res['phi_log'],
        'Reactions': raw_res.get('Reactions', {})
    }