import streamlit as st
import numpy as np
import copy
import bricos_kernels as kernels  # Importing the renamed kernel module

# ==========================================
# 1. CORE FEM CLASSES & FUNCTIONS
# ==========================================

class FrameElement:
    def __init__(self, node_i, node_j, E, I, A=1e4):
        k_loc, k_glob, T, L, cx, cy = kernels.jit_beam_matrices(
            node_i[0], node_i[1], node_j[0], node_j[1], float(E), float(I), float(A)
        )
        self.k_local = k_loc
        self.k_global = k_glob
        self.T = T
        self.L = L
        self.cx = cx
        self.cy = cy
        self.E = E
        self.I = I

    def get_fixed_end_forces(self, load_type, params):
        if load_type == 'point':
            return kernels.jit_fef_point(params[0], params[1], self.L)
        elif load_type == 'distributed_trapezoid':
            return kernels.jit_fef_trapezoid(params[0], params[1], params[2], params[3], self.L)
        return np.zeros(6)

def build_stiffness_matrix(nodes, elements, restraints_stiffness):
    node_keys = sorted(nodes.keys())
    node_map = {nid: i for i, nid in enumerate(node_keys)}
    NDOF = 3 * len(nodes)
    K_global = np.zeros((NDOF, NDOF))
    elem_objects = []
    
    # 1. Assemble Element Stiffness
    for el_data in elements:
        ni_id, nj_id = el_data['nodes']
        el = FrameElement(nodes[ni_id], nodes[nj_id], el_data['E'], el_data['I'])
        elem_objects.append(el)
        idx_i, idx_j = node_map[ni_id]*3, node_map[nj_id]*3
        indices = [idx_i, idx_i+1, idx_i+2, idx_j, idx_j+1, idx_j+2]
        K_global[np.ix_(indices, indices)] += el.k_global
        
    # 2. Apply Boundary Springs / Restraints
    # restraints_stiffness is a dict: {node_id: [kx, ky, km]}
    for nid, stiff in restraints_stiffness.items():
        if nid in node_map:
            idx = node_map[nid]*3
            for i in range(3):
                # Add spring stiffness to the diagonal (Penalty Method / Direct Stiffness)
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
            f_start += el.get_fixed_end_forces(load['type'], load['params'])
            
        load_arr = np.zeros((len(active_loads), 5))
        for k, load in enumerate(active_loads):
            p = load['params']
            if load['type'] == 'point':
                load_arr[k, :] = [1, p[0], p[1], 0.0, 0.0]
            elif load['type'] == 'distributed_trapezoid':
                load_arr[k, :] = [2, p[0], p[1], p[2], p[3]]

        num_pts = max(5, int(el.L / mesh_size) + 1)
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
            'ni_id': ni, 'nj_id': nj
        }
    return results

def calculate_reactions(nodes, detailed_results):
    reactions = {nid: np.zeros(3) for nid in nodes}
    for eid, data in detailed_results.items():
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
# 2. MAIN SOLVER CONTROLLER (Strategy 5 Integrated)
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
        valid_geom = False
        if params['mode'] == 'Frame':
            h_left_total = params['h_list'][0]
            if h_left_total > 0: valid_geom = True
            soil_L = [s['h'] for s in params.get('soil', []) if s['wall_idx'] == 0]
            h_emb_L = max(soil_L) if soil_L else 0.0
            h_free_L = max(0.0, h_left_total - h_emb_L)
            lengths_for_phi.append(h_free_L)
            comp_desc.append(f"LeftLeg({h_free_L:.2f}m)")
            
            for i in range(params['num_spans']):
                L = params['L_list'][i]
                if L > 0: valid_geom = True
                lengths_for_phi.append(L)
                comp_desc.append(f"Span{i+1}({L:.2f}m)")
            
            end_idx = params['num_spans']
            h_right_total = params['h_list'][end_idx]
            if h_right_total > 0: valid_geom = True
            soil_R = [s['h'] for s in params.get('soil', []) if s['wall_idx'] == end_idx]
            h_emb_R = max(soil_R) if soil_R else 0.0
            h_free_R = max(0.0, h_right_total - h_emb_R)
            lengths_for_phi.append(h_free_R)
            comp_desc.append(f"RightLeg({h_free_R:.2f}m)")
        else:
            for i in range(params['num_spans']):
                L = params['L_list'][i]
                if L > 0: valid_geom = True
                lengths_for_phi.append(L)
                comp_desc.append(f"Span{i+1}({L:.2f}m)")

        if valid_geom:
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

            raw_val = 1.25 - (L_phi - 5.0) / 225.0
            if L_phi <= 5.0: calc_phi = 1.25 
            else: calc_phi = max(1.0, raw_val) 
            phi_log.append(f"Phi = 1.25 - ({L_phi:.3f}-5)/225 = {raw_val:.3f}")
            phi_log.append(f"Final Phi = {calc_phi:.3f}")
        else:
            phi_log.append("Geometry invalid/empty. Phi=1.0")

    phi = phi_val_override if phi_val_override is not None else calc_phi

    # --- SETUP FEM ---
    nodes = {}
    elems_base = []
    restraints = {}
    
    num_spans = params['num_spans']
    num_supp = num_spans + 1
    curr_x = 0.0
    
    # Retrieve Supports Config
    supports_cfg = params.get('supports', [])
    
    # --- PREPARE E-MODULUS LISTS (Backward Compat) ---
    E_default = params.get('E', 30e6)
    E_spans = params.get('E_span_list', [E_default] * num_spans)
    E_walls = params.get('E_wall_list', [E_default] * num_supp)
    
    if params['mode'] == 'Frame':
        for i in range(num_supp):
            h = params['h_list'][i]
            nid_b, nid_t = 100+i, 200+i
            nodes[nid_b] = (curr_x, -h)
            nodes[nid_t] = (curr_x, 0.0)
            
            # Variable Stiffness (Bottom Node)
            if i < len(supports_cfg): k_vec = supports_cfg[i]['k']
            else: k_vec = [1e14, 1e14, 1e14] 
            
            restraints[nid_b] = k_vec
            
            # Element Prop
            e_val = E_walls[i] if i < len(E_walls) else E_default
            
            elems_base.append({
                'id': f'W{i+1}', 'nodes': (nid_b, nid_t),
                'E': e_val, 'I': params['Iw_list'][i]
            })
            if i < num_spans: curr_x += params['L_list'][i]
            
    else: # Superstructure
        for i in range(num_supp):
            nid_t = 200+i
            nodes[nid_t] = (curr_x, 0.0)
            
            if i < len(supports_cfg): k_vec = supports_cfg[i]['k']
            else:
                if i == 0: k_vec = [1e14, 1e14, 0.0]
                else: k_vec = [0.0, 1e14, 0.0]
                
            restraints[nid_t] = k_vec
            if i < num_spans: curr_x += params['L_list'][i]

    for i in range(num_spans):
        nid_s, nid_e = 200+i, 200+i+1
        # Element Prop
        e_val = E_spans[i] if i < len(E_spans) else E_default
        
        elems_base.append({
            'id': f'S{i+1}', 'nodes': (nid_s, nid_e),
            'E': e_val, 'I': params['Is_list'][i]
        })

    K_glob, node_map, elem_objects, NDOF = build_stiffness_matrix(nodes, elems_base, restraints)
    
    # --- SOLVER STABILITY CHECK ---
    try:
        K_inv = np.linalg.inv(K_glob)
    except np.linalg.LinAlgError:
        raise ValueError("Structural Instability Detected: The model is insufficiently constrained (Mechanism). Please check boundary conditions.")

    def solve_static(loads_dict):
        F = np.zeros(NDOF)
        for idx, load_list in loads_dict.items():
            el = elem_objects[idx]
            for load in load_list:
                f_loc = el.get_fixed_end_forces(load['type'], load['params'])
                f_glob = el.T.T @ f_loc
                ni, nj = elems_base[idx]['nodes']
                idx_i, idx_j = node_map[ni]*3, node_map[nj]*3
                indices = [idx_i, idx_i+1, idx_i+2, idx_j, idx_j+1, idx_j+2]
                F[indices] -= f_glob
        D = K_inv @ F
        return get_detailed_results_optimized(elem_objects, elems_base, nodes, D, loads_dict, params.get('mesh_size', 0.5))

    sw_loads_map = {}
    for i in range(num_spans):
        if params['sw_list'][i] != 0:
            idx = i + (num_supp if params['mode'] == 'Frame' else 0)
            val = params['sw_list'][i]
            sw_loads_map[idx] = [{'type': 'distributed_trapezoid', 'params': [val, val, 0, params['L_list'][i]]}]
    res_sw = solve_static(sw_loads_map)

    soil_loads_map = {}
    if params['mode'] == 'Frame':
        for s in params.get('soil', []):
            if s['wall_idx'] < num_supp:
                sign = 1.0 if s['face'] == 'L' else -1.0 
                if s['wall_idx'] not in soil_loads_map: soil_loads_map[s['wall_idx']] = []
                soil_loads_map[s['wall_idx']].append({
                    'type': 'distributed_trapezoid', 'params': [sign*s['q_bot'], sign*s['q_top'], 0, s['h']]
                })
    res_soil = solve_static(soil_loads_map)

    surch_loads_map = {}
    if params['mode'] == 'Frame':
        for sur in params.get('surcharge', []):
            if sur['wall_idx'] < num_supp:
                sign = 1.0 if sur['face'] == 'L' else -1.0
                if sur['wall_idx'] not in surch_loads_map: surch_loads_map[sur['wall_idx']] = []
                surch_loads_map[sur['wall_idx']].append({
                    'type': 'distributed_trapezoid', 'params': [sign*sur['q'], sign*sur['q'], 0, sur['h']]
                })
    res_surch = solve_static(surch_loads_map)

    def get_empty_env():
        env = {}
        res_empty = solve_static({})
        if res_empty:
            for eid, data in res_empty.items():
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
    
    # --- STRATEGY 5 IMPLEMENTATION ---
    def run_stepping(vehicle_key, env_to_fill):
        v_loads = np.array(params[vehicle_key]['loads']) * 9.81
        v_steps_res = [] 
        
        if len(v_loads) > 0 and env_to_fill:
            dists = np.cumsum(params[vehicle_key]['spacing'])
            total_len = sum(params['L_list'][:num_spans])
            step_val = params.get('step_size', 0.2)
            x_steps = np.arange(-max(dists)-1.0, total_len + max(dists) + 1.0, step_val)
            total_steps = len(x_steps)
            
            # --- PREPARE DATA FOR BATCH KERNELS ---
            sp_start_idx = num_supp if params['mode'] == 'Frame' else 0
            sp_start_x = np.zeros(num_spans)
            sp_lens = np.zeros(num_spans)
            sp_el_indices = np.zeros(num_spans, dtype=np.int32)
            c_x = 0
            sp_elems_info = [] 
            for sp_i in range(num_spans):
                L = params['L_list'][sp_i]
                el_idx = sp_start_idx + sp_i
                sp_start_x[sp_i] = c_x
                sp_lens[sp_i] = L
                sp_el_indices[sp_i] = el_idx
                sp_elems_info.append((c_x, L, el_idx))
                c_x += L

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
            max_len = np.max(el_L)
            n_pts_kernel = max(5, int(max_len / mesh_sz) + 1)
            S_matrices = kernels.jit_precompute_stress_recovery(n_elems, n_pts_kernel, el_L, el_k_local)

            env_results_accum = np.zeros((n_elems, n_pts_kernel, 10))
            CHUNK_SIZE = 2000
            
            for start_idx in range(0, total_steps, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, total_steps)
                x_chunk = x_steps[start_idx:end_idx]
                n_chunk = len(x_chunk)
                is_init = (start_idx == 0)
                
                F_chunk = kernels.jit_build_batch_F(NDOF, n_chunk, x_chunk, v_loads, dists, sp_start_x, sp_lens, sp_el_indices, el_L, el_T, el_dof_indices)
                D_chunk = K_inv @ F_chunk
                
                kernels.jit_envelope_batch_parallel(
                    n_chunk, n_elems, n_pts_kernel,
                    x_chunk, v_loads, dists,
                    sp_start_x, sp_lens, sp_el_indices,
                    D_chunk, el_dof_indices, el_T, el_L,
                    S_matrices,
                    env_results_accum, 
                    is_init
                )
                
                for i_local in range(n_chunk):
                    global_i = start_idx + i_local
                    #if global_i % 5 != 0: continue 
                    x_front = x_chunk[i_local]
                    D_step = D_chunk[:, i_local]
                    step_loads_map = {}
                    has_loads = False
                    for ax_i, d in enumerate(dists):
                        ax_x = x_front - d
                        for (start_x, L_span, el_idx) in sp_elems_info:
                            local_x = ax_x - start_x
                            if 0 <= local_x <= L_span:
                                P_val = v_loads[ax_i]
                                if el_idx not in step_loads_map: step_loads_map[el_idx] = []
                                load_p = {'type': 'point', 'params': [P_val, local_x]}
                                step_loads_map[el_idx].append(load_p)
                                has_loads = True
                                break
                    if has_loads:
                        step_res = get_detailed_results_optimized(elem_objects, elems_base, nodes, D_step, step_loads_map, params.get('mesh_size', 0.5))
                        v_steps_res.append({'x': x_front, 'res': step_res})

            for k in range(n_elems):
                eid = elems_base[k]['id']
                t = env_to_fill[eid]
                real_L = el_L[k]
                target_x = t['base']['x']
                kernel_x = np.linspace(0, real_L, n_pts_kernel)
                valid_res = env_results_accum[k, :n_pts_kernel, :]
                def interp_res(idx): return np.interp(target_x, kernel_x, valid_res[:, idx])

                t['M_max'] = np.maximum(t['M_max'], interp_res(0))
                t['M_min'] = np.minimum(t['M_min'], interp_res(1))
                t['V_max'] = np.maximum(t['V_max'], interp_res(2))
                t['V_min'] = np.minimum(t['V_min'], interp_res(3))
                t['N_max'] = np.maximum(t['N_max'], interp_res(4))
                t['N_min'] = np.minimum(t['N_min'], interp_res(5))
                t['def_x_max'] = np.maximum(t['def_x_max'], interp_res(6))
                t['def_x_min'] = np.minimum(t['def_x_min'], interp_res(7))
                t['def_y_max'] = np.maximum(t['def_y_max'], interp_res(8))
                t['def_y_min'] = np.minimum(t['def_y_min'], interp_res(9))

        return v_steps_res

    steps_A = run_stepping('vehicle', veh_env_A)
    steps_B = run_stepping('vehicleB', veh_env_B)

    return {
        'Selfweight': res_sw,
        'Soil': res_soil,
        'Surcharge': res_surch,
        'Vehicle Envelope A': veh_env_A,
        'Vehicle Envelope B': veh_env_B,
        'Vehicle Steps A': steps_A,
        'Vehicle Steps B': steps_B,
        'phi_calc': calc_phi,
        'phi_log': phi_log,
        'Reactions': calculate_reactions(nodes, res_sw) 
    }, nodes, 0

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
                'def_x': dat['def_x']*f, 'def_y': dat['def_y']*f,
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
        n_p = 50
        if eid in out_sw: n_p = len(out_sw[eid]['x'])
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
        'Vehicle Steps B': raw_res.get('Vehicle Steps B', []),
        'f_vehA': f_vehA, 'f_vehB': f_vehB,
        'phi_calc': phi, 'phi_log': raw_res['phi_log'],
        'Reactions': raw_res.get('Reactions', {})
    }