import hashlib
import itertools
import json

import streamlit as st
import numpy as np
import bricos_kernels as kernels
import bricos_data as data_mod

# Process-unique tokens identifying raw analysis results; see the
# combine_results memo.
_combine_token_counter = itertools.count(1)

# ==========================================
# 1. CORE FEM CLASSES & FUNCTIONS
# ==========================================

class FrameElement:
    def __init__(self, node_i, node_j, E, geom_data, shear_config):
        # geom_data convention: {'type': 0(I)/1(H), 'shape': 0(Const)/1(Lin)/2(3pt),
        # 'vals': [start, mid, end]}. Linear profiles use start/end and reserve
        # vals[1] for the mid point so the UI, parent geometry, and kernels agree.
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
        
        # Effective [start, mid, end] values for this sub-element. Keep this
        # convention aligned with bricos_kernels.get_section_value_at_x. Even
        # when a parent 3-point kink falls inside a sub-element, sampling the
        # sub-element midpoint prevents a symmetric [start == end, mid higher]
        # profile from being mistaken for a constant section.
        v_start = interp_val(rel_start)
        v_mid = interp_val(rel_mid)
        v_end = interp_val(rel_end)

        # --- FIX: DETECT "FAKE" TAPER ---
        # Treat as prismatic only when start, mid, and end all match.
        is_effectively_constant = (
            abs(v_start - v_mid) < 1e-6 and
            abs(v_mid - v_end) < 1e-6
        )

        eff_vals = [v_start, v_mid, v_end]
        # Use a three-point sub-element representation for all genuine variable
        # sections. Linear parents have a collinear midpoint, while 3-point
        # parents retain the local kink shape.
        eff_shape = 2 if geom_data.get('shape', 0) != 0 else 0

        if is_effectively_constant:
            eff_shape = 0
            eff_vals = [v_start, v_start, v_start]

        # Defaults
        b_eff = shear_config.get('b_eff', 1.0)
        if b_eff < 0.01: b_eff = 1.0
        
        if v_type == 1:
            # Input was H (Height).
            h_avg = (v_start + 4.0 * v_mid + v_end) / 6.0
            I_avg = (b_eff * h_avg**3) / 12.0
            A_approx = b_eff * h_avg
        else:
            # Input was I.
            I_avg = (v_start + 4.0 * v_mid + v_end) / 6.0
            h_est = (12.0 * I_avg / b_eff)**(1.0/3.0)
            A_approx = b_eff * h_est

        # --- SAFETY FLOOR ---
        MIN_I = 1e-9  # m^4
        MIN_A = 1e-6  # m^2
        if I_avg < MIN_I: I_avg = MIN_I
        if A_approx < MIN_A: A_approx = MIN_A
        if v_start < 1e-6: v_start = 1e-6
        if v_end < 1e-6: v_end = 1e-6

        # Gross area at the sub-element ends (same section convention as
        # A_approx). The auto self-weight reads these so a tapered member is
        # loaded with a true trapezoid - resultant at the section centroid -
        # rather than a midpoint-lumped uniform load, keeping reactions and
        # M/V diagrams mesh-independent for linear tapers.
        if v_type == 1:
            A_start = b_eff * v_start
            A_end = b_eff * v_end
        else:
            A_start = b_eff * (12.0 * v_start / b_eff) ** (1.0 / 3.0)
            A_end = b_eff * (12.0 * v_end / b_eff) ** (1.0 / 3.0)
        if A_start < MIN_A: A_start = MIN_A
        if A_end < MIN_A: A_end = MIN_A

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
        self.eff_vals = np.array(eff_vals, dtype=np.float64)
        self.b_eff = float(b_eff)
        self.As_avg = float(As_avg)
        # Axial (gross) area b_eff x h, on the same interpolated section the
        # stiffness uses. The auto self-weight load reads this so the
        # computed weight and the stiffness model never disagree.
        self.A_avg = float(A_approx)
        # End areas for the trapezoidal auto self-weight (see above).
        self.A_start = float(A_start)
        self.A_end = float(A_end)
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
                self.eff_shape, v_type, self.eff_vals, float(A_approx), float(As_avg),
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
        if load_type == 'axial_point':
            return kernels.jit_fef_axial_point(params[0], params[1], self.L)
        elif load_type == 'axial_distributed_trapezoid':
            return kernels.jit_fef_axial_trapezoid(params[0], params[1], params[2], params[3], self.L)

        # Hermite-consistent transverse equivalent nodal loads are independent
        # of EI(x), so use the same analytical kernels for both prismatic and
        # non-prismatic elements.
        if load_type == 'point':
            return kernels.jit_fef_point(params[0], params[1], self.L)
        elif load_type == 'distributed_trapezoid':
            return kernels.jit_fef_trapezoid(params[0], params[1], params[2], params[3], self.L)

        return np.zeros(6)

def decompose_gravity_params(load_type, params, cx, cy):
    """Return local transverse and axial load records for a global vertical gravity load."""
    if load_type == 'point':
        return (
            ('point', [params[0] * cx, params[1]]),
            ('axial_point', [params[0] * cy, params[1]]),
        )
    if load_type == 'distributed_trapezoid':
        return (
            ('distributed_trapezoid', [params[0] * cx, params[1] * cx, params[2], params[3]]),
            ('axial_distributed_trapezoid', [params[0] * cy, params[1] * cy, params[2], params[3]]),
        )
    return ((load_type, list(params)),)

def get_local_fixed_end_forces_for_load(el, load):
    if load.get('is_gravity', False):
        f_loc = np.zeros(6)
        for local_type, local_params in decompose_gravity_params(load['type'], load['params'], el.cx, el.cy):
            f_loc += el.get_fixed_end_forces(local_type, local_params)
        return f_loc
    return el.get_fixed_end_forces(load['type'], load['params'])

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

def get_detailed_results_optimized(elem_objects, elements_source_data, nodes, D_total, loads_map, mesh_size=0.5, node_map=None):
    # node_map can be passed in by callers that invoke this once per vehicle
    # step; rebuilding it (sort + dict over all nodes) dominated the
    # per-step cost on profiling.
    if node_map is None:
        node_keys = sorted(nodes.keys())
        node_map = {nid: i for i, nid in enumerate(node_keys)}
    results = {}
    
    for i, el_data in enumerate(elements_source_data):
        el = elem_objects[i]
        ni, nj = el_data['nodes']
        idx_i, idx_j = node_map[ni]*3, node_map[nj]*3
        d_glob = np.concatenate([D_total[idx_i:idx_i+3], D_total[idx_j:idx_j+3]])
        d_loc = el.T @ d_glob
        
        f_elem_local = el.k_local @ d_loc
        active_loads = loads_map.get(i, [])
        
        for load in active_loads:
            f_elem_local += get_local_fixed_end_forces_for_load(el, load)

        f_start_for_internal = f_elem_local[:3]

        load_records = []
        for load in active_loads:
            if load.get('is_gravity', False):
                for local_type, local_params in decompose_gravity_params(load['type'], load['params'], el.cx, el.cy):
                    if local_type == 'point':
                        load_records.append([1, local_params[0], local_params[1], 0.0, 0.0])
                    elif local_type == 'distributed_trapezoid':
                        load_records.append([2, local_params[0], local_params[1], local_params[2], local_params[3]])
                    elif local_type == 'axial_point':
                        load_records.append([3, local_params[0], local_params[1], 0.0, 0.0])
                    elif local_type == 'axial_distributed_trapezoid':
                        load_records.append([4, local_params[0], local_params[1], local_params[2], local_params[3]])
            else:
                p = load['params']
                if load['type'] == 'point':
                    load_records.append([1, p[0], p[1], 0.0, 0.0])
                elif load['type'] == 'distributed_trapezoid':
                    load_records.append([2, p[0], p[1], p[2], p[3]])
                elif load['type'] == 'axial_point':
                    load_records.append([3, p[0], p[1], 0.0, 0.0])
                elif load['type'] == 'axial_distributed_trapezoid':
                    load_records.append([4, p[0], p[1], p[2], p[3]])

        load_arr = np.zeros((len(load_records), 5))
        for k, rec in enumerate(load_records):
            load_arr[k, :] = rec

        num_pts = max(3, int(el.L / mesh_size) + 1)
        # Parent-interior sub-element starts carry the loaded-side V/N of a
        # point load sitting exactly on them (see jit_internal_forces).
        start_jump = 1.0 if el_data.get('local_offset', 0.0) > 1e-9 else 0.0
        x_vals, M_vals, V_vals, N_vals = kernels.jit_internal_forces(
            el.L, f_start_for_internal, num_pts, load_arr, start_jump
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
            'f_start_local': f_elem_local[:3].copy(),
            'f_end_local': f_elem_local[3:6].copy(),
            'ni_id': ni, 'nj_id': nj,
            'parent': el_data.get('parent'),
            'local_offset': el_data.get('local_offset', 0.0)
        }
    return results

def aggregate_member_results(detailed_results, global_loads_override=None):
    agg_res = {}
    grouped = {}
    
    # 1. Group by Parent
    for eid, res in detailed_results.items():
        parent = res.get('parent', eid)
        if parent not in grouped: grouped[parent] = []
        grouped[parent].append(res)
    
    # 2. Optimized Aggregation
    for parent, parts in grouped.items():
        parts.sort(key=lambda x: x['local_offset'])
        
        # Calculate Total Points upfront for Pre-allocation
        total_pts = sum(len(p['x']) for p in parts)
        
        # Pre-allocate Arrays
        x_agg = np.zeros(total_pts)
        M_agg = np.zeros(total_pts)
        V_agg = np.zeros(total_pts)
        N_agg = np.zeros(total_pts)
        dx_agg = np.zeros(total_pts)
        dy_agg = np.zeros(total_pts)
        
        raw_loads_all = []
        
        base = parts[0]
        last = parts[-1]
        total_L = sum(p['L'] for p in parts)
        
        current_idx = 0
        
        for p in parts:
            n_p = len(p['x'])
            offset = p['local_offset']
            
            # Direct Slice Filling (Faster than append + concat)
            x_agg[current_idx : current_idx + n_p] = p['x'] + offset
            M_agg[current_idx : current_idx + n_p] = p['M']
            V_agg[current_idx : current_idx + n_p] = p['V']
            N_agg[current_idx : current_idx + n_p] = p['N']
            dx_agg[current_idx : current_idx + n_p] = p['def_x']
            dy_agg[current_idx : current_idx + n_p] = p['def_y']
            
            current_idx += n_p
            
            # Load Aggregation (Lists are unavoidable here, but less critical)
            if not global_loads_override or parent not in global_loads_override:
                for load in p['loads']:
                    # Loads are flat dicts with a scalar params list; copying
                    # them directly is far cheaper than deepcopy in this hot
                    # path (called once per vehicle step).
                    new_load = {**load, 'params': list(load['params'])}
                    params = new_load['params']
                    if load['type'] == 'point':
                        params[1] += offset
                    elif load['type'] == 'distributed_trapezoid':
                        params[2] += offset
                        params[3] += offset
                    raw_loads_all.append(new_load)
        
        # Load Deduplication / Override Logic
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
            'x': x_agg,
            'M': M_agg,
            'V': V_agg,
            'N': N_agg,
            'def_x': dx_agg,
            'def_y': dy_agg,
            'L': total_L,
            'cx': base['cx'], 'cy': base['cy'],
            'ni': base['ni'], 'nj': last['nj'],
            'ni_id': base['ni_id'], 'nj_id': last['nj_id'],
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
        'Dead Load': empty_dict, 'Self-weight': empty_dict,
        'Soil': empty_dict, 'Surcharge': empty_dict,
        'Vehicle Envelope A': empty_dict, 'Vehicle Envelope B': empty_dict,
        'Vehicle Steps A': [], 'Vehicle Steps B': [],
        'phi_calc': 1.0, 'phi_log': ["System Unstable or Empty"],
        'Phi Members': {},
        'Traffic UDL': {}, 'udl_line_load': 0.0,
        'Equilibrium': {},
        'Restrained Nodes': []
    }

def split_distributed_trapezoid_for_sub_element(params_list, loc_start, loc_end, tol=1e-9):
    q_s_glob, q_e_glob, x_s_glob, L_load = params_list
    x_e_glob = x_s_glob + L_load
    overlap_start = max(loc_start, x_s_glob)
    overlap_end = min(loc_end, x_e_glob)

    if overlap_end <= overlap_start + tol:
        return None

    len_glob = x_e_glob - x_s_glob
    if len_glob > tol:
        q_sub_start = q_s_glob + (q_e_glob - q_s_glob) * (overlap_start - x_s_glob) / len_glob
        q_sub_end = q_s_glob + (q_e_glob - q_s_glob) * (overlap_end - x_s_glob) / len_glob
    else:
        q_sub_start = q_s_glob
        q_sub_end = q_e_glob

    local_x_s = overlap_start - loc_start
    local_x_e = overlap_end - loc_start
    return [q_sub_start, q_sub_end, local_x_s, local_x_e]

def point_load_belongs_to_sub_element(x_pos_glob, loc_start, loc_end, is_last, tol=1e-9):
    return (loc_start - tol <= x_pos_glob < loc_end - tol) or (is_last and abs(x_pos_glob - loc_end) <= tol)

# ==========================================
# 2. MAIN SOLVER CONTROLLER
# ==========================================

# Keys that never influence the raw analysis results. Cosmetic/UI state
# (name, plot scale, text projections) and combination-stage factors
# (gammas, KFI) are stripped from the cache key so that editing them does
# not trigger a full re-analysis. Combination factors are applied later in
# combine_results, which is not cached.
NON_SOLVER_PARAM_KEYS = frozenset({
    'name', 'scale_manual', 'last_mode', 'backup',
    '_vehicle_text_errors',
    'vehicle_loads', 'vehicle_space', 'vehicleB_loads', 'vehicleB_space',
    'KFI', 'gamma_g', 'gamma_j', 'gamma_veh', 'gamma_vehB',
    'combine_surcharge_vehicle',
    # SLS phi treatment is applied in combine_results, not in the raw analysis.
    'phi_sls_mode', 'phi_sls',
    # Traffic UDL partial factor is combination-stage only; the UDL
    # definition (udl_q, udl_gap, udl_mode) stays in the key.
    'gamma_udl',
    # The SLS factor set and the ULS/SLS analysis toggles never affect the
    # raw analysis.
    'sls_g', 'sls_j', 'sls_veh', 'sls_vehB', 'sls_udl',
    'analyze_uls', 'analyze_sls',
})


def solver_cache_params(params):
    """Return only the parameters that influence run_raw_analysis output."""
    return {k: v for k, v in params.items() if k not in NON_SOLVER_PARAM_KEYS}


# Feature flag for the batched per-step recovery kernel. Kept switchable so
# the legacy Python recovery path stays available for equivalence testing,
# and as an automatic fallback if per-element point counts ever differ from
# the uniform kernel grid.
USE_BATCH_STEP_RECOVERY = True

# Limits for solving all load cases (static + every vehicle step) against a
# single factorization of K. Beyond these the RHS block is solved per chunk
# instead, trading extra factorizations for memory.
MAX_SINGLE_SOLVE_RHS = 6000
MAX_SINGLE_SOLVE_ELEMENTS = 12_000_000


def phi_from_length(L_inf):
    """Stoedfaktor curve per DS/EN 1991-2 DK NA:2017, Anneks A, A.2.3.5(2)."""
    if L_inf <= 5.0:
        return 1.25
    if L_inf >= 50.0:
        return 1.05
    return 1.25 - (L_inf - 5.0) / 225.0


# Session-state result cache. st.cache_data pickled the full result on every
# write and unpickled it again on EVERY rerun (~0.1 s per interaction for
# nothing); storing the result objects directly avoids the round-trip.
# Contract: cached results are shared objects and must be treated as
# read-only by all consumers (combine_results and the UI already copy
# before scaling).
_SOLVER_CACHE_KEY = '_bricos_solver_result_cache'
_SOLVER_CACHE_MAX_ENTRIES = 4
# Stamp the session caches with the app version. The result-cache key is
# only the params hash, so across a code update under a LIVE session (e.g.
# a Streamlit hot reload, or a redeployed long-lived server) an unchanged
# model would otherwise be served a raw result cached by the previous
# version - with the previous result schema, such as a renamed load case -
# and the new consumers would KeyError. Resetting both caches on a version
# change avoids that for this rename and any future schema change. (Frozen
# EXE users get a fresh process per launch, so they never see the stale
# cache; this guards dev hot reload and server redeploys.)
_CACHE_VERSION_KEY = '_bricos_cache_version'


def _ensure_cache_version():
    try:
        if st.session_state.get(_CACHE_VERSION_KEY) != data_mod.APP_VERSION:
            st.session_state[_SOLVER_CACHE_KEY] = {}
            st.session_state[_COMBINE_MEMO_KEY] = {}
            st.session_state[_CACHE_VERSION_KEY] = data_mod.APP_VERSION
    except Exception:
        pass


def _solver_cache_hash(cache_params, phi_val_override):
    payload = json.dumps(cache_params, sort_keys=True, default=str)
    return (hashlib.md5(payload.encode('utf-8')).hexdigest(), phi_val_override)


def clear_solver_cache():
    """Drop all cached raw-analysis results and combined results (used by
    tests and migrations)."""
    try:
        st.session_state.pop(_SOLVER_CACHE_KEY, None)
        st.session_state.pop(_COMBINE_MEMO_KEY, None)
    except Exception:
        pass


def run_raw_analysis(params, phi_val_override=None):
    cache_params = solver_cache_params(params)
    key = _solver_cache_hash(cache_params, phi_val_override)
    _ensure_cache_version()
    cache = st.session_state.setdefault(_SOLVER_CACHE_KEY, {})
    if key in cache:
        return cache[key]
    result = _run_raw_analysis_cached(cache_params, phi_val_override)
    while len(cache) >= _SOLVER_CACHE_MAX_ENTRIES:
        cache.pop(next(iter(cache)))
    cache[key] = result
    return result


def _run_raw_analysis_cached(params, phi_val_override=None):
    phi_mode = params.get('phi_mode', 'Calculate')
    # L_inf methodology for the calculated stoedfaktor:
    # 'Determinant': one structure-wide influence length per EN 1991-2:2003
    #   Table 6.2 Case 5.1/5.2/5.3 (frame treated as equivalent continuous
    #   beam). Historical BriCoS behaviour.
    # 'Span': the DK NA A.2.3.5(2) simplification L_inf = actual span,
    #   applied per member; walls take the max of the adjacent spans' phi
    #   (the NA is silent for substructure - conservative assumption).
    linf_mode = params.get('phi_linf_mode', 'Determinant')

    # --- PHI CALCULATION ---
    phi_log = []
    calc_phi = 1.0
    # eid -> phi. Empty means uniform calc_phi applies to all members.
    phi_members = {}

    if phi_mode == "Manual":
        if params.get('phi_manual_scope') == 'Per span':
            phi_log.append("Methodology: manual input per span; walls take the "
                           "max of the adjacent spans' Phi")
            phi_spans = params.get('phi_span_list', [])
            span_phis = []
            for i in range(params['num_spans']):
                p_i = float(phi_spans[i]) if i < len(phi_spans) else params.get('phi', 1.0)
                phi_members[f"S{i+1}"] = p_i
                span_phis.append(p_i)
                phi_log.append(f"Span{i+1}: manual Phi = {p_i:.3f}")
            if params['mode'] == 'Frame' and span_phis:
                for i in range(params['num_spans'] + 1):
                    if i == 0:
                        p_w = span_phis[0]
                    elif i == params['num_spans']:
                        p_w = span_phis[-1]
                    else:
                        p_w = max(span_phis[i-1], span_phis[i])
                    phi_members[f"W{i+1}"] = p_w
            calc_phi = max(span_phis) if span_phis else params.get('phi', 1.0)
            phi_log.append(f"Governing (max) Phi = {calc_phi:.3f}")
        else:
            calc_phi = params.get('phi', 1.0)
            phi_log.append(f"Manual input: {calc_phi}")
    elif linf_mode == 'Span':
        phi_log.append("Methodology: L_inf = actual span, per member "
                       "(DS/EN 1991-2 DK NA:2017, Anneks A, A.2.3.5(2))")
        span_phis = []
        for i in range(params['num_spans']):
            L = params['L_list'][i]
            p_i = phi_from_length(L)
            phi_members[f"S{i+1}"] = p_i
            span_phis.append(p_i)
            phi_log.append(f"Span{i+1}: L_inf = {L:.2f} m -> Phi = {p_i:.3f}")
        if params['mode'] == 'Frame' and span_phis:
            for i in range(params['num_spans'] + 1):
                if i == 0:
                    p_w = span_phis[0]
                elif i == params['num_spans']:
                    p_w = span_phis[-1]
                else:
                    p_w = max(span_phis[i-1], span_phis[i])
                phi_members[f"W{i+1}"] = p_w
            phi_log.append("Walls: Phi = max of adjacent spans "
                           "(NA gives no rule for substructure; conservative assumption)")
        if span_phis:
            calc_phi = max(span_phis)
            if params.get('phi_application', 'Per member') == 'Governing':
                # Collapse to a single conservative value for all members.
                phi_members = {}
                phi_log.append(f"Application: governing (max) Phi = {calc_phi:.3f} applied to ALL members")
            else:
                phi_log.append(f"Application: per member; governing (max) Phi = {calc_phi:.3f}")
        else:
            phi_log.append("Geometry invalid/empty. Phi=1.0")
    else:
        phi_log.append("Methodology: determinant length per DS/EN 1991-2:2003, "
                       "Table 6.2 (renumbered Table 8.2 in the 2023 edition), "
                       "used as L_inf in DK NA:2017 Anneks A, A.2.3.5(2)")
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

            calc_phi = phi_from_length(L_phi)
            if L_phi <= 5.0:
                phi_log.append(f"L_phi <= 5.0m: Phi set to upper limit (1.25)")
            elif L_phi >= 50.0:
                phi_log.append(f"L_phi >= 50.0m: Phi set to lower limit (1.05)")
            else:
                phi_log.append(f"5.0 < L_phi < 50.0: Calc Formula applied.")
                phi_log.append(f"Phi = 1.25 - ({L_phi:.3f}-5)/225 = {calc_phi:.3f}")
            phi_log.append(f"Final Phi = {calc_phi:.3f}")
        else:
            phi_log.append("Geometry invalid/empty. Phi=1.0")

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

    def create_member_mesh(start_xy, end_xy, start_node_id, end_node_id, props):
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
                {'parent': f'W{i+1}', 'E': e_val, 'geom': g_data}
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
            {'parent': f'S{i+1}', 'E': e_val, 'geom': g_data}
        )
        elems_base.extend(span_subs)
        model_props['Spans'][f'S{i+1}'] = {'E': e_val}

    if len(elems_base) == 0:
        # Return None for nodes and props to signal invalid geometry to UI/Report
        return get_safe_error_result(), None, None, "No valid structural elements defined."

    shear_config = {
        'use': params.get('use_shear_def', False),
        'b_eff': params.get('b_eff', 1.0),
        'nu': params.get('nu', 0.2)
    }

    K_glob, node_map, elem_objects, NDOF = build_stiffness_matrix(nodes, elems_base, restraints, shear_config)
    
    # --- REPLACED EXPLICIT INVERSION WITH DIRECT SOLVER ---
    # K_inv calculation removed for speed/stability
    # We now check for singularity by attempting to solve dummy or catching errors later
    
    # Basic stability check using condition number is too expensive for real-time.
    # We rely on the solver throwing LinAlgError if singular.

    def assemble_F(loads_dict):
        F = np.zeros(NDOF)
        for idx, load_list in loads_dict.items():
            el = elem_objects[idx]
            for load in load_list:
                f_loc = get_local_fixed_end_forces_for_load(el, load)
                f_glob = el.T.T @ f_loc
                ni, nj = elems_base[idx]['nodes']
                idx_i, idx_j = node_map[ni]*3, node_map[nj]*3
                indices = [idx_i, idx_i+1, idx_i+2, idx_j, idx_j+1, idx_j+2]
                F[indices] -= f_glob
        return F

    def recover_results(loads_dict, D):
        return get_detailed_results_optimized(elem_objects, elems_base, nodes, D, loads_dict, params.get('mesh_size', 0.5), node_map)

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
                sub_params = split_distributed_trapezoid_for_sub_element(params_list, loc_start, loc_end, 1e-6)
                if sub_params is not None:
                    if idx not in target_map: target_map[idx] = []
                    target_map[idx].append({
                        'type': 'distributed_trapezoid',
                        'is_gravity': is_gravity,
                        'params': sub_params
                    })
                    
            elif load_type == 'point':
                P_val, x_pos_glob = params_list
                is_last = idx == indices[-1]
                if point_load_belongs_to_sub_element(x_pos_glob, loc_start, loc_end, is_last, 1e-9):
                    local_x = min(max(x_pos_glob - loc_start, 0.0), el_obj.L)
                    if idx not in target_map: target_map[idx] = []
                    target_map[idx].append({
                        'type': 'point',
                        'is_gravity': is_gravity,
                        'params': [P_val, local_x]
                    })

    # 1. Dead Load
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

    # 1b. Self-weight (auto). When enabled, the structural weight of every
    # member (decks AND walls) is computed as density x A per unit length,
    # using each sub-element's own interpolated gross area b_eff x h - the
    # SAME section the stiffness uses, so weight and stiffness can never
    # disagree. The load is a true trapezoid from the start-area to the
    # end-area, so a tapered member's resultant sits at the section centroid
    # (mesh-independent reactions/M/V for linear tapers); 3-point profiles
    # converge with mesh refinement. Vertical walls resolve gravity to pure
    # axial via the is_gravity path.
    sw_auto_loads_map = {}
    sw_auto_global_loads = {}
    auto_sw = bool(params.get('auto_selfweight'))
    try:
        density = float(params.get('density', 25.0))
    except (TypeError, ValueError):
        density = 25.0
    if auto_sw and density != 0.0:
        for idx, el_data in enumerate(elems_base):
            w_s = density * elem_objects[idx].A_start
            w_e = density * elem_objects[idx].A_end
            if w_s == 0.0 and w_e == 0.0:
                continue
            el_L = elem_objects[idx].L
            entry = {'type': 'distributed_trapezoid', 'is_gravity': True,
                     'params': [w_s, w_e, 0, el_L]}
            sw_auto_loads_map.setdefault(idx, []).append(entry)
            # Per-sub-element entries in the parent's equilibrium list make
            # the applied total exact for variable sections (a single
            # parent trapezoid would miss a 3-point mid bulge).
            sw_auto_global_loads.setdefault(el_data['parent'], []).append(dict(entry))

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

    # Static load case RHS. Solved further below together with all vehicle
    # step RHS against a SINGLE factorization of K. Column order is fixed
    # (Dead Load, Soil, Surcharge); the auto Self-weight column is appended
    # only when enabled, so existing column indices never shift.
    F_cols = [
        assemble_F(sw_loads_map),
        assemble_F(soil_loads_map),
        assemble_F(surch_loads_map),
    ]
    if auto_sw:
        F_cols.append(assemble_F(sw_auto_loads_map))
    F_static = np.column_stack(F_cols)

    def get_empty_env():
        # The unloaded case has D = 0 identically; recover the result
        # skeleton directly instead of factorizing K for a zero RHS.
        res_empty_det = recover_results({}, np.zeros(NDOF))
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
    
    # --- VEHICLE ANALYSIS SETUP ---
    # Model arrays, span mapping, S-matrices and parent assembly are
    # vehicle-independent; computed once and shared by both vehicles
    # (previously rebuilt inside run_stepping per vehicle).
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
    el_L = np.zeros(n_elems, dtype=np.float64)
    el_T = np.zeros((n_elems, 6, 6), dtype=np.float64)
    el_k_local = np.zeros((n_elems, 6, 6), dtype=np.float64)
    el_dof_indices = np.zeros((n_elems, 6), dtype=np.int32)
    el_E = np.zeros(n_elems, dtype=np.float64)
    el_G = np.zeros(n_elems, dtype=np.float64)
    el_eff_shape = np.zeros(n_elems, dtype=np.int32)
    el_v_type = np.zeros(n_elems, dtype=np.int32)
    el_eff_vals = np.zeros((n_elems, 3), dtype=np.float64)
    el_b_eff = np.zeros(n_elems, dtype=np.float64)
    el_As_avg = np.zeros(n_elems, dtype=np.float64)
    # Sub-element start offset within the parent member: the recovery
    # kernels include a point load sitting exactly at a sub-element start
    # in that start point's V/N only when the start is parent-INTERIOR
    # (loaded-side value at shared mesh sections; member ends keep the
    # historical convention the reaction envelopes rely on).
    el_loc_off = np.zeros(n_elems, dtype=np.float64)

    for k in range(n_elems):
        el_obj = elem_objects[k]
        el_L[k] = el_obj.L
        el_T[k] = el_obj.T
        el_k_local[k] = el_obj.k_local
        el_E[k] = el_obj.E
        el_G[k] = el_obj.G_val
        el_eff_shape[k] = el_obj.eff_shape
        el_v_type[k] = el_obj.v_type
        el_eff_vals[k, :] = el_obj.eff_vals
        el_b_eff[k] = el_obj.b_eff
        el_As_avg[k] = el_obj.As_avg
        el_loc_off[k] = elems_base[k].get('local_offset', 0.0)
        ni, nj = elems_base[k]['nodes']
        idx_i, idx_j = node_map[ni]*3, node_map[nj]*3
        el_dof_indices[k] = [idx_i, idx_i+1, idx_i+2, idx_j, idx_j+1, idx_j+2]

    mesh_sz = params.get('mesh_size', 0.5)
    max_len = np.max(el_L) if len(el_L) > 0 else 1.0
    n_pts_kernel = max(3, int(max_len / mesh_sz) + 1)

    S_matrices = kernels.jit_precompute_stress_recovery(n_elems, n_pts_kernel, el_L, el_k_local)

    # --- BATCHED STEP RECOVERY (fast path) ---
    # The batched kernel evaluates every element on the uniform
    # n_pts_kernel grid. That matches the legacy per-element grid only
    # when max(3, int(L/mesh)+1) is the same for all elements (always
    # true for mesh sub-elements, which are at most mesh_size long);
    # otherwise fall back to the legacy per-step Python recovery.
    legacy_pts_match = all(
        max(3, int(el_L[k] / mesh_sz) + 1) == n_pts_kernel for k in range(n_elems)
    )
    use_batch_recovery = USE_BATCH_STEP_RECOVERY and legacy_pts_match and n_elems > 0

    parent_assembly = []
    if use_batch_recovery:
        parent_groups = {}
        for k, el_data in enumerate(elems_base):
            parent_groups.setdefault(el_data['parent'], []).append(k)
        for pid, indices in parent_groups.items():
            indices.sort(key=lambda k: elems_base[k]['local_offset'])
            offsets = [elems_base[k]['local_offset'] for k in indices]
            x_agg = np.concatenate([
                np.linspace(0.0, el_L[k], n_pts_kernel) + off
                for k, off in zip(indices, offsets)
            ])
            first, last = indices[0], indices[-1]
            ni_id = elems_base[first]['nodes'][0]
            nj_id = elems_base[last]['nodes'][1]
            parent_assembly.append({
                'pid': pid,
                'indices': np.array(indices, dtype=np.int64),
                'x': x_agg,
                'L': float(sum(el_L[k] for k in indices)),
                'cx': elem_objects[first].cx,
                'cy': elem_objects[first].cy,
                'ni': nodes[ni_id], 'nj': nodes[nj_id],
                'ni_id': ni_id, 'nj_id': nj_id,
            })

    def build_vehicle_runs(vehicle_key):
        """One run per travel direction: step positions and axle data.

        RHS assembly and solving happen separately so that all runs (and the
        static cases) can share a single factorization of K.
        """
        v_loads_raw = np.array(params[vehicle_key]['loads']) * 9.81
        if len(v_loads_raw) == 0:
            return []
        v_dists_raw = np.cumsum(params[vehicle_key]['spacing'])
        veh_dir = params.get('vehicle_direction', 'Forward')
        directions_to_run = ['Forward', 'Reverse'] if veh_dir == 'Both' else [veh_dir]
        max_d = max(v_dists_raw) if len(v_dists_raw) > 0 else 0
        step_val = params.get('step_size', 0.5)

        runs = []
        for current_dir in directions_to_run:
            if current_dir == 'Forward':
                x_steps = np.arange(-max_d-1.0, total_structure_len + max_d + 1.0, step_val)
                v_dists_run = v_dists_raw
            else:
                x_steps = np.arange(total_structure_len + max_d + 1.0, -max_d-1.0, -step_val)
                v_dists_run = -v_dists_raw
            runs.append({
                'dir': current_dir,
                'x_steps': x_steps,
                'v_loads_raw': v_loads_raw,
                'v_dists_run': v_dists_run,
            })
        return runs

    runs_A = build_vehicle_runs('vehicle')
    runs_B = build_vehicle_runs('vehicleB')
    all_runs = runs_A + runs_B

    # --- TRAFFIC UDL (fladelast) SEGMENT INFLUENCE CASES ---
    # Adverse-only application (EN 1991-2:2003, 4.3.2(1)(b)) requires the
    # effect of each loaded deck segment separately: one RHS column per deck
    # sub-element, fully loaded with the UDL line load. The adverse envelope
    # is then the sum of only-positive (resp. only-negative) contributions
    # per result point, exact at sub-element resolution.
    udl_q_line = data_mod.udl_line_load(params)
    udl_seg_maps = []
    if udl_q_line > 0.0:
        for (_, seg_L, el_idx) in sp_elems_info:
            udl_seg_maps.append({el_idx: [{
                'type': 'distributed_trapezoid', 'is_gravity': True,
                'params': [udl_q_line, udl_q_line, 0.0, seg_L],
            }]})
    n_udl = len(udl_seg_maps)
    F_udl = (np.column_stack([assemble_F(m) for m in udl_seg_maps])
             if n_udl else np.zeros((NDOF, 0)))

    # --- SINGLE FACTORIZATION FOR ALL LOAD CASES ---
    # The static cases, the UDL segment cases and every vehicle step share
    # one np.linalg.solve call (one O(n^3) factorization, all RHS at O(n^2)
    # each). For very large models/step counts the vehicle RHS blocks are
    # built and solved per run in chunks instead, trading the extra
    # factorizations for memory.
    n_static = F_static.shape[1]
    total_rhs = n_static + n_udl + sum(len(r['x_steps']) for r in all_runs)
    single_solve = (total_rhs <= MAX_SINGLE_SOLVE_RHS
                    and NDOF * total_rhs <= MAX_SINGLE_SOLVE_ELEMENTS)

    try:
        if single_solve and all_runs:
            F_blocks = [F_static, F_udl]
            for r in all_runs:
                F_blocks.append(kernels.jit_build_batch_F(
                    NDOF, len(r['x_steps']), r['x_steps'],
                    r['v_loads_raw'], r['v_dists_run'],
                    sp_start_x, sp_lens, sp_el_indices,
                    el_L, el_T, el_dof_indices,
                    el_E, el_G, el_eff_shape, el_v_type, el_eff_vals, el_b_eff, el_As_avg,
                ))
            D_all = np.linalg.solve(K_glob, np.concatenate(F_blocks, axis=1))
            D_static = D_all[:, :n_static]
            D_udl = D_all[:, n_static:n_static + n_udl]
            col = n_static + n_udl
            for r in all_runs:
                w = len(r['x_steps'])
                r['D'] = D_all[:, col:col+w]
                col += w
        else:
            # Fallback: static + UDL solves now; vehicle runs solved per
            # chunk in process_vehicle_runs (no precomputed 'D' on the runs).
            D_combined = np.linalg.solve(
                K_glob, np.concatenate([F_static, F_udl], axis=1))
            D_static = D_combined[:, :n_static]
            D_udl = D_combined[:, n_static:]
    except np.linalg.LinAlgError:
        raise ValueError("Structural Instability Detected: The model is insufficiently constrained (Mechanism). Please check boundary conditions.")

    res_sw = aggregate_member_results(recover_results(sw_loads_map, D_static[:, 0]), sw_global_loads)
    res_soil = aggregate_member_results(recover_results(soil_loads_map, D_static[:, 1]), soil_global_loads)
    res_surch = aggregate_member_results(recover_results(surch_loads_map, D_static[:, 2]), surch_global_loads)
    # Auto self-weight occupies static column 3 when enabled. No global
    # load override - its arrows are not drawn - so the real per-sub-element
    # loads attach to the result; the equilibrium total uses the dedicated
    # per-parent map built above.
    res_sw_auto = (aggregate_member_results(recover_results(sw_auto_loads_map, D_static[:, 3]))
                   if auto_sw else {})

    # --- GLOBAL EQUILIBRIUM CHECK (static load cases) ---
    # Applied totals are computed from the parent-level load definitions by
    # simple statics (total = average intensity x loaded length, resolved to
    # global axes); reactions are the boundary-spring forces (-k*D at the
    # restrained DOFs). Comparing the two verifies the consistent-load
    # assembly, splitting and solve end to end for each case.
    parent_orient = {}
    for k, el_data in enumerate(elems_base):
        pid = el_data['parent']
        if pid not in parent_orient:
            parent_orient[pid] = (elem_objects[k].cx, elem_objects[k].cy)

    def case_applied_totals(global_loads_map):
        # Besides the net sums, accumulate the positive and negative
        # contributions per axis separately so consumers can show that
        # loads were applied even when opposing definitions (e.g. mirrored
        # earth pressure on both walls) cancel in the net.
        ax = ay = 0.0
        ax_pos = ax_neg = ay_pos = ay_neg = 0.0
        for pid, loads in global_loads_map.items():
            cx_p, cy_p = parent_orient.get(pid, (1.0, 0.0))
            for ld in loads:
                prm = ld['params']
                if ld['type'] == 'point':
                    Q = prm[0]
                elif ld['type'] == 'distributed_trapezoid':
                    # params = [q_start, q_end, x_start, loaded_length]
                    # (the convention of split_distributed_trapezoid_for_
                    # sub_element; x_start is 0 in all global maps today).
                    Q = (prm[0] + prm[1]) / 2.0 * prm[3]
                else:
                    continue
                if ld.get('is_gravity'):
                    # Positive gravity load values act vertically downward.
                    dx, dy = 0.0, -Q
                else:
                    # Positive transverse load values act in local -y
                    # (the same sense gravity takes on a horizontal member).
                    dx, dy = Q * cy_p, Q * (-cx_p)
                ax += dx
                ay += dy
                if dx >= 0.0: ax_pos += dx
                else: ax_neg += dx
                if dy >= 0.0: ay_pos += dy
                else: ay_neg += dy
        return ax, ay, ax_pos, ax_neg, ay_pos, ay_neg

    def case_spring_reactions(D_col):
        # Force exerted by the supports on the structure: -k * d.
        rx = ry = 0.0
        for nid, kvec in restraints.items():
            base = node_map[nid] * 3
            if kvec[0]: rx -= kvec[0] * D_col[base]
            if kvec[1]: ry -= kvec[1] * D_col[base + 1]
        return rx, ry

    equilibrium = {}
    eq_cases = [
        ('Dead Load', sw_global_loads, 0),
        ('Soil', soil_global_loads, 1),
        ('Surcharge', surch_global_loads, 2),
    ]
    if auto_sw:
        eq_cases.append(('Self-weight', sw_auto_global_loads, 3))
    for case_name, loads_map_g, col in eq_cases:
        a_x, a_y, a_x_pos, a_x_neg, a_y_pos, a_y_neg = case_applied_totals(loads_map_g)
        r_x, r_y = case_spring_reactions(D_static[:, col])
        equilibrium[case_name] = {
            'applied_x': a_x, 'applied_y': a_y,
            'applied_x_pos': a_x_pos, 'applied_x_neg': a_x_neg,
            'applied_y_pos': a_y_pos, 'applied_y_neg': a_y_neg,
            'reactions_x': r_x, 'reactions_y': r_y,
            'residual_x': a_x + r_x, 'residual_y': a_y + r_y,
            # Opposing definitions (e.g. mirrored earth pressure on both
            # walls) cancel in the global sums; this flag lets consumers
            # tell "loads cancel" apart from "no loads defined".
            'has_loads': bool(loads_map_g),
        }

    # --- TRAFFIC UDL ADVERSE ENVELOPE ---
    # Static (full deck) application: per result point and force component,
    # sum only the adverse segment contributions (positive ones for the max
    # envelope, negative ones for the min envelope), per EN 1991-2:2003,
    # 4.3.2(1)(b). The point-wise patterned nature means the UDL has no
    # single applied total, so it is excluded from the equilibrium table.
    udl_env = {}
    udl_comp_keys = (('M', 'M_max', 'M_min'), ('V', 'V_max', 'V_min'),
                     ('N', 'N_max', 'N_min'),
                     ('def_x', 'def_x_max', 'def_x_min'),
                     ('def_y', 'def_y_max', 'def_y_min'))
    # Per-parent adverse prefix sums over the deck segments (ordered along
    # the deck). For a contiguous excluded window the adverse sum of the
    # remaining segments is prefix[first_inside] + (total - prefix[last_
    # inside+1]) per side - O(1) numpy adds per step for the step coupling.
    udl_prefix = {}     # pid -> {(comp, 'max'|'min'): (n_seg+1, n_pts) array}
    udl_seg_bounds = [] # (deck_start, deck_end) per segment
    if n_udl and use_batch_recovery:
        # --- BATCHED UDL SEGMENT RECOVERY (fast path) ---
        # One kernel call per chunk replaces the per-segment Python
        # detailed recovery below, which dominated large fine-mesh
        # analyses (one full element loop per deck sub-element). Gated by
        # the same grid-match condition as the batched step recovery: the
        # envelope skeleton and prefix arrays are sized on the legacy
        # per-element grids.
        udl_env = get_empty_env()
        udl_seg_bounds = [(s, s + L) for (s, L, _k) in sp_elems_info]
        seg_el_idx_arr = np.array([info[2] for info in sp_elems_info], dtype=np.int64)
        seg_fef = np.zeros((n_udl, 6))
        seg_q_t = np.zeros(n_udl)
        seg_q_a = np.zeros(n_udl)
        for i in range(n_udl):
            el_k = int(seg_el_idx_arr[i])
            el_obj = elem_objects[el_k]
            # Same decomposition + FEF as the legacy path uses per segment.
            seg_fef[i] = get_local_fixed_end_forces_for_load(
                el_obj, udl_seg_maps[i][el_k][0])
            seg_q_t[i] = udl_q_line * el_obj.cx
            seg_q_a[i] = udl_q_line * el_obj.cy

        comp_col = {'M': 0, 'V': 1, 'N': 2, 'def_x': 3, 'def_y': 4}
        parent_stacks = {
            pa['pid']: {c[0]: np.empty((n_udl, len(pa['indices']) * n_pts_kernel))
                        for c in udl_comp_keys}
            for pa in parent_assembly
        }
        # Chunked: the raw kernel output is n_segs x n_elems x n_pts x 5,
        # which at viaduct scale would rival the dense K in memory.
        UDL_CHUNK = 256
        for c0 in range(0, n_udl, UDL_CHUNK):
            c1 = min(c0 + UDL_CHUNK, n_udl)
            seg_out = np.empty((c1 - c0, n_elems, n_pts_kernel, 5))
            kernels.jit_udl_segment_recovery_batch(
                c1 - c0, n_elems, n_pts_kernel,
                seg_el_idx_arr[c0:c1], seg_fef[c0:c1],
                seg_q_t[c0:c1], seg_q_a[c0:c1],
                D_udl[:, c0:c1], el_dof_indices, el_T, el_L,
                S_matrices, seg_out)
            for pa in parent_assembly:
                block = seg_out[:, pa['indices'], :, :]
                stacks = parent_stacks[pa['pid']]
                for base_key, col in comp_col.items():
                    stacks[base_key][c0:c1] = block[:, :, :, col].reshape(c1 - c0, -1)

        for pa in parent_assembly:
            pid = pa['pid']
            target = udl_env.get(pid)
            udl_prefix[pid] = {}
            for base_key, max_key, min_key in udl_comp_keys:
                stack = parent_stacks[pid][base_key]
                pos = np.maximum(stack, 0.0)
                neg = np.minimum(stack, 0.0)
                if target is not None:
                    target[max_key] = target[max_key] + pos.sum(axis=0)
                    target[min_key] = target[min_key] + neg.sum(axis=0)
                pre_max = np.zeros((n_udl + 1, stack.shape[1]))
                pre_min = np.zeros((n_udl + 1, stack.shape[1]))
                np.cumsum(pos, axis=0, out=pre_max[1:])
                np.cumsum(neg, axis=0, out=pre_min[1:])
                udl_prefix[pid][(base_key, 'max')] = pre_max
                udl_prefix[pid][(base_key, 'min')] = pre_min
    elif n_udl:
        # --- LEGACY PER-SEGMENT RECOVERY (uneven element grids) ---
        udl_env = get_empty_env()
        udl_seg_bounds = [(s, s + L) for (s, L, _k) in sp_elems_info]
        seg_parent_fields = {}  # pid -> comp -> list of per-segment arrays
        for i, seg_map in enumerate(udl_seg_maps):
            detailed = get_detailed_results_optimized(
                elem_objects, elems_base, nodes, D_udl[:, i], seg_map,
                params.get('mesh_size', 0.5), node_map)
            agg = aggregate_member_results(detailed)
            for pid, dat in agg.items():
                target = udl_env.get(pid)
                if target is None:
                    continue
                store = seg_parent_fields.setdefault(pid, {c[0]: [] for c in udl_comp_keys})
                for base_key, max_key, min_key in udl_comp_keys:
                    arr = dat[base_key]
                    target[max_key] = target[max_key] + np.maximum(arr, 0.0)
                    target[min_key] = target[min_key] + np.minimum(arr, 0.0)
                    store[base_key].append(arr)

        for pid, comps in seg_parent_fields.items():
            udl_prefix[pid] = {}
            for base_key, arrs in comps.items():
                stack = np.stack(arrs)  # (n_seg, n_pts)
                pos = np.maximum(stack, 0.0)
                neg = np.minimum(stack, 0.0)
                n_pts_p = stack.shape[1]
                pre_max = np.zeros((stack.shape[0] + 1, n_pts_p))
                pre_min = np.zeros((stack.shape[0] + 1, n_pts_p))
                np.cumsum(pos, axis=0, out=pre_max[1:])
                np.cumsum(neg, axis=0, out=pre_min[1:])
                udl_prefix[pid][(base_key, 'max')] = pre_max
                udl_prefix[pid][(base_key, 'min')] = pre_min

    # Per-step coupling data for the step viewer: adverse UDL fields with
    # the moving window excluded, plus the loaded ranges for visualization.
    udl_gap = max(0.0, float(params.get('udl_gap', 10.0) or 0.0))
    udl_moving = params.get('udl_mode', 'Moving') != 'Static'
    udl_footprint = bool(params.get('udl_footprint', False))
    seg_starts_arr = np.array([b[0] for b in udl_seg_bounds]) if n_udl else np.zeros(0)
    seg_ends_arr = np.array([b[1] for b in udl_seg_bounds]) if n_udl else np.zeros(0)

    # Deck extent per span parent (global deck coordinates), for converting
    # loaded ranges to parent-local coordinates for visualization.
    parent_deck_extent = {}
    for (seg_s, seg_L, el_k) in sp_elems_info:
        pid = elems_base[el_k]['parent']
        lo, hi = parent_deck_extent.get(pid, (seg_s, seg_s + seg_L))
        parent_deck_extent[pid] = (min(lo, seg_s), max(hi, seg_s + seg_L))

    def attach_step_udl(step_res, axle_lo, axle_hi):
        """Attach per-step adverse UDL fields and loaded ranges to a step.

        The window [axle_lo - gap, axle_hi + gap] is excluded, snapped
        inward to segment boundaries: a segment is excluded only when it
        lies fully inside the window, which is conservative for the load
        effects. The axle extent spans ALL axles of the run, including
        axles still off the deck while the vehicle enters or leaves, so
        the vehicle footprint and its clear distance are honoured near
        the abutments. With the footprint option, or in static
        application, no window is excluded.
        """
        if not n_udl:
            return
        if udl_moving and not udl_footprint:
            w_lo = axle_lo - udl_gap
            w_hi = axle_hi + udl_gap
            # Segments fully inside the window: start >= w_lo and end <= w_hi.
            i0 = int(np.searchsorted(seg_starts_arr, w_lo, side='left'))
            i1 = int(np.searchsorted(seg_ends_arr, w_hi, side='right'))
            if i1 <= i0:
                i0 = i1 = 0  # window narrower than any segment: nothing excluded
        else:
            i0 = i1 = 0

        # Loaded deck ranges (global coordinates).
        ranges = []
        if i1 > i0:
            if i0 > 0:
                ranges.append((float(seg_starts_arr[0]), float(seg_ends_arr[i0 - 1])))
            if i1 < len(seg_starts_arr):
                ranges.append((float(seg_starts_arr[i1]), float(seg_ends_arr[-1])))
        elif len(seg_starts_arr):
            ranges = [(float(seg_starts_arr[0]), float(seg_ends_arr[-1]))]

        for pid, prefixes in udl_prefix.items():
            res_p = step_res.get(pid)
            if res_p is None:
                continue
            for base_key, _mx, _mn in udl_comp_keys:
                pre_max = prefixes[(base_key, 'max')]
                pre_min = prefixes[(base_key, 'min')]
                res_p[f'{base_key}_udl_max'] = pre_max[i0] + (pre_max[-1] - pre_max[i1])
                res_p[f'{base_key}_udl_min'] = pre_min[i0] + (pre_min[-1] - pre_min[i1])
            extent = parent_deck_extent.get(pid)
            if extent is not None:
                p_lo, p_hi = extent
                local = []
                for r_lo, r_hi in ranges:
                    a = max(r_lo, p_lo); b = min(r_hi, p_hi)
                    if b > a + 1e-9:
                        local.append((a - p_lo, b - p_lo))
                res_p['udl_loaded_ranges'] = local

    def process_vehicle_runs(runs, env_to_fill):
        steps_out = {'Forward': [], 'Reverse': []}
        if not runs:
            return steps_out
        env_results_accum = np.zeros((n_elems, n_pts_kernel, 10))
        is_first_run = True

        for r in runs:
            current_dir = r['dir']
            x_steps = r['x_steps']
            v_loads_raw = r['v_loads_raw']
            v_dists_run = r['v_dists_run']
            # Full vehicle extent per step (all axles, on deck or not) for
            # the moving UDL window; reverse runs carry negated distances.
            v_d_min = float(np.min(v_dists_run))
            v_d_max = float(np.max(v_dists_run))
            total_steps = len(x_steps)
            v_steps_res_list = []
            CHUNK_SIZE = 2000

            for start_idx in range(0, total_steps, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, total_steps)
                x_chunk = x_steps[start_idx:end_idx]
                n_chunk = len(x_chunk)
                is_init_chunk = (is_first_run and start_idx == 0)

                if 'D' in r:
                    # Pre-solved against the shared single factorization.
                    D_chunk = r['D'][:, start_idx:end_idx]
                else:
                    # Memory fallback: build and solve this chunk only.
                    F_chunk = kernels.jit_build_batch_F(
                        NDOF, n_chunk, x_chunk, v_loads_raw, v_dists_run,
                        sp_start_x, sp_lens, sp_el_indices,
                        el_L, el_T, el_dof_indices,
                        el_E, el_G, el_eff_shape, el_v_type, el_eff_vals, el_b_eff, el_As_avg,
                    )
                    try:
                        D_chunk = np.linalg.solve(K_glob, F_chunk)
                    except np.linalg.LinAlgError:
                        raise ValueError("Structural Instability Detected during Vehicle Analysis.")
                
                kernels.jit_envelope_batch_parallel(
                    n_chunk, n_elems, n_pts_kernel,
                    x_chunk, v_loads_raw, v_dists_run,
                    sp_start_x, sp_lens, sp_el_indices,
                    D_chunk, el_dof_indices, el_T, el_L,
                    el_E, el_G, el_eff_shape, el_v_type, el_eff_vals, el_b_eff, el_As_avg,
                    el_loc_off,
                    S_matrices,
                    env_results_accum,
                    is_init_chunk
                )

                step_out = None
                if use_batch_recovery:
                    step_out = np.zeros((n_chunk, n_elems, n_pts_kernel, 5))
                    kernels.jit_step_recovery_batch(
                        n_chunk, n_elems, n_pts_kernel,
                        x_chunk, v_loads_raw, v_dists_run,
                        sp_start_x, sp_lens, sp_el_indices,
                        D_chunk, el_dof_indices, el_T, el_L,
                        el_E, el_G, el_eff_shape, el_v_type, el_eff_vals, el_b_eff, el_As_avg,
                        el_loc_off,
                        S_matrices,
                        step_out,
                    )

                # Sorted sub-element start positions for O(log n) span lookup
                # per axle; the original linear scan dominated step mapping.
                n_sp = len(sp_elems_info)
                for i_local in range(n_chunk):
                    x_front = x_chunk[i_local]
                    step_loads_map = {}
                    parent_loads = {}
                    has_loads = False
                    for ax_i, d in enumerate(v_dists_run):
                        ax_x = x_front - d
                        guess = int(np.searchsorted(sp_start_x, ax_x, side='right')) - 1
                        if guess < 0: guess = 0
                        if guess >= n_sp: guess = n_sp - 1
                        # The original predicate stays authoritative for the
                        # boundary tolerance; a load within tolerance just
                        # below a boundary belongs to the next sub-element,
                        # so test the neighbour as well.
                        for info_idx in (guess, guess + 1):
                            if info_idx >= n_sp: break
                            start_x, L_span, el_idx = sp_elems_info[info_idx]
                            is_last = info_idx == n_sp - 1
                            if point_load_belongs_to_sub_element(ax_x, start_x, start_x + L_span, is_last, 1e-9):
                                local_x = min(max(ax_x - start_x, 0.0), L_span)
                                P_val = v_loads_raw[ax_i]
                                if use_batch_recovery:
                                    pid = elems_base[el_idx]['parent']
                                    parent_x = elems_base[el_idx]['local_offset'] + local_x
                                    parent_loads.setdefault(pid, []).append(
                                        {'type': 'point', 'is_gravity': True, 'params': [P_val, parent_x]})
                                else:
                                    step_loads_map.setdefault(el_idx, []).append(
                                        {'type': 'point', 'is_gravity': True, 'params': [P_val, local_x]})
                                has_loads = True
                                break
                    if not has_loads:
                        continue

                    if use_batch_recovery:
                        # Slice the batched kernel output into the same
                        # per-parent dict structure aggregate_member_results
                        # produces. The static metadata (x grid, geometry) is
                        # shared across steps; only the value arrays are new.
                        out_s = step_out[i_local]
                        step_res = {}
                        for pa in parent_assembly:
                            block = out_s[pa['indices']]
                            M_agg = block[:, :, 0].reshape(-1)
                            V_agg = block[:, :, 1].reshape(-1)
                            N_agg = block[:, :, 2].reshape(-1)
                            loads_p = parent_loads.get(pa['pid'], [])
                            if len(loads_p) > 1:
                                loads_p.sort(key=lambda l: l['params'][1])
                            step_res[pa['pid']] = {
                                'x': pa['x'],
                                'M': M_agg, 'V': V_agg, 'N': N_agg,
                                'def_x': block[:, :, 3].reshape(-1),
                                'def_y': block[:, :, 4].reshape(-1),
                                'L': pa['L'], 'cx': pa['cx'], 'cy': pa['cy'],
                                'ni': pa['ni'], 'nj': pa['nj'],
                                'ni_id': pa['ni_id'], 'nj_id': pa['nj_id'],
                                'loads': loads_p,
                                # Synthesized end forces (unused by step
                                # consumers): f_start = [N0, V0, M0] with
                                # N(0) = -N0; f_end from element equilibrium.
                                # M fields are sagging-positive; nodal end
                                # moments stay in the nodal convention.
                                'f_start_local': np.array([-N_agg[0], V_agg[0], -M_agg[0]]),
                                'f_end_local': np.array([N_agg[-1], -V_agg[-1], M_agg[-1]]),
                            }
                        attach_step_udl(step_res, x_front - v_d_max, x_front - v_d_min)
                        v_steps_res_list.append({'x': x_front, 'res': step_res})
                    else:
                        D_step = D_chunk[:, i_local]
                        raw_step = get_detailed_results_optimized(elem_objects, elems_base, nodes, D_step, step_loads_map, params.get('mesh_size', 0.5), node_map)
                        agg_step = aggregate_member_results(raw_step)
                        attach_step_udl(agg_step, x_front - v_d_max, x_front - v_d_min)
                        v_steps_res_list.append({'x': x_front, 'res': agg_step})
            
            steps_out[current_dir] = v_steps_res_list
            is_first_run = False 

        if env_to_fill:
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
                
                for (idx, offset, _) in parts:
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

    steps_A = process_vehicle_runs(runs_A, veh_env_A)
    steps_B = process_vehicle_runs(runs_B, veh_env_B)

    return {
        'Dead Load': res_sw,
        # Auto-computed structural self-weight; empty dict when the toggle
        # is off (the case then has no presence in the UI/report/export).
        'Self-weight': res_sw_auto,
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
        # Per-member phi (eid -> value). Empty dict means calc_phi is uniform.
        'Phi Members': phi_members,
        # Traffic UDL adverse envelope (empty dict when inactive) and the
        # line load it was computed with [kN/m].
        'Traffic UDL': udl_env,
        'udl_line_load': udl_q_line,
        # Global equilibrium check per static case (applied vs reactions).
        'Equilibrium': equilibrium,
        # Nodes carrying boundary springs. Reaction summaries must use this
        # instead of guessing from node ids/coordinates: in Frame mode a
        # zero-height wall places its restraint at the top node (200+i),
        # which coordinate/id heuristics misclassify as a free node.
        'Restrained Nodes': sorted(restraints.keys()),
        'Reactions': calculate_reactions(nodes, res_sw),
        # Process-unique identity for the combine_results memo: a raw
        # result reused from the solver cache keeps its token, a fresh
        # analysis gets a new one.
        'combine_token': next(_combine_token_counter),
    }, nodes, model_props, 0

def coupled_vehicle_udl_envelope(steps, f_veh_map, f_veh_default, f_udl):
    """Exact coupled traffic envelope: vehicle steps + accompanying UDL.

    Each step carries the adverse min/max application of the UDL over the
    deck MINUS that position's clear-distance window (attach_step_udl).
    Enveloping the factored sum over all steps therefore gives the exact
    coupled traffic term per EN 1991-2 - the vehicle replaces the UDL
    locally - instead of the former independent-superposition bound
    (vehicle envelope + full adverse UDL), which added the UDL share of
    the window region that a physical vehicle displaces. With the static
    or footprint application the per-step UDL fields equal the full
    adverse envelope, so the coupling reduces to the former bound.

    The factors enter per step, so the governing position follows the
    selected result mode's vehicle/UDL factor ratio (incl. per-member phi).
    Returns {eid: {10 envelope keys}}; empty when there are no steps.
    """
    out = {}
    if not steps:
        return out
    eids = set()
    for s in steps:
        eids.update(s['res'].keys())
    for eid in eids:
        rows = [s['res'][eid] for s in steps if eid in s['res']]
        if not rows:
            continue
        f_v = f_veh_map.get(eid, f_veh_default)
        entry = {}
        for base in ('M', 'V', 'N', 'def_x', 'def_y'):
            veh_stack = np.stack([r[base] for r in rows])
            zero = np.zeros_like(rows[0][base])
            u_max = np.stack([r.get(f'{base}_udl_max', zero) for r in rows])
            u_min = np.stack([r.get(f'{base}_udl_min', zero) for r in rows])
            entry[f'{base}_max'] = (veh_stack * f_v + u_max * f_udl).max(axis=0)
            entry[f'{base}_min'] = (veh_stack * f_v + u_min * f_udl).min(axis=0)
        out[eid] = entry
    return out


# ==========================================
# COMBINATION MEMO
# ==========================================
# Combining is pure in (raw result identity, factor params, result mode),
# yet it ran from scratch on every Streamlit rerun, on every result-mode
# switch back and forth, per report chapter (the report generator added a
# local cache for exactly this) and up to six times per Excel-export
# click. The memo lives in session state like the raw solver cache;
# memoized results are shared READ-ONLY objects (the established cache
# contract - no consumer mutates combined results).
_COMBINE_MEMO_KEY = '_bricos_combine_memo'
_COMBINE_MEMO_MAX_ENTRIES = 24

# Every params key combine_results reads. test_combine_memo verifies this
# list against the implementation's source, so a new params read cannot
# silently bypass the memo key.
COMBINE_PARAM_KEYS = (
    'KFI', 'gamma_g', 'gamma_j', 'gamma_veh', 'gamma_vehB', 'gamma_udl',
    'phi_mode', 'phi', 'phi_sls_mode', 'phi_sls',
    'sls_g', 'sls_j', 'sls_veh', 'sls_vehB', 'sls_udl',
    'combine_surcharge_vehicle',
)


def _combine_memo_lookup_key(raw_res, params, result_mode):
    """Value-based memo key, or None when the memo must be bypassed
    (raw results from before the token existed)."""
    token = raw_res.get('combine_token') if raw_res else None
    if token is None:
        return None
    factors = []
    for k in COMBINE_PARAM_KEYS:
        v = params.get(k)
        if not isinstance(v, (int, float, str, bool, type(None))):
            v = repr(v)
        factors.append(v)
    return (token, result_mode, tuple(factors))


def combine_results(raw_res, params, result_mode="Design (ULS)"):
    key = _combine_memo_lookup_key(raw_res, params, result_mode)
    if key is None:
        return _combine_results_impl(raw_res, params, result_mode)
    try:
        _ensure_cache_version()
        memo = st.session_state.setdefault(_COMBINE_MEMO_KEY, {})
    except Exception:
        return _combine_results_impl(raw_res, params, result_mode)
    hit = memo.get(key)
    if hit is not None:
        return hit
    out = _combine_results_impl(raw_res, params, result_mode)
    while len(memo) >= _COMBINE_MEMO_MAX_ENTRIES:
        memo.pop(next(iter(memo)))
    memo[key] = out
    return out


def _combine_results_impl(raw_res, params, result_mode):
    KFI = params.get('KFI', 1.0)
    gamma_g = params.get('gamma_g', 1.0)
    gamma_j = params.get('gamma_j', 1.0)
    gamma_vA = params.get('gamma_veh', 1.0)
    gamma_vB = params.get('gamma_vehB', 1.0)
    # Per-member phi (calculated span-based or manual per-span values are
    # embedded by the solver). Empty means the single phi applies uniformly.
    phi_members = raw_res.get('Phi Members') or {}
    if params.get('phi_mode') == 'Calculate':
        phi = raw_res['phi_calc']
    else:
        # Manual: per-span values arrive via Phi Members; the global manual
        # value is the uniform fallback.
        phi = params.get('phi', 1.0)
        if phi_members:
            phi = max(phi_members.values())

    # Phi treatment in the Characteristic (SLS) result mode:
    # 'Same' (= ULS), 'Reduced' (1+(phi-1)/2, Vejledning 5.4.2) or
    # 'Manual' (user-defined uniform value).
    sls_mode = params.get('phi_sls_mode', 'Same')
    sls_active = result_mode == "Characteristic (SLS)" and sls_mode in ('Reduced', 'Manual')
    phi_sls_manual = params.get('phi_sls', 1.0)

    # Any mode other than ULS/SLS is the unfactored combination: every load
    # at factor 1.0 with no dynamic factor. (The legacy mode name
    # "Characteristic (No Dynamic Factor)" maps here as well.)
    is_unfactored = result_mode not in ("Design (ULS)", "Characteristic (SLS)")

    def phi_for(eid):
        if is_unfactored:
            return 1.0
        if sls_active and sls_mode == 'Manual':
            return phi_sls_manual
        p = phi_members.get(eid, phi)
        if sls_active:
            return 1.0 + (p - 1.0) / 2.0
        return p

    phi_repr = phi_for('__representative__')

    if result_mode == "Design (ULS)":
        f_sw = KFI * gamma_g
        f_soil = KFI * gamma_j
        f_vehA_base = KFI * gamma_vA
        f_vehB_base = KFI * gamma_vB
        f_surch = KFI * gamma_vA
        # Traffic UDL: KFI * gamma (Vejledning Fig. B3.1; 0.56 in LC1).
        # The traffic UDL intensity includes the dynamic increment
        # (DK NA A.2.3.2), so phi is never applied to it.
        f_udl = KFI * params.get('gamma_udl', 0.56)
    elif result_mode == "Characteristic (SLS)":
        # Dedicated SLS factor set (Vejledning Fig. B3.2, characteristic
        # combination). KFI is not applied in SLS. The surcharge follows
        # the vehicle A factor, mirroring the ULS convention.
        f_sw = params.get('sls_g', 1.0)
        f_soil = params.get('sls_j', 1.0)
        f_vehA_base = params.get('sls_veh', 1.0)
        f_vehB_base = params.get('sls_vehB', 1.0)
        f_surch = params.get('sls_veh', 1.0)
        f_udl = params.get('sls_udl', 0.40)
    else:
        # Unfactored: every load with factor 1.0 and no dynamic factor.
        f_sw = 1.0; f_soil = 1.0; f_vehA_base = 1.0; f_vehB_base = 1.0; f_surch = 1.0
        f_udl = 1.0

    # Representative (governing) factors, plus per-member maps used by the
    # step viewer when phi varies per member.
    f_vehA = f_vehA_base * phi_repr
    f_vehB = f_vehB_base * phi_repr
    f_vehA_map = {}
    f_vehB_map = {}

    def factor_res(res, f):
        out = {}
        if not res: return out
        for eid, dat in res.items():
            new_loads = []
            for l in dat.get('loads', []):
                new_l = {**l, 'params': list(l['params'])}
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

    out_sw = factor_res(raw_res['Dead Load'], f_sw)
    # Auto self-weight is a permanent action: same factor as dead load
    # (KFI*gamma_g in ULS, sls_g in SLS, 1.0 unfactored). Empty when the
    # toggle is off, contributing nothing to the permanent total.
    out_swa = factor_res(raw_res.get('Self-weight') or {}, f_sw)
    out_soil = factor_res(raw_res['Soil'], f_soil)
    out_surch = factor_res(raw_res['Surcharge'], f_surch)
    
    raw_env_A = raw_res['Vehicle Envelope A']
    raw_env_B = raw_res['Vehicle Envelope B']
    out_veh_env = {}
    env_keys_10 = ('M_max', 'M_min', 'V_max', 'V_min', 'N_max', 'N_min',
                   'def_x_max', 'def_x_min', 'def_y_max', 'def_y_min')
    # Per-vehicle factored envelopes, kept separate for the coupled traffic
    # term (cross terms coupledA + envB / envA + coupledB).
    out_envA_fac = {}
    out_envB_fac = {}

    if raw_env_A:
        for eid in raw_env_A:
            dA = raw_env_A[eid]
            dB = raw_env_B.get(eid)
            if dB is None:
                # An element missing from envelope B must contribute nothing.
                # Falling back to dA here would double-count vehicle A.
                z = np.zeros_like(dA['M_max'])
                dB = {'M_max': z, 'M_min': z, 'V_max': z, 'V_min': z,
                      'N_max': z, 'N_min': z,
                      'def_x_max': z, 'def_x_min': z,
                      'def_y_max': z, 'def_y_min': z}
            base = dA['base']
            phi_e = phi_for(eid)
            fA_e = f_vehA_base * phi_e
            fB_e = f_vehB_base * phi_e
            f_vehA_map[eid] = fA_e
            f_vehB_map[eid] = fB_e
            envA_f = {k: dA[k] * fA_e for k in env_keys_10}
            envB_f = {k: dB[k] * fB_e for k in env_keys_10}
            out_envA_fac[eid] = envA_f
            out_envB_fac[eid] = envB_f
            out_veh_env[eid] = {
                **base,
                **{k: envA_f[k] + envB_f[k] for k in env_keys_10},
            }

    # Traffic UDL: factored full adverse envelope. Per Vejledning Fig. B3.1
    # the fladelast acts in the same combination as the vehicles; in the
    # Total Envelope it enters through the exact coupling below (and as the
    # vehicle-absent alternative), not as a blanket additive term.
    out_udl = {}
    raw_udl = raw_res.get('Traffic UDL') or {}
    for eid, dat in raw_udl.items():
        out_udl[eid] = {
            **dat['base'],
            **{k: dat[k] * f_udl for k in env_keys_10},
        }

    # Exact coupled traffic term: each vehicle's steps combined with the
    # adverse UDL outside that step's window (factored per result mode),
    # enveloped per point. The vehicle-absent situation (full adverse UDL
    # alone) is added as an explicit alternative in the total assembly -
    # the step set never contains it (the window clips the deck even with
    # the vehicle off-deck). With both vehicles defined, the UDL couples
    # with each vehicle's own steps and the cross-vehicle combination
    # remains an independent superposition (slightly conservative).
    use_coupled = bool(raw_udl)
    coupled_A = {}
    coupled_B = {}
    if use_coupled:
        steps_A_all = ((raw_res.get('Vehicle Steps A') or [])
                       + (raw_res.get('Vehicle Steps A_Rev') or []))
        steps_B_all = ((raw_res.get('Vehicle Steps B') or [])
                       + (raw_res.get('Vehicle Steps B_Rev') or []))
        coupled_A = coupled_vehicle_udl_envelope(steps_A_all, f_vehA_map, f_vehA, f_udl)
        coupled_B = coupled_vehicle_udl_envelope(steps_B_all, f_vehB_map, f_vehB, f_udl)

    out_total = {}
    all_ids = set(out_sw.keys()) | set(out_swa.keys()) | set(out_veh_env.keys()) | set(out_soil.keys()) | set(out_surch.keys()) | set(out_udl.keys())
    combine_surcharge_vehicle = params.get('combine_surcharge_vehicle', False)

    for eid in all_ids:
        ref = out_sw.get(eid) or out_swa.get(eid) or out_soil.get(eid) or out_veh_env.get(eid) or out_surch.get(eid)
        if not ref: continue
        n_p = len(ref['M_max'])
        z = np.zeros(n_p)
        _zero = {'M_max':z, 'M_min':z, 'V_max':z, 'V_min':z, 'N_max':z, 'N_min':z, 'def_x_max':z, 'def_x_min':z, 'def_y_max':z, 'def_y_min':z}
        sw = out_sw.get(eid, _zero)
        swa = out_swa.get(eid, _zero)
        sl = out_soil.get(eid, _zero)
        ve = out_veh_env.get(eid, _zero)
        su = out_surch.get(eid, _zero)
        ud = out_udl.get(eid, _zero)

        # Permanent term = dead load + auto self-weight + soil (all share
        # the gamma_g/sls_g factor, so they simply sum).
        M_perm_max = sw['M_max'] + swa['M_max'] + sl['M_max']
        M_perm_min = sw['M_min'] + swa['M_min'] + sl['M_min']
        V_perm_max = sw['V_max'] + swa['V_max'] + sl['V_max']
        V_perm_min = sw['V_min'] + swa['V_min'] + sl['V_min']
        N_perm_max = sw['N_max'] + swa['N_max'] + sl['N_max']
        N_perm_min = sw['N_min'] + swa['N_min'] + sl['N_min']
        def_x_perm_max = sw['def_x_max'] + swa['def_x_max'] + sl['def_x_max']
        def_x_perm_min = sw['def_x_min'] + swa['def_x_min'] + sl['def_x_min']
        def_y_perm_max = sw['def_y_max'] + swa['def_y_max'] + sl['def_y_max']
        def_y_perm_min = sw['def_y_min'] + swa['def_y_min'] + sl['def_y_min']

        # The traffic term is vehicle + accompanying UDL (same load
        # combination per Vejledning Fig. B3.1). With the UDL active it is
        # the EXACT coupled envelope: per point the worst of (each
        # vehicle's coupled steps + the other vehicle's envelope) and the
        # vehicle-absent full adverse UDL. Without UDL it reduces to the
        # plain vehicle envelope. The surcharge interaction setting
        # governs only the surcharge.
        cA = coupled_A.get(eid)
        cB = coupled_B.get(eid)
        eAf = out_envA_fac.get(eid)
        eBf = out_envB_fac.get(eid)
        traf = {}
        for k in env_keys_10:
            if use_coupled and (cA is not None or cB is not None):
                cands = [ud[k]]
                if cA is not None:
                    cands.append(cA[k] + (eBf[k] if eBf is not None else 0.0))
                if cB is not None:
                    cands.append((eAf[k] if eAf is not None else 0.0) + cB[k])
                if k.endswith('_max'):
                    traf[k] = np.maximum.reduce(cands)
                else:
                    traf[k] = np.minimum.reduce(cands)
            else:
                traf[k] = ve[k] + ud[k]

        if combine_surcharge_vehicle:
            M_tot_max = M_perm_max + su['M_max'] + traf['M_max']
            M_tot_min = M_perm_min + su['M_min'] + traf['M_min']
            V_tot_max = V_perm_max + su['V_max'] + traf['V_max']
            V_tot_min = V_perm_min + su['V_min'] + traf['V_min']
            N_tot_max = N_perm_max + su['N_max'] + traf['N_max']
            N_tot_min = N_perm_min + su['N_min'] + traf['N_min']
            def_x_tot_max = def_x_perm_max + su['def_x_max'] + traf['def_x_max']
            def_x_tot_min = def_x_perm_min + su['def_x_min'] + traf['def_x_min']
            def_y_tot_max = def_y_perm_max + su['def_y_max'] + traf['def_y_max']
            def_y_tot_min = def_y_perm_min + su['def_y_min'] + traf['def_y_min']
        else:
            M_tot_max = np.maximum(M_perm_max + traf['M_max'], M_perm_max + su['M_max'])
            M_tot_min = np.minimum(M_perm_min + traf['M_min'], M_perm_min + su['M_min'])
            V_tot_max = np.maximum(V_perm_max + traf['V_max'], V_perm_max + su['V_max'])
            V_tot_min = np.minimum(V_perm_min + traf['V_min'], V_perm_min + su['V_min'])
            N_tot_max = np.maximum(N_perm_max + traf['N_max'], N_perm_max + su['N_max'])
            N_tot_min = np.minimum(N_perm_min + traf['N_min'], N_perm_min + su['N_min'])
            def_x_tot_max = np.maximum(def_x_perm_max + traf['def_x_max'], def_x_perm_max + su['def_x_max'])
            def_x_tot_min = np.minimum(def_x_perm_min + traf['def_x_min'], def_x_perm_min + su['def_x_min'])
            def_y_tot_max = np.maximum(def_y_perm_max + traf['def_y_max'], def_y_perm_max + su['def_y_max'])
            def_y_tot_min = np.minimum(def_y_perm_min + traf['def_y_min'], def_y_perm_min + su['def_y_min'])

        out_total[eid] = {
            **sw, 
            'M_max': M_tot_max, 'M_min': M_tot_min,
            'V_max': V_tot_max, 'V_min': V_tot_min,
            'N_max': N_tot_max, 'N_min': N_tot_min,
            'def_x_max': def_x_tot_max, 'def_x_min': def_x_tot_min,
            'def_y_max': def_y_tot_max, 'def_y_min': def_y_tot_min
        }
    
    return {
        'Dead Load': out_sw, 'Self-weight': out_swa,
        'Soil': out_soil, 'Surcharge': out_surch,
        'Vehicle Envelope': out_veh_env, 'Traffic UDL': out_udl,
        'Total Envelope': out_total,
        'f_udl': f_udl,
        'Vehicle Steps A': raw_res.get('Vehicle Steps A', []),
        'Vehicle Steps A_Rev': raw_res.get('Vehicle Steps A_Rev', []),
        'Vehicle Steps B': raw_res.get('Vehicle Steps B', []),
        'Vehicle Steps B_Rev': raw_res.get('Vehicle Steps B_Rev', []),
        'f_vehA': f_vehA, 'f_vehB': f_vehB,
        # Per-member vehicle factors (eid -> factor incl. phi). Used by the
        # step viewer so step results scale consistently with the envelopes
        # when phi varies per member.
        'f_vehA_map': f_vehA_map, 'f_vehB_map': f_vehB_map,
        'phi_calc': phi, 'phi_log': raw_res['phi_log'],
        'Phi Members': phi_members,
        'phi_sls_mode_applied': sls_mode if sls_active else 'Same',
        'Restrained Nodes': raw_res.get('Restrained Nodes', []),
        'Reactions': raw_res.get('Reactions', {})
    }
