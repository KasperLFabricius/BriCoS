import numpy as np
from numba import jit, prange

# ==========================================
# NUMBA KERNELS (MATH & PHYSICS ENGINE)
# ==========================================

@jit(nopython=True, cache=True)
def jit_beam_matrices(xi, yi, xj, yj, E, I, A):
    dx = xj - xi
    dy = yj - yi
    L = np.sqrt(dx**2 + dy**2)
    if L < 1e-6: L = 1e-6
    
    cx = dx / L
    cy = dy / L
    
    k = E * I / L**3
    w1 = E * A / L
    w2 = 12 * k
    w3 = 6 * L * k
    w4 = 4 * L**2 * k
    w5 = 2 * L**2 * k
    
    k_local = np.array([
        [w1, 0, 0, -w1, 0, 0],
        [0, w2, w3, 0, -w2, w3],
        [0, w3, w4, 0, -w3, w5],
        [-w1, 0, 0, w1, 0, 0],
        [0, -w2, -w3, 0, w2, -w3],
        [0, w3, w5, 0, -w3, w4]
    ])
    
    T = np.zeros((6, 6))
    T[0,0] = cx;  T[0,1] = cy
    T[1,0] = -cy; T[1,1] = cx
    T[2,2] = 1.0
    T[3,3] = cx;  T[3,4] = cy
    T[4,3] = -cy; T[4,4] = cx
    T[5,5] = 1.0
    
    k_global = T.T @ k_local @ T
    
    return k_local, k_global, T, L, cx, cy

@jit(nopython=True, cache=True)
def jit_fef_point(P, a, L):
    f = np.zeros(6)
    b = L - a
    if 0 <= a <= L:
        m1 = P * a * b**2 / L**2
        m2 = -P * a**2 * b / L**2
        R1 = P * b**2 * (3*a + b) / L**3
        R2 = P * a**2 * (a + 3*b) / L**3
        f[1] = R1
        f[2] = m1
        f[4] = R2
        f[5] = m2
    return f

@jit(nopython=True, cache=True)
def jit_fef_trapezoid(q_s, q_e, h_s, h_e, L):
    f = np.zeros(6)
    if h_e > L: h_e = L
    if h_s < 0: h_s = 0
    if h_s >= h_e: return f

    num_int = 20
    step = (h_e - h_s) / (num_int - 1)
    
    F_equiv = np.zeros(4) 
    
    for i in range(num_int):
        x = h_s + i * step
        weight = 1.0 if (i == 0 or i == num_int - 1) else 2.0
        wx = q_s + (q_e - q_s) * (x - h_s) / (h_e - h_s)
        
        xi = x/L
        xi2 = xi*xi
        xi3 = xi2*xi
        
        n_v1 = 1.0 - 3.0*xi2 + 2.0*xi3
        n_m1 = x * (1.0 - xi)**2
        n_v2 = 3.0*xi2 - 2.0*xi3
        n_m2 = x * (xi2 - xi)
        
        val = weight * wx * step / 2.0
        
        F_equiv[0] += val * n_v1
        F_equiv[1] += val * n_m1
        F_equiv[2] += val * n_v2
        F_equiv[3] += val * n_m2
        
    f[1] = F_equiv[0]
    f[2] = F_equiv[1]
    f[4] = F_equiv[2]
    f[5] = F_equiv[3]
    return f

@jit(nopython=True, cache=True)
def jit_internal_forces(L, f_start, num_pts, load_data):
    x_vals = np.linspace(0, L, num_pts)
    M_vals = np.zeros(num_pts)
    V_vals = np.zeros(num_pts)
    N_vals = np.zeros(num_pts)
    
    N0, V0, M0 = f_start[0], f_start[1], f_start[2]
    
    for i in range(num_pts):
        x = x_vals[i]
        Mx = M0 - V0 * x
        Vx = V0
        Nx = -N0 
        
        for j in range(len(load_data)):
            l_type = int(load_data[j, 0])
            
            if l_type == 2:
                q_s, q_e, h_s, h_e = load_data[j, 1], load_data[j, 2], load_data[j, 3], load_data[j, 4]
                if x > h_s:
                    lim = x if x < h_e else h_e
                    len_eff = lim - h_s
                    
                    if abs(h_e - h_s) > 1e-9:
                        q_at_lim = q_s + (q_e - q_s) * (len_eff) / (h_e - h_s)
                        F_part = (q_s + q_at_lim) / 2.0 * len_eff
                        Vx -= F_part
                        
                        denom = q_s + q_at_lim
                        if abs(denom) > 1e-9:
                            d_c = (len_eff / 3.0) * (2.0 * q_s + q_at_lim) / denom
                        else:
                            d_c = 0.0
                        dist_arm = (x - h_s) - d_c
                        Mx += F_part * dist_arm

            elif l_type == 1:
                P, a = load_data[j, 1], load_data[j, 2]
                if x > a:
                    Vx -= P
                    Mx += P * (x - a)
                    
        M_vals[i] = Mx
        V_vals[i] = Vx
        N_vals[i] = Nx
        
    return x_vals, M_vals, V_vals, N_vals

@jit(nopython=True, cache=True)
def jit_disp_shape(u_vec, L, num_pts):
    x = np.linspace(0, L, num_pts)
    xi = x / L
    u_x = u_vec[0] * (1 - xi) + u_vec[3] * xi
    phi1 = 1 - 3*xi**2 + 2*xi**3
    phi2 = x * (1 - xi)**2
    phi3 = 3*xi**2 - 2*xi**3
    phi4 = x * (xi**2 - xi)
    u_y = phi1*u_vec[1] + phi2*u_vec[2] + phi3*u_vec[4] + phi4*u_vec[5]
    return u_x, u_y

@jit(nopython=True, cache=True)
def jit_annotation_solver(data_arr):
    N = len(data_arr)
    iterations = 30
    for _ in range(iterations):
        moves_x = np.zeros(N)
        moves_y = np.zeros(N)
        collision = False
        for i in range(N):
            for j in range(i + 1, N):
                ax, ay, aw, ah = data_arr[i, 0], data_arr[i, 1], data_arr[i, 2], data_arr[i, 3]
                bx, by, bw, bh = data_arr[j, 0], data_arr[j, 1], data_arr[j, 2], data_arr[j, 3]
                
                dx = abs(ax - bx)
                dy = abs(ay - by)
                min_dx = (aw + bw) * 0.75
                min_dy = (ah + bh) * 0.75
                
                if dx < min_dx and dy < min_dy:
                    collision = True
                    overlap_x = min_dx - dx
                    overlap_y = min_dy - dy
                    a_perp_x, a_perp_y = data_arr[i, 4], data_arr[i, 5]
                    is_vert_a = abs(a_perp_y) < abs(a_perp_x)
                    
                    if is_vert_a:
                        direction = 1.0 if ax > bx else -1.0
                        moves_x[i] += overlap_x * 0.5 * direction
                        moves_x[j] -= overlap_x * 0.5 * direction
                    else:
                        direction = 1.0 if ay > by else -1.0
                        moves_y[i] += overlap_y * 0.5 * direction
                        moves_y[j] -= overlap_y * 0.5 * direction
        if not collision: break
        for k in range(N):
            data_arr[k, 0] += moves_x[k]
            data_arr[k, 1] += moves_y[k]
    return data_arr

# ----------------------------------------
# OPTIMIZED BATCH KERNELS (STRATEGY 1-5)
# ----------------------------------------

@jit(nopython=True, cache=True)
def jit_build_batch_F(NDOF, n_steps, x_steps, v_loads, v_dists, sp_start_x, sp_lens, sp_el_indices, el_L, el_T, el_dof_indices):
    F_all = np.zeros((NDOF, n_steps))
    num_axles = len(v_loads)
    num_spans = len(sp_start_x)
    
    for i in range(n_steps):
        x_front = x_steps[i]
        for j in range(num_axles):
            load_x = x_front - v_dists[j]
            for k in range(num_spans):
                s_x = sp_start_x[k]
                e_x = s_x + sp_lens[k]
                if s_x <= load_x <= e_x:
                    local_x = load_x - s_x
                    el_idx = sp_el_indices[k]
                    f_local = jit_fef_point(v_loads[j], local_x, el_L[el_idx])
                    T = el_T[el_idx]
                    f_global_vec = T.T @ f_local
                    dofs = el_dof_indices[el_idx]
                    for d in range(6):
                        F_all[dofs[d], i] -= f_global_vec[d]
                    break
    return F_all

@jit(nopython=True, cache=True)
def jit_precompute_stress_recovery(n_elems, n_pts, el_L, el_k_local):
    S = np.zeros((n_elems, n_pts, 5, 6))
    for e in range(n_elems):
        L = el_L[e]
        k = el_k_local[e]
        dx = L / (n_pts - 1)
        for p in range(n_pts):
            x = p * dx
            xi = x / L
            for j in range(6):
                S[e, p, 0, j] = k[2, j] - x * k[1, j] # M
                S[e, p, 1, j] = k[1, j]               # V
                S[e, p, 2, j] = -k[0, j]              # N
            S[e, p, 3, 0] = 1.0 - xi
            S[e, p, 3, 3] = xi
            xi2 = xi*xi
            xi3 = xi2*xi
            phi1 = 1.0 - 3.0*xi2 + 2.0*xi3
            phi2 = x * (1.0 - xi)**2
            phi3 = 3.0*xi2 - 2.0*xi3
            phi4 = x * (xi2 - xi)
            S[e, p, 4, 1] = phi1
            S[e, p, 4, 2] = phi2
            S[e, p, 4, 4] = phi3
            S[e, p, 4, 5] = phi4
    return S

@jit(nopython=True, parallel=True, cache=True)
def jit_envelope_batch_parallel(
    n_steps, n_elems, n_pts, 
    x_steps, v_loads, v_dists, 
    sp_start_x, sp_lens, sp_el_indices, 
    D_all, el_dof_indices, el_T, el_L, 
    S_matrices,
    res_accum, # Accumulator updated in-place
    is_init    # Reset flag
):
    num_axles = len(v_loads)
    num_spans = len(sp_start_x)
    
    for e in prange(n_elems):
        if is_init:
            for p in range(n_pts):
                res_accum[e, p, 0] = -1e15; res_accum[e, p, 1] = 1e15 # M
                res_accum[e, p, 2] = -1e15; res_accum[e, p, 3] = 1e15 # V
                res_accum[e, p, 4] = -1e15; res_accum[e, p, 5] = 1e15 # N
                res_accum[e, p, 6] = -1e15; res_accum[e, p, 7] = 1e15 # dx
                res_accum[e, p, 8] = -1e15; res_accum[e, p, 9] = 1e15 # dy

        L_el = el_L[e]
        T = el_T[e]
        cx, cy = T[0,0], T[0,1]
        dofs = el_dof_indices[e]
        
        span_idx = -1
        span_offset = 0.0
        for sp in range(num_spans):
            if sp_el_indices[sp] == e:
                span_idx = sp
                span_offset = sp_start_x[sp]
                break
        
        s_start = 0.0
        s_end = 0.0
        has_span = False
        if span_idx >= 0:
            has_span = True
            s_start = span_offset
            s_end = s_start + sp_lens[span_idx]

        dx_step = L_el / (n_pts - 1)
        
        d_glob = np.zeros(6)
        d_loc = np.zeros(6)
        fef_total = np.zeros(6)

        for s in range(n_steps):
            x_front = x_steps[s]
            for k in range(6): d_glob[k] = D_all[dofs[k], s]
            
            for r in range(6):
                val = 0.0
                for c in range(6): val += T[r, c] * d_glob[c]
                d_loc[r] = val
            
            fef_total[:] = 0.0
            if has_span:
                for ax in range(num_axles):
                    load_x_glob = x_front - v_dists[ax]
                    if s_start <= load_x_glob <= s_end:
                        local_x = load_x_glob - s_start
                        fef_total += jit_fef_point(v_loads[ax], local_x, L_el)
            
            FEF_N, FEF_V, FEF_M = fef_total[0], fef_total[1], fef_total[2]
            
            for p in range(n_pts):
                x = p * dx_step
                M_hom = 0.0; V_hom = 0.0; N_hom = 0.0; ux_loc = 0.0; uy_loc = 0.0
                
                for j in range(6):
                    dj = d_loc[j]
                    M_hom += S_matrices[e, p, 0, j] * dj
                    V_hom += S_matrices[e, p, 1, j] * dj
                    N_hom += S_matrices[e, p, 2, j] * dj
                    ux_loc += S_matrices[e, p, 3, j] * dj
                    uy_loc += S_matrices[e, p, 4, j] * dj

                Mx = M_hom + (FEF_M - FEF_V * x)
                Vx = V_hom + FEF_V
                Nx = N_hom - FEF_N 
                
                if has_span:
                    for ax in range(num_axles):
                        load_x_glob = x_front - v_dists[ax]
                        if s_start <= load_x_glob <= s_end:
                            a = load_x_glob - s_start
                            if x > a:
                                P = v_loads[ax]
                                Vx -= P
                                Mx += P * (x - a)
                
                def_x_glob = cx * ux_loc - cy * uy_loc
                def_y_glob = cy * ux_loc + cx * uy_loc
                
                if Mx > res_accum[e, p, 0]: res_accum[e, p, 0] = Mx
                if Mx < res_accum[e, p, 1]: res_accum[e, p, 1] = Mx
                if Vx > res_accum[e, p, 2]: res_accum[e, p, 2] = Vx
                if Vx < res_accum[e, p, 3]: res_accum[e, p, 3] = Vx
                if Nx > res_accum[e, p, 4]: res_accum[e, p, 4] = Nx
                if Nx < res_accum[e, p, 5]: res_accum[e, p, 5] = Nx
                if def_x_glob > res_accum[e, p, 6]: res_accum[e, p, 6] = def_x_glob
                if def_x_glob < res_accum[e, p, 7]: res_accum[e, p, 7] = def_x_glob
                if def_y_glob > res_accum[e, p, 8]: res_accum[e, p, 8] = def_y_glob
                if def_y_glob < res_accum[e, p, 9]: res_accum[e, p, 9] = def_y_glob